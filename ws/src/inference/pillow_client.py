#!/usr/bin/env python3
"""Read one recorded frame or synchronized ROS1 feedback; print/log predictions, never drive hardware."""
import argparse
import http.client
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ws.src.inference.pillow_wire import ARM, CAMERAS, MIME, NAMES, SHAPE, pack


def recorded(root, episode, frame):
    import pyarrow.parquet as pq
    from PIL import Image
    info = json.loads((root/'meta/info.json').read_text())
    if info['features']['observation.state']['names'] != NAMES:
        raise ValueError('Unexpected recorded joint order')
    path = root / info['data_path'].format(episode_chunk=episode//info['chunks_size'], episode_index=episode)
    pf = pq.ParquetFile(path)
    cams = ['observation.images.' + k for k in CAMERAS]
    columns = ['frame_index', 'episode_index', 'observation.state'] + cams  # never read an action label
    offset = frame
    for group in range(pf.num_row_groups):
        rows = pf.metadata.row_group(group).num_rows
        if offset >= rows:
            offset -= rows
            continue
        for batch in pf.iter_batches(batch_size=8, row_groups=[group], columns=columns):
            if offset >= len(batch):
                offset -= len(batch)
                continue
            row = batch.slice(offset, 1).to_pylist()[0]
            if (row['episode_index'], row['frame_index']) != (episode, frame):
                raise ValueError('Recording index does not match requested row')
            images = []
            for k in cams:
                with Image.open(io.BytesIO(row[k]['bytes'])) as im:
                    if im.size != (SHAPE[1], SHAPE[0]) or im.mode != 'RGB':
                        raise ValueError('Unexpected recorded image: ' + k)
                    images.append(np.array(im))
            return np.array(row['observation.state'], dtype=np.float32), images, dict(episode=episode, frame=frame)
    raise ValueError('No such frame in ' + str(path))


def measured_arm(msg, normalize):
    # Match real_env.get_qpos (not the actuator-angle read by replay). Refuse an unverified driver layout.
    names = list(msg.name)
    if (len(names) != len(msg.position) or tuple(names[:6]) != ARM or len(names) < 8
            or names[6:8] != ['gripper', 'left_finger']):
        raise ValueError('JointState layout differs from recording; verify first six joints and finger position[7]: ' + str(names))
    return list(msg.position[:6]) + [normalize(msg.position[7])]


def valid_stamps(stamps, now, max_skew, max_age):
    return (len(stamps) == 5 and all(np.isfinite(t) and t > 0 for t in stamps)
            and max(stamps) - min(stamps) <= max_skew and max(stamps) <= now and now-min(stamps) <= max_age)


class RosSource:
    """Subscriptions only. Keep latest aligned bundle, never queue work behind slow inference."""
    def __init__(self, constants, skew_ms, age_ms):
        import message_filters
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image, JointState
        spec = importlib.util.spec_from_file_location('aloha_record_calibration', str(constants.resolve(strict=True)))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # use the same constants.py as the recording on this robot
        self.normalize = mod.PUPPET_GRIPPER_POSITION_NORMALIZE_FN
        self.rospy, self.bridge = rospy, CvBridge()
        self.skew, self.age = skew_ms / 1000., age_ms / 1000.
        rospy.init_node('pillow_shadow_readonly', anonymous=True)
        if rospy.get_param('/use_sim_time', False):
            raise ValueError('Use live ROS time for freshness checks (not /use_sim_time)')
        self.cv, self.latest, self.last_stamp = threading.Condition(), None, 0.
        topics = [('/usb_' + c + '/image_raw', Image) for c in CAMERAS]
        topics += [('/puppet_' + s + '/joint_states', JointState) for s in ('left', 'right')]
        self.subs = [message_filters.Subscriber(k, t, queue_size=1, buff_size=2**21) for k, t in topics]
        self.sync = message_filters.ApproximateTimeSynchronizer(self.subs, queue_size=8, slop=self.skew)
        self.sync.registerCallback(self._receive)

    def _receive(self, *msgs):
        stamps = [m.header.stamp.to_sec() for m in msgs]
        if valid_stamps(stamps, self.rospy.Time.now().to_sec(), self.skew, self.age):
            with self.cv:
                self.latest = (stamps, msgs)
                self.cv.notify_all()

    def read(self):
        deadline = time.monotonic() + 12.
        while not self.rospy.is_shutdown() and time.monotonic() < deadline:
            with self.cv:
                item = self.latest
                self.latest = None
                if item is None:
                    self.cv.wait(.2)
                    continue
            stamps, msgs = item
            if min(stamps) <= self.last_stamp or not valid_stamps(stamps, self.rospy.Time.now().to_sec(), self.skew, self.age):
                continue
            images = [self.bridge.imgmsg_to_cv2(m, desired_encoding='rgb8') for m in msgs[:3]]
            q = [v for m in msgs[3:] for v in measured_arm(m, self.normalize)]
            if not valid_stamps(stamps, self.rospy.Time.now().to_sec(), self.skew, self.age):
                continue
            self.last_stamp = min(stamps)
            return q, images, dict(ros_stamps=stamps, skew_ms=1000*(max(stamps)-min(stamps)))
        raise ValueError('No fresh synchronized observation; check the five ROS topics, header clocks, image size and rates')


def post(port, body, expected):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
    try:
        conn.request('POST', '/predict', body, {'Content-Type': MIME})
        r = conn.getresponse()
        blob = r.read(65536)
        if r.status != 200:
            raise RuntimeError('Inference HTTP %d: %s' % (r.status, blob[:500].decode(errors='replace')))
        out = json.loads(blob)
    finally:
        conn.close()
    act = np.asarray(out['action'])
    if (out['mode'] != 'shadow' or out['checkpoint'] != expected or out['action_names'] != NAMES or out['fps'] != 50
            or act.shape != (24, 14) or not np.isfinite(act).all()):
        raise ValueError('Unexpected model, layout or non-finite prediction')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='mode', required=True)
    rec = s.add_parser('recording', help='同一条真实录制的同一帧，适合测试和计时')
    rec.add_argument('--data', type=Path, default=Path('data/flip_the_pillow'))
    rec.add_argument('--episode', type=int, default=91)
    rec.add_argument('--frame', type=int, default=216)
    ros = s.add_parser('ros', help='只订阅 ROS1 三相机和双从臂反馈，不创建电机发布者')
    ros.add_argument('--constants', type=Path, default=Path('../aloha_record/aloha_scripts/constants.py'))
    ros.add_argument('--max-skew-ms', type=float, default=40.)
    ros.add_argument('--max-age-ms', type=float, default=150.)
    for sub in (rec, ros):
        sub.add_argument('--port', type=int, default=8766)
        sub.add_argument('--count', type=int, default=1)
        sub.add_argument('--expect-checkpoint', default='checkpoint-1600')
        sub.add_argument('--output', type=Path, help='可选：保存每次的输入关节、24×14 动作和计时，不覆盖')
    a = p.parse_args()
    if a.count < 1 or not 1 <= a.port <= 65535 or (a.mode == 'recording' and min(a.episode, a.frame) < 0):
        p.error('Count/port must be positive and episode/frame non-negative')
    src = None
    if a.mode == 'ros':
        if not (0 < a.max_skew_ms <= a.max_age_ms < float('inf')):
            p.error('Require finite 0 < max-skew-ms <= max-age-ms')
        src = RosSource(a.constants, a.max_skew_ms, a.max_age_ms)
    else:
        fixed = recorded(a.data, a.episode, a.frame)
    times, ages = [], []
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
    f = a.output.open('x', encoding='utf-8') if a.output else None
    try:
        for i in range(a.count):
            q, frames, meta = src.read() if src is not None else fixed
            start = time.perf_counter()
            out = post(a.port, pack(q, frames), a.expect_checkpoint)
            ms = (time.perf_counter()-start)*1000
            times.append(ms)
            extra = ''
            if src is not None:
                age = (src.rospy.Time.now().to_sec()-min(meta['ros_stamps']))*1000
                if not np.isfinite(age) or age < 0:
                    raise ValueError('ROS clock changed while waiting for inference')
                ages.append(age)
                meta['obs_age_at_reply_ms'] = age
                extra = ' oldest_obs_at_reply=%.0f ms' % age
            if f:
                f.write(json.dumps(dict(out, source=a.mode, observation=meta, state=list(map(float, q)), rtt_ms=ms), allow_nan=False) + '\n')
                f.flush()
            print('[%d/%d] %s action=(24,14) model=%.0f ms RTT=%.0f ms CUDA peak reserved=%.2f GiB%s' %
                  (i+1, a.count, out['checkpoint'], out['model_ms'], ms, out['cuda_peak_reserved_gib'], extra), flush=True)
        if len(times) > 1:
            print('RTT excluding first request (n=%d): p50=%.0f ms p95=%.0f ms; action horizon=480 ms' %
                  (len(times)-1, *np.percentile(times[1:], [50, 95])))
            if ages:
                print('Oldest ROS observation at reply excluding first: p50=%.0f ms p95=%.0f ms' %
                      tuple(np.percentile(ages[1:], [50, 95])))
    finally:
        if f:
            f.close()


if __name__ == '__main__':
    main()
