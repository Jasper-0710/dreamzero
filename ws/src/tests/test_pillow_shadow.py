"""No CUDA, sockets, ROS drivers or real checkpoint required."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from ws.src.common.pillow import TASK, VIEWS
from ws.src.evaluation.pillow_open_loop import WIDTHS
from ws.src.inference import pillow_client as client
from ws.src.inference import pillow_wire as wire
from ws.src.inference import serve_pillow as serve


def fixture():
    return np.arange(14, dtype='<f4'), [np.full(wire.SHAPE, v, dtype=np.uint8) for v in (10, 20, 30)]


class Shadow(unittest.TestCase):
    def test_wire_and_model(self):
        q, views = fixture()
        payload = wire.pack(q, views)
        self.assertEqual(len(payload), wire.SIZE)
        q2, rgb = wire.unpack(payload)
        np.testing.assert_equal(q, q2)
        slices, start = {}, 0
        for k, w in WIDTHS.items():
            slices[k] = slice(start, start+w)
            start += w
        def predict(obs):
            self.assertEqual(obs['annotation.task'], TASK)
            for j, v in enumerate(VIEWS):
                np.testing.assert_equal(obs['video.'+v.split('.')[-1]], views[j][None])
            for k, s in slices.items():
                np.testing.assert_equal(obs['state.'+k], q[None, s])
            self.assertEqual(len(obs), 3+4+1)
            return SimpleNamespace(act={'action.'+k: np.tile(q[s], (24,1)) for k,s in slices.items()}), None
        act = serve.infer(q2, rgb, serve.Layout(slices,slices), predict)
        np.testing.assert_equal(act, np.tile(q, (24,1)))  # no extra state addition

    def test_reject_wrong_sizes_and_nan(self):
        q, frames = fixture()
        with self.assertRaises(ValueError): wire.pack(q[:12], frames)
        with self.assertRaises(ValueError): wire.pack(q, frames[:2])
        with self.assertRaises(ValueError): wire.pack(q, [frames[0][::2]] + frames[1:])
        with self.assertRaises(ValueError): wire.unpack(b'short')
        body = wire.pack(q, frames)
        with self.assertRaises(ValueError): wire.unpack(np.float32(np.nan).tobytes() + body[4:])

    def test_layout_and_train_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); m = root/'meta'; m.mkdir()
            def dump(name, obj): (m/name).write_text(json.dumps(obj)+'\n')
            dump('tasks.jsonl', {'task': TASK}); dump('embodiment.json', {'embodiment_tag': 'yam'})
            f = {k: {'shape':[14], 'names':wire.NAMES} for k in ('observation.state', 'action')}
            f.update({v:{'shape':wire.SHAPE, 'dtype':'video'} for v in VIEWS})
            dump('info.json', {'fps':50, 'features':f})
            mod = {'video':{v.split('.')[-1]:{'original_key':v} for v in VIEWS}}
            for g, k in [('state','observation.state'), ('action','action')]:
                start = 0; mod[g] = {}
                for key, w in WIDTHS.items():
                    mod[g][key] = dict(absolute=True, start=start, end=start+w, original_key=k)
                    start += w
            dump('modality.json', mod)
            self.assertEqual(serve.train_layout(root).state['right_gripper_pos'], slice(13,14))
            mod['action']['right_joint_pos'], mod['action']['left_joint_pos'] = mod['action']['left_joint_pos'],mod['action']['right_joint_pos']
            dump('modality.json', mod)
            with self.assertRaises(ValueError): serve.train_layout(root)

    def test_finger_not_motor_and_bad_timestamps(self):
        m = SimpleNamespace(name=list(wire.ARM)+['gripper','left_finger','right_finger'], position=[1]*6+[99,.03822,0])
        np.testing.assert_allclose(client.measured_arm(m, lambda x: (x-.01844)/(.058-.01844)), [1]*6+[.5])
        m.name[7] = 'other'
        with self.assertRaises(ValueError): client.measured_arm(m, lambda x:x)
        valid = client.valid_stamps
        self.assertTrue(valid([.98,.99,1.,.99,.98], 1.02,.04,.15))
        for t in (0.,.8,1.2,float('nan')):
            self.assertFalse(valid([1.,1.,1.,1.,t], 1.02,.04,.15))
        self.assertFalse(valid([.5]*5, 1.,.04,.15))

    def test_http_request_to_prediction(self):
        q, frames = fixture(); body = wire.pack(q, frames)
        run = Mock(return_value=(np.tile(q, (24,1)), 400., 50.))
        handler = serve.handler(run, 'checkpoint-1600')
        # Drive BaseHTTPRequestHandler's real parser/response into a fake socket.
        def request(path, content, ctype):
            conn = Mock(); out = io.BytesIO(); conn.sendall.side_effect = out.write
            packet = ('POST '+path+' HTTP/1.0\r\nContent-Type: '+ctype+'\r\nContent-Length: '+str(len(content))+'\r\n\r\n').encode() + content
            conn.makefile.return_value = io.BytesIO(packet)
            with patch.object(handler,'log_message'):
                handler(conn, ('127.0.0.1', 1), object())
            return out.getvalue().split(b'\r\n\r\n', 1)
        head, data = request('/predict', body, wire.MIME)
        self.assertIn(b' 200 ', head)
        res = json.loads(data)
        self.assertEqual((res['checkpoint'],res['mode'],res['action_names']), ('checkpoint-1600','shadow',wire.NAMES))
        np.testing.assert_equal(res['action'], np.tile(q, (24,1)))
        conn = Mock()
        conn.getresponse.return_value.status = 200
        conn.getresponse.return_value.read.return_value = data
        with patch.object(client.http.client, 'HTTPConnection', return_value=conn):
            self.assertEqual(client.post(8766, body, 'checkpoint-1600')['action'], res['action'])
            with self.assertRaises(ValueError): client.post(8766, body, 'checkpoint-1200')
        for content, mime in [(b'x',wire.MIME),(body,'application/json')]:
            self.assertIn(b' 400 ', request('/predict',content,mime)[0])
        self.assertIn(b' 404 ', request('/other',body,wire.MIME)[0]); run.assert_called_once()

    def test_recorded_row_across_groups(self):
        from PIL import Image
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'meta').mkdir()
            (root/'meta/info.json').write_text(json.dumps(dict(features={'observation.state':dict(names=wire.NAMES)},
                    chunks_size=1000, data_path='ep{episode_index}.parquet')))
            rgb = []
            for color in [(255,0,0),(0,255,0),(0,0,255)]:
                b = io.BytesIO(); Image.new('RGB',(640,480),color).save(b, format='JPEG'); rgb.append({'bytes':b.getvalue()})
            n = 25
            cols = {'frame_index':list(range(n)), 'episode_index':[91]*n,'observation.state':[[float(i)]*14 for i in range(n)]}
            cols.update({'observation.images.'+cam:[rgb[j]]*n for j,cam in enumerate(wire.CAMERAS)})
            pq.write_table(pa.table(cols),root/'ep91.parquet',row_group_size=12)
            q, frames, meta = client.recorded(root,91,21)
            self.assertEqual(meta['frame'],21); np.testing.assert_equal(q, [21.]*14)
            self.assertEqual([np.argmax(f[0,0]) for f in frames], [0,1,2])
            with self.assertRaises(ValueError): client.recorded(root,91,n)


if __name__ == '__main__':
    unittest.main()
