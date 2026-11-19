"""One shadow observation, fixed layout. No ROS, torch, pickle or motor commands."""
import numpy as np

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')  # top, left, right
ARM = ('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate')
NAMES = [s + '_' + k for s in ('left', 'right') for k in ARM + ('gripper',)]
SHAPE = (480, 640, 3)  # HWC RGB uint8
MIME = 'application/x-aloha-pillow-observation-v1'
STATE_BYTES = 14 * 4  # little-endian float32, followed by the three RGB frames
SIZE = STATE_BYTES + len(CAMERAS) * int(np.prod(SHAPE))


def pack(state, frames):
    q = np.asarray(state, dtype='<f4')
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError('Expected 14 finite measured state values')
    if len(frames) != 3 or any(f.dtype != np.uint8 or f.shape != SHAPE for f in frames):
        raise ValueError('Expected top, left wrist, right wrist RGB uint8, each (480,640,3)')
    return b''.join([q.tobytes()] + [f.tobytes(order='C') for f in frames])


def unpack(body):
    if len(body) != SIZE:
        raise ValueError('Incomplete or wrong-sized observation')
    q = np.frombuffer(body, '<f4', 14).copy()
    if not np.isfinite(q).all():
        raise ValueError('Non-finite robot state')
    return q, np.frombuffer(body, np.uint8, offset=STATE_BYTES).reshape(3, *SHAPE).copy()
