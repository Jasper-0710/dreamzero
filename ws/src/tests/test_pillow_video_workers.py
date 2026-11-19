"""True subprocess tests for GPU assignment, failure cleanup and chronological merge (no CUDA)."""
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ws.src.evaluation import export_pillow_video as video
from ws.src.evaluation.pillow_video_workers import assignments, run_workers, video_devices


def fake_part(path, jobs, layout, ckpt, verified, opts, *, load_slots, ready):
    device = os.environ['CUDA_VISIBLE_DEVICES']
    with load_slots:
        ep, starts = jobs[0]
        assert starts == [video.SOURCE91[int(device)-1]] and ep.source == 91
    ready.wait(timeout=45)
    with path.open('xb') as f:
        for j in range(len(video.OFFSETS)):
            np.save(f, np.full((24,32,3), 32*int(device) + j, dtype=np.uint8), allow_pickle=False)
    return False


def crash_before_load(*, load_slots, ready):
    if os.environ['CUDA_VISIBLE_DEVICES'] == '2':
        os._exit(7)  # model sudden death, e.g. SIGKILL/OOM; healthy peers cannot reach barrier
    ready.wait(timeout=30)
    return False


class WorkersTest(unittest.TestCase):
    def test_balanced_no_idle_no_duplicates(self):
        self.assertEqual(assignments(range(7), ('5','1','4')), [('5',[0,1,2]), ('1',[3,4]), ('4',[5,6])])
        with self.assertRaisesRegex(ValueError,'不足'):
            assignments(range(5), tuple('123456'))
        for env in ({'CUDA_VISIBLE_DEVICES':'1,1'}, {'CUDA_VISIBLE_DEVICES':'1-6'}, {'CUDA_VISIBLE_DEVICES':'-1'},
                    {'CUDA_VISIBLE_DEVICES':'1,2','WORLD_SIZE':'2'}, {'CUDA_VISIBLE_DEVICES':'1','LOAD_TRT_ENGINE':'x'}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
                video_devices(False)
        with patch.dict(os.environ, clear=True):
            self.assertEqual(video_devices(True), ())

    def test_gpu_failure_does_not_strand_peers(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU 2 .*7'):
            run_workers(crash_before_load, [('1',()), ('2',())], 1)

    def test_six_spawned_processes_merge_correct_order_and_pts(self):
        import av
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, run, full, out = [root/k for k in ('train','run','full','result')]
            for d in (train,run,full):
                d.mkdir()
            opts = video.parse_args(['--run',str(run),'--train-data',str(train),'--output',str(out)])
            data = SimpleNamespace(full=full, state={}, episodes=[SimpleNamespace(source=91, source_start=0, length=561, id=91)])
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'1,2,3,4,5,6'}, clear=False), \
                    patch.object(video, 'Holdout', return_value=data), \
                    patch.object(video, 'recipe', return_value=SimpleNamespace(protected_dirs=())), \
                    patch.object(video, 'check_video_recipe'), patch.object(video, 'render_part', fake_part), \
                    redirect_stdout(StringIO()):
                video.main(opts)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '1,2,3,4,5,6')
            meta = json.loads((out/'windows.json').read_text())
            self.assertEqual([(s['device'], s['anchors']) for s in meta['gpu_assignments']],
                             [(str(i+1), [a]) for i,a in enumerate(video.SOURCE91)])
            self.assertEqual([w['first_output_frame'] for w in meta['windows']], list(range(0,54,9)))
            self.assertTrue((out/'index.html').is_file())
            with av.open(str(out/'comparison.mp4')) as clip:
                decoded = list(clip.decode(video=0))
            self.assertEqual([f.pts*f.time_base for f in decoded], [Fraction(3*i,50) for i in range(54)])
            for i, f in enumerate(decoded):
                self.assertLess(abs(float(f.to_ndarray(format='rgb24').mean()) - (32*(1+i//9)+i%9)), 4)
            self.assertFalse(any(root.glob('.pillow-video-*')))
