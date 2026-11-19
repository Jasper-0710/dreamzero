"""CPU end-to-end comparison with synthetic cameras/actions and an injectable joint policy."""
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, nullcontext, redirect_stdout
from fractions import Fraction
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from ws.src.evaluation import compare_pillow_checkpoints as cmp
from ws.src.tests.test_pillow_open_loop import fixture, commands


class CompareTest(unittest.TestCase):
    def test_schedule_waves_every_item_once_and_nonempty_lanes(self):
        actions = [(SimpleNamespace(id=i), 24*(i+1)) for i in range(10)]
        windows = [(actions[0][0], x) for x in (168,192,216,240,264,288)]
        devices = '123456'
        for steps in ([1200,1600,2000], list(range(1,10))):
            waves = list(cmp.waves(steps, devices, actions, windows))
            for wave in waves:
                self.assertEqual([x.gpu for x in wave], list(devices))
                self.assertTrue(all(x.cost > 0 for x in wave))
            for s in steps:
                lanes = [x for wave in waves for x in wave if x.step == s]
                self.assertEqual(sorted(ep.id for x in lanes for ep,n in x.actions), list(range(10)))
                self.assertEqual(sorted(a for x in lanes for ep,a in x.videos), [a for ep,a in windows])
                if len(steps)==3:
                    self.assertEqual(len(lanes),2)

    def test_missing_step_before_dataset_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'checkpoint-1200').mkdir(); (root/'checkpoint-1600').mkdir()
            o = cmp.parse_args(['--run',tmp,'--steps','1200','1500','--output',str(root/'out')])
            with self.assertRaisesRegex(ValueError, r'缺少 step \[1500\].*1200, 1600'), patch.object(cmp.ev,'Holdout',side_effect=AssertionError):
                cmp.main(o)
            self.assertFalse((root/'out').exists())

    def test_combined_report_matches_serial_actions_and_frame_locked_video(self):
        steps = (1200,1600,2000)
        def act(obs, step):
            a = commands(int(obs['state.left_joint_pos'][0,0])+np.arange(24))
            a[:,0] += (2000-step)/1000
            return {'action.'+k:a[:,s] for k,s in data.action.items()}
        @contextmanager
        def mock_policy(ckpt, verified, seed):
            step = int(ckpt.name.split('-')[-1]); self.assertEqual(seed,1140)
            yield SimpleNamespace(trained_model=SimpleNamespace(action_head=SimpleNamespace(tiled=False))), \
                lambda obs:(SimpleNamespace(act=act(obs,step)), np.full((9,16,16,3), (step-1000)//5, dtype=np.uint8))
        def run_local(render, tasks, parallel_loads):
            self.assertEqual([g for g,_ in tasks],list('123456'))
            return [render(*args, load_slots=nullcontext(), ready=SimpleNamespace(wait=lambda:None)) for _,args in tasks]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp); train, _ = fixture(root); data = cmp.ev.Holdout(train); run=root/'run'; out=root/'out'
            for step in steps:
                (run/f'checkpoint-{step}').mkdir(parents=True)
            o=cmp.parse_args(['--run',str(run),'--train-data',str(train),'--anchors','0','3','--output',str(out)])
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1,2,3,4,5,6'},clear=True), \
                    patch.object(cmp,'recipe',return_value=SimpleNamespace(protected_dirs=())), \
                    patch.object(cmp.video,'check_video_recipe'), patch.object(cmp,'run_workers',side_effect=run_local), \
                    patch.object(cmp,'independent_policy',side_effect=mock_policy), \
                    patch.object(cmp.video,'decode_prediction',side_effect=lambda head,x,tiled:x), \
                    patch.object(cmp.video,'recorded_grid',side_effect=lambda p,obs:obs[cmp.video.CAMERAS[0]][0]):
                o.check_only=True; cmp.main(o); self.assertFalse(out.exists())
                o.check_only=False; cmp.main(o)
                with self.assertRaises(FileExistsError):
                    cmp.main(o)
            reference=root/'serial'; reference.mkdir(); reports=[]
            for step in steps:
                reports.extend(cmp.ev.evaluate_step(step,data,cmp.ev.plan(data.episodes,None), lambda obs:act(obs,step),
                                                    reference,25,baseline=step==steps[0]))
            cmp.ev.save_tables(reference,reports)
            for f in reference.glob('*.csv'):
                self.assertEqual(f.read_bytes(), (out/f.name).read_bytes())
            self.assertEqual(json.loads((out/'windows.json').read_text())['windows'][1]['source_frames'], list(range(3,28,3)))
            self.assertFalse(list(out.rglob('*.npyseq')))
            html=(out/'index.html').read_text(); self.assertIn('src="comparison.mp4"',html); self.assertIn('id="report-data"',html)
            with av.open(str(out/'comparison.mp4')) as container:
                frames=list(container.decode(video=0))
            self.assertEqual([f.pts*f.time_base for f in frames], [Fraction(3*i,50) for i in range(18)])
            for i,f in enumerate(frames):
                x=f.to_ndarray(format='rgb24')
                self.assertEqual(x.shape,(128,32,3))
                # Row-major mosaic: recorded, 1200, 1600, 2000; all cells share this frame's PTS.
                for pix,want in zip((x[40,8],x[40,24],x[88,8],x[88,24]),
                                    ((i//9*3+i%9*3)*3,40,120,200)):
                    self.assertLess(abs(float(pix.mean())-want), 8)

    def test_refuses_different_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,s in enumerate((1200,1600)):
                p=root/f'checkpoint-{s}'; p.mkdir()
                with (p/'window-0.npyseq').open('wb') as f:
                    np.save(f,np.full((2,16,16,3),i,dtype=np.uint8),allow_pickle=False)
            with self.assertRaisesRegex(ValueError, '实录预处理不一致'):
                next(cmp.mosaic_frames(root,(1200,1600),91,[0]))
