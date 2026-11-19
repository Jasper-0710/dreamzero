"""CPU tests. No robot, network, pretrained weights or private recordings required."""
import csv
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from fractions import Fraction
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from safetensors.torch import save_file

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import TASK, VIEWS, WAN_FILES
from ws.src.evaluation import eval_flip_pillow as ev
from ws.src.evaluation.pillow_eval_policy import recipe, restore, weights_at
from ws.src.evaluation.pillow_open_loop import (
    FPS, H, WIDTHS, Holdout, action_chunk, anchor_images, crossings, match_events, metrics, read_rows,
)


def dump(p, x):
    p.write_text(json.dumps(x))


def dump_lines(p, xs):
    p.write_text(''.join(json.dumps(x)+'\n' for x in xs))


def commands(t):
    t = np.asarray(t)
    x = (t[:, None] + np.arange(14, dtype=np.float32)) / 10
    x[:, 6] = (t % 24 >= 10)
    x[:, 13] = (t % 24 < 12)
    return x.astype('float32')


def video(p, n, fps=FPS):
    with av.open(str(p), 'w') as container:
        stream = container.add_stream('mpeg4', rate=fps)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for t in range(n):
            f = av.VideoFrame.from_ndarray(np.full((16, 16, 3), 3*t, dtype='uint8'), format='rgb24')
            f.pts, f.time_base = t, Fraction(1, fps)
            container.mux(stream.encode(f))
        container.mux(stream.encode())


def fixture(root):
    full, train = root/'full', root/'train'
    fm, tm = full/'meta', train/'meta'
    fm.mkdir(parents=True)
    tm.mkdir(parents=True)
    # Two separate clips of source 91 straddle a pause. Source 92 must get the same macro weight.
    specs = [(90, 0, 26), (91, 0, 29), (91, 43, 10), (92, 0, 25)]
    lines, origins = [], []
    layout = {'video': {v.split('.')[-1]: {'original_key': v} for v in VIEWS}}
    for kind, col in [('state', 'observation.state'), ('action', 'action')]:
        layout[kind] = {}
        start = 0
        for k, w in WIDTHS.items():
            layout[kind][k] = dict(start=start, end=start+w, original_key=col, absolute=True)
            start += w
    info = dict(fps=FPS, chunks_size=1000, data_path='episode_{episode_index}.parquet',
                video_path='{video_key}-{episode_index}.mp4')
    for i, (source, start, n) in enumerate(specs):
        t = np.arange(start, start+n)
        state = np.tile(t[:, None], (1, 14)).astype('float32')  # exact anchor marker; not the label
        gt = commands(t)
        pq.write_table(pa.table({'observation.state': list(state), 'action': list(gt), 'timestamp': np.arange(n)/FPS,
                                 'task_index': [0]*n, 'frame_index': np.arange(n), 'episode_index': [i]*n}),
                       full/f'episode_{i}.parquet')
        for v in VIEWS:
            video(full/f'{v}-{i}.mp4', n)
        lines.append(dict(episode_index=i, length=n, tasks=[TASK]))
        origins.append(dict(episode_index=i, source_episode_index=source, source_frame_start=start,
                            source_frame_stop=start+n, task_index=0))
    for m, count in [(fm, 4), (tm, 1)]:
        dump(m/'info.json', info)
        dump(m/'modality.json', layout)
        dump(m/'embodiment.json', {'embodiment_tag': 'yam'})
        dump_lines(m/'tasks.jsonl', [dict(task_index=0, task=TASK)])
        dump_lines(m/'episodes.jsonl', lines[:count])
        dump_lines(m/'source_episodes.jsonl', origins[:count])
    stats = {'q01': [0.]*14, 'q99': [1.]*14}
    rel = {k: {'q01': [-.3]*w, 'q99': [.7]*w} for k, w in WIDTHS.items()}
    dump(tm/'stats.json', {'action': stats, 'observation.state': stats})
    dump(tm/'relative_stats_dreamzero.json', rel)
    dump(tm/'holdout.json', dict(full_dataset=str(full), train_source_ids=[90], held_out_source_ids=[91,92],
                                 held_out_episode_ids_in_full_dataset=[1,2,3]))
    return train, full


def fake_checkpoint(root, train):
    ckpt, base, wan, tok = root/'run/checkpoint-1200', root/'base', root/'wan', root/'tok'
    (ckpt/'experiment_cfg').mkdir(parents=True)
    for p in (base, wan, tok):
        p.mkdir()
    for name in WAN_FILES:
        (wan/name).write_bytes(b'test component file')  # check-only never deserializes these
    for name in ('tokenizer_config.json', 'tokenizer.json'):
        dump(tok/name, {})
    base_key = 'action_head.model.blocks.weight'
    tuned = ['action_head.model.' + name + '.weight' for name in ('lora_A', 'lora_B', 'action_encoder', 'action_decoder', 'state_encoder')]
    # A sharded AgiBot-like base and BF16 headers, without instantiating a 14B model.
    keymap = {base_key: 'shard1.safetensors', 'action_head.model.extra.weight': 'shard2.safetensors'}
    for key, name in keymap.items():
        save_file({key: torch.full((1,1), 9., dtype=torch.bfloat16)}, str(base/name))
    dump(base/'model.safetensors.index.json', {'weight_map': keymap})
    save_file({k: torch.full((1,1), 3., dtype=torch.bfloat16) for k in tuned}, str(ckpt/'model.safetensors'))
    state_stats = {k: {'q01': [0.]*w, 'q99': [1.]*w} for k,w in WIDTHS.items()}
    rel = json.loads((train/'meta/relative_stats_dreamzero.json').read_text())
    dump(ckpt/'experiment_cfg/metadata.json', {'yam': {'statistics': {'action': rel, 'state': state_stats}}})
    cfg = dict(model={'_target_': 'groot.vla.model.dreamzero.base_vla.VLA'}, train_architecture='lora', save_lora_only=True,
               action_horizon=H, fps={'yam': FPS}, relative_action=True, relative_action_per_horizon=False,
               relative_action_keys=list(WIDTHS), pretrained_model_path=str(base), dit_version=str(wan), tokenizer_path=str(tok))
    (ckpt/'experiment_cfg/conf.yaml').write_text(yaml.safe_dump(cfg))
    dump(ckpt/'trainer_state.json', {'global_step': 1200})
    dump(ckpt/'config.json', {'action_head_cfg': {'config': dict(defer_lora_injection=True, skip_component_loading=True,
        train_architecture='lora', text_encoder_cfg={}, image_encoder_cfg={}, vae_cfg={}, diffusion_model_cfg={})}})
    return ckpt


class MiniHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        for k in ('blocks', 'action_encoder', 'action_decoder', 'state_encoder'):
            setattr(self.model, k, torch.nn.Linear(1,1,bias=False))

    def inject_lora_after_loading(self):
        self.requires_grad_(False)
        for k in ('lora_A', 'lora_B'):
            setattr(self.model, k, torch.nn.Linear(1,1,bias=False))
        for k in ('action_encoder', 'action_decoder', 'state_encoder'):
            getattr(self.model, k).requires_grad_(True)


class Mini(torch.nn.Module):
    config_class = dict

    def __init__(self, cfg):
        super().__init__()
        self.action_head = MiniHead()


class OpenLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.train, cls.full = fixture(cls.root)
        cls.ckpt = fake_checkpoint(cls.root, cls.train)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.data = Holdout(self.train)

    def test_full_pipeline_current_only_all_horizons_and_tail(self):
        anchors = []
        def predict(obs):
            self.assertEqual(set(obs), set(self.data.episodes[0].cameras) | {'annotation.task'} | {'state.'+k for k in WIDTHS})
            self.assertEqual(obs['annotation.task'], TASK)
            t = int(obs['state.left_joint_pos'][0,0])
            anchors.append(t)
            for k in self.data.episodes[0].cameras:
                self.assertEqual(obs[k].shape, (1,16,16,3))
            y = commands(t+np.arange(H))
            # Both squeezed 1D grippers and batched joint vectors occur in official outputs.
            return {'action.'+k: y[:, s].squeeze() if s.stop-s.start == 1 else y[None, :, s] for k,s in self.data.action.items()}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reports = ev.evaluate_step(1200, self.data, ev.plan(self.data.episodes, None), predict, out, 25, baseline=True)
            ev.save_tables(out, reports)
            self.assertEqual(anchors, [0,24,43,0,24])
            s = reports[1]['comparison'][0]
            self.assertEqual((s['n_sources'], s['n_frames']), (2,64))
            self.assertTrue(all(s[k] == 0 for k in s if '_mae' in k or '_rmse' in k))
            self.assertGreater(s['grip_matched'], 0)
            self.assertEqual(s['grip_f1'], 1.)
            self.assertEqual(s['grip_mean_abs_lag_ms'], 0.)
            self.assertGreater(reports[0]['comparison'][0]['left_arm_mae_rad'], 0.)
            with (out/'comparison.csv').open() as f:
                self.assertEqual([r['checkpoint'] for r in csv.DictReader(f)], ['hold_state','1200'])
            with np.load(out/'checkpoint-1200/episode_000002.npz') as z:
                np.testing.assert_equal(z['source_frames'], np.arange(43,53))
                np.testing.assert_equal(z['gt'], z['pred'])
        # Smoke cap is per checkpoint, not per episode. No future frames survive the tail crop.
        self.assertEqual([(ep.id,n) for ep,n in ev.plan(self.data.episodes, 1)], [(1,24)])
        self.assertEqual([(ep.id,n) for ep,n in ev.plan(self.data.episodes, 3)], [(1,29),(2,10)])
        self.assertEqual([e.id for e in Holdout(self.train, self.full, limit=1).episodes],[1,2])

    def test_one_frame_shift_and_dimension_guards(self):
        ep = self.data.episodes[0]
        state, gt = read_rows(ep)
        def shifted(obs):
            t = int(obs['state.left_joint_pos'][0,0])
            x = commands(t+np.arange(H)+1)
            return {'action.'+k: x[:,s] for k,s in self.data.action.items()}
        pred = ev.make_predictions(self.data, ep, ep.length, state, shifted)
        score = metrics(pred, gt, self.data.action)
        self.assertAlmostEqual(score['left_arm_mae_rad'], .1, places=6)
        self.assertGreater(score['right_gripper_mae'], 0.)
        bad = {'action.'+k: np.zeros((H, w)) for k,w in WIDTHS.items()}
        bad['action.left_joint_pos'] = bad['action.left_joint_pos'][:1]
        with self.assertRaisesRegex(ValueError, 'Invalid prediction'):
            action_chunk(bad, self.data.action)

    def test_early_late_spurious_missing_and_empty(self):
        self.assertEqual(crossings([.5,.7,.6,.4,.3,.45,.65],(.35,.65)), {'open':[6], 'close':[4]})
        self.assertEqual(match_events([10,15],[7,12],5), [(10,7),(15,12)])  # nearest-first would lose one
        self.assertCountEqual(match_events([20],[14,21],6), [(20,21),(None,14)])  # min cost second
        self.assertCountEqual(match_events([6,60],[8,90],2), [(6,8),(60,None),(None,90)])
        self.assertIsNone(ev.event_counts([])['grip_mean_abs_lag_ms'])
        ep = self.data.episodes[1]
        g = np.zeros((40,14)); p = g.copy()
        g[10:18,6] = 1; p[12:17,6] = 1; p[28:33,6] = 1; g[21:35,13] = 1
        es = ev.events_for(1600,self.data,ep,g,p,3)
        c = ev.event_counts(es)
        self.assertEqual((c['grip_matched'],c['grip_missed'],c['grip_extra']), (2,2,2))
        self.assertEqual((c['grip_mean_lag_ms'],c['grip_mean_abs_lag_ms']), (10.,30.))
        self.assertIn((53,55,40.), [(e['gt_source_frame'],e['pred_source_frame'],e['lag_ms']) for e in es])
        # A constant closed clip followed by a constant open clip has no in-clip events.
        for v in [g[:5], np.ones((5,14))]:
            self.assertEqual(ev.events_for(1600,self.data,ep,v,v,25), [])

    def test_macro_by_source_not_frame_weighted(self):
        a,b,c = self.data.episodes
        z = lambda n: np.zeros((n,14))
        r = ev.report('x',[(a,z(2),z(2)), (b,z(1),z(1)), (c,z(30),z(30)+2)], [], self.data.action)
        self.assertEqual(r['comparison'][0]['right_arm_rmse_rad'], 1.)
        self.assertEqual(r['comparison'][0]['left_arm_mae_rad'], 1.)

    def test_holdout_leakage_and_video_wrong_rate_rejected(self):
        # Use a temporary copy of the split metadata, never edit the actual held-out dataset.
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)/'train'; shutil.copytree(self.train,copy)
            split = json.loads((copy/'meta/holdout.json').read_text()); split['train_source_ids'].append(91)
            dump(copy/'meta/holdout.json', split)
            with self.assertRaisesRegex(ValueError, 'split'):
                Holdout(copy)
            ep = self.data.episodes[0]
            bad = Path(tmp)/'wrong.mp4'; video(bad, ep.length, fps=25)
            with self.assertRaisesRegex(ValueError, 'Video/row mismatch'):
                list(anchor_images(replace(ep, cameras={'bad':bad}),ep.length))

    @patch.dict(os.environ, {k:'' for k in ('WAN_CKPT_DIR','DZ_CKPT_DIR','TOKENIZER_DIR')})
    def test_real_headers_preflight_mismatch_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'report'
            args = ev.parse_args(['--run',str(self.ckpt.parent),'--steps','1200','--train-data',str(self.train),
                                  '--output',str(out),'--check-only'])
            with redirect_stdout(StringIO()) as output:
                ev.main(args)
            self.assertIn('预检通过', output.getvalue())
            self.assertFalse(out.exists())
            r = recipe(self.ckpt, self.train, self.data)
            model = restore(Mini, r)
            self.assertEqual(model.action_head.model.blocks.weight.item(), 9.)  # from base, frozen
            self.assertEqual(model.action_head.model.action_decoder.weight.item(), 3.)  # from tuned
            key = 'action_head.model.lora_B.weight'
            with self.assertRaisesRegex(ValueError, 'missing=1'):
                restore(Mini, replace(r, tuned_keys=r.tuned_keys-{key}))
            with self.assertRaisesRegex(ValueError, 'AgiBot base: missing'):
                restore(Mini, replace(r, base_keys=set()))
            import shutil
            copy = Path(tmp)/'train'; shutil.copytree(self.train,copy)
            stats = json.loads((copy/'meta/relative_stats_dreamzero.json').read_text())
            stats['left_gripper_pos']['q99'] = [123.]; dump(copy/'meta/relative_stats_dreamzero.json',stats)
            with self.assertRaisesRegex(ValueError, 'differs from training stats'):
                recipe(self.ckpt,copy,self.data)
            shard = Path(tmp)/'bad'; shutil.copytree(r.base_files[0].parent,shard)
            dump(shard/'model.safetensors.index.json', {'weight_map': {'wrong.key': 'shard1.safetensors'}})
            with self.assertRaisesRegex(ValueError, 'Invalid shard index'):
                weights_at(shard)


if __name__ == '__main__':
    unittest.main()
