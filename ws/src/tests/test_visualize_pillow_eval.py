"""Report tests: real evaluator output writer, deterministic CPU-only dummy actions."""
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.evaluation import eval_flip_pillow as ev
from ws.src.evaluation.visualize_pillow_eval import build_data, write_report


def fixture(root, steps=(1200, 2000), limited=False):
    # Intentionally put the right arm first: the viewer must use NPZ column names, not fixed slices.
    names = [f'{s}_{kind}_{j}' for s in ('right', 'left') for kind, width in (('joint_pos', 6), ('gripper_pos', 1)) for j in range(width)]
    layout = {k[:-2]: slice(i, i+(1 if 'gripper' in k else 6)) for i, k in enumerate(names) if k.endswith('_0')}
    data = SimpleNamespace(names=names, action=layout, state=layout, thresholds={s: (.3, .7) for s in ('left', 'right')})
    episodes = [SimpleNamespace(source=91, id=1, source_start=41, length=24), SimpleNamespace(source=92, id=3, source_start=50, length=25)]
    jobs = [(e, e.length) for e in (episodes[:1] if limited else episodes)]

    def recorded(e):
        t = np.arange(e.length)
        gt = (.02 * (t[:, None] + np.arange(14)[None, :])).astype(np.float32)
        gt[:, names.index('left_gripper_pos_0')] = (t >= 6)
        gt[:, names.index('right_gripper_pos_0')] = 0
        return np.zeros_like(gt), gt

    reports = []
    for i, step in enumerate(steps):
        with patch.object(ev, 'read_rows', recorded), patch.object(ev, 'make_predictions', side_effect=lambda d, e, n, state, fn: recorded(e)[1] if i else state), redirect_stdout(StringIO()):
            reports.extend(ev.evaluate_step(step, data, jobs, None, root, 25, baseline=i == 0))
    ev.save_tables(root, reports)
    protocol = dict(steps=list(steps), horizon=24, fps=50, thresholds=data.thresholds, tolerance_frames=25,
                    segments=[dict(source_id=e.source, episode_index=e.id, source_start=e.source_start, frames=n) for e, n in jobs],
                    frames_per_checkpoint=sum(n for _, n in jobs), blocks_per_checkpoint=sum((n+23)//24 for _, n in jobs),
                    blocks_limit=1 if limited else None, recordings_limit=1 if limited else None)
    (root/'protocol.json').write_text(json.dumps(protocol))


class ReportTest(unittest.TestCase):
    def test_one_window_nonzero_source_frame_and_missing_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, (2000,), limited=True)
            dest = write_report(root)
            html = dest.read_text()
            d = json.loads(re.search(r'id="report-data">(.*?)</script>', html, re.S)[1])
            self.assertTrue(d['limited'])
            self.assertEqual((d['sources'], d['frames'], d['blocks']), (1, 24, 1))
            (ep,) = d['episodes']
            self.assertEqual((ep['start'], ep['anchors'], list(ep['preds'])), (41, [0], ['2000']))
            self.assertEqual(ep['gt'][d['names'].index('left_gripper_pos_0')][6], 1)
            (event,) = d['events']
            self.assertEqual((event['status'], event['gt_source_frame'], event['pred_source_frame']), ('missed', 47, None))
            left_open = next(r for r in d['groups'] if (r['checkpoint'], r['side'], r['direction'])==('2000','left','open'))
            self.assertEqual(left_open['grip_f1'], 0)
            self.assertIsNone(left_open['grip_mean_lag_ms'])
            self.assertNotIn('<script src=', html)

    def test_overlay_and_partial_run_uses_only_finished_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            p = root/'protocol.json'; v = json.loads(p.read_text()); v['steps'].append(2400); p.write_text(json.dumps(v))
            (root/'checkpoint-2400').mkdir()  # Not published in comparison.csv
            d = build_data(root)
            self.assertEqual(d['steps'], ['1200','2000'])
            self.assertEqual((d['episodes'][1]['anchors'], d['frames'], d['blocks']), ([0,24],49,3))
            self.assertEqual(d['episodes'][1]['gt'], d['episodes'][1]['preds']['2000'])
            self.assertNotEqual(d['episodes'][1]['gt'], d['episodes'][1]['preds']['1200'])
            self.assertTrue(any(r['checkpoint']=='hold_state' for r in d['summary']))
            matched = next(r for r in d['groups'] if (r['checkpoint'], r['side'], r['direction'])==('2000','left','open'))
            self.assertEqual((matched['grip_f1'], matched['grip_mean_lag_ms']), (1.,0.))

    def test_inconsistent_frames_or_truth_not_silently_plotted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root)
            file = root/'checkpoint-2000/episode_000001.npz'
            with np.load(file, allow_pickle=False) as z:
                orig = dict(z)
            for key in ('source_frames', 'gt'):
                with self.subTest(key=key):
                    np.savez(file, **{**orig, key: orig[key]+1})
                    with self.assertRaises(ValueError):
                        write_report(root)
                    self.assertFalse((root/'report.html').exists())


if __name__ == '__main__':
    unittest.main()
