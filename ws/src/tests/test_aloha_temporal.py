"""Small CPU-only regression tests; no recordings, robot, FFmpeg or training job needed."""

import sys
import unittest
from pathlib import Path

import numpy as np

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.data_prep.convert_aloha_to_dreamzero import HORIZON, SLICES, relative_statistics, source_spans, timestamps


class TemporalTests(unittest.TestCase):
    def test_anchor_reference_and_last_complete_window(self):
        # Vary both state and target, leave a discontinuity between episodes, and make
        # the last action a sentinel: t+1 shifts, cross-episode windows, or omission of
        # the last complete window will change at least the extrema/mean of the result.
        rng = np.random.default_rng(21)
        pairs = []
        for baseline, n in [(0, HORIZON + 1), (1000, HORIZON + 4)]:
            state = rng.normal(baseline, 1, (n, 14)).astype('float32')
            action = rng.normal(baseline + 5, 3, (n, 14)).astype('float32')
            action[-1] += 100
            pairs.append((state, action))
        ref = np.concatenate([a[t:t + HORIZON] - s[t] for s, a in pairs
                              for t in range(len(a) - HORIZON + 1)]).astype('float64')
        stats = relative_statistics(pairs)
        for key, (a, b) in SLICES.items():
            data = ref[:, a:b]
            for name, expected in [('mean', data.mean(0)), ('std', data.std(0)), ('min', data.min(0)),
                                   ('max', data.max(0)), ('q01', np.quantile(data, .01, axis=0)),
                                   ('q99', np.quantile(data, .99, axis=0))]:
                np.testing.assert_allclose(stats[key][name], expected, rtol=1e-6, atol=1e-7)

    def test_pauses_gaps_and_short_segments(self):
        n = 75
        rows = np.arange(n)
        elapsed = rows / 50 + (rows >= 25) + (rows >= 50)
        arrays = {'frame_index': rows, 'index': 100 + rows, 'episode_index': np.full(n, 2),
                  'timestamp': timestamps(n, 50), 'task_index': np.zeros(n, dtype=int),
                  'recording.segment_index': (rows >= 25).astype(int), 'recording.elapsed_time': elapsed}
        ep = {'episode_index': 2, 'length': n, 'tasks': ['test'],
              'segments': [{'segment_index': 0, 'start_frame': 0}, {'segment_index': 1, 'start_frame': 25}]}
        tasks = [{'task_index': 0, 'task': 'test'}]
        _, spans = source_spans(arrays, ep, 100, 50, tasks, .25)
        self.assertEqual(spans, [(0, 25, 0, 'episode_start'), (25, 50, 1, 'segment_change'),
                                 (50, 75, 1, 'elapsed_gap')])
        elapsed[3:] += 1  # too short to train; never silently join, pad, or drop it
        with self.assertRaisesRegex(ValueError, 'too short'):
            source_spans(arrays, ep, 100, 50, tasks, .25)


if __name__ == '__main__':
    unittest.main()
