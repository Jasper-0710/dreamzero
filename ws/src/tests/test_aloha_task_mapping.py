"""CPU task-label regressions, plus a tiny real Parquet/JPEG/MP4 conversion."""

import hashlib
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ws.src.common.pillow import TASK, check_data
from ws.src.data_prep import convert_aloha_to_dreamzero as conv
from ws.src.data_prep.check_aloha_alignment import compare


def raw_fixture(root, *, split_first=False):
    tasks = [{'task_index': 0, 'task': 'pick_cup'}, {'task_index': 1, 'task': 'flip_the_pillow'}]
    features = {key: {'dtype': 'float32', 'shape': [14], 'names': conv.JOINTS}
                for key in (conv.STATE, conv.ACTION)}
    features.update({key: {'dtype': dtype, 'shape': [1]} for key, dtype in [
        ('timestamp', 'float32'), ('recording.elapsed_time', 'float64'),
        *[(key, 'int64') for key in ('frame_index', 'episode_index', 'index', 'task_index',
                                    'recording.segment_index')]]})
    features.update({key: {'dtype': 'image', 'shape': [16, 16, 3],
                          'names': ['height', 'width', 'channels']} for key in conv.CAMERAS})
    episodes, offset = [], 0
    for ep in range(2):
        n = 52 if ep == 0 and split_first else 26
        ids = np.arange(n, dtype=np.int64)
        values = (ids[:, None] * .01 + np.arange(14)[None, :] + ep).astype(np.float32)
        arrays = {conv.STATE: pa.array(values.tolist(), type=pa.list_(pa.float32(), 14)),
                  conv.ACTION: pa.array((values + .1).tolist(), type=pa.list_(pa.float32(), 14)),
                  'timestamp': pa.array(conv.timestamps(n, 50)),
                  'recording.elapsed_time': pa.array(ids / 50),
                  'frame_index': pa.array(ids), 'index': pa.array(ids + offset),
                  'episode_index': pa.array(np.full(n, ep, dtype=np.int64)),
                  'task_index': pa.array(np.full(n, ep, dtype=np.int64)),
                  'recording.segment_index': pa.array((ids >= 26).astype(np.int64))}
        for camera, key in enumerate(conv.CAMERAS):
            jpeg = io.BytesIO()
            Image.new('RGB', (16, 16), (30 + 30 * camera, 80 + ep * 40, 120)).save(jpeg, format='JPEG')
            arrays[key] = pa.array([{'bytes': jpeg.getvalue()}] * n,
                                   type=pa.struct([('bytes', pa.binary())]))
        path = conv.ep_path(root, conv.DATA, ep, 1000)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(arrays), path)
        segments = [{'segment_index': 0, 'start_frame': 0}]
        if n == 52:
            segments.append({'segment_index': 1, 'start_frame': 26})
        episodes.append({'episode_index': ep, 'length': n, 'tasks': [tasks[ep]['task']], 'segments': segments})
        offset += n
    info = dict(codebase_version='v2.1', robot_type='mobile_aloha', data_path=conv.DATA,
                chunks_size=1000, fps=50, total_episodes=2, total_frames=offset, total_tasks=2, features=features)
    (root / 'meta').mkdir()
    conv.write_json(root / 'meta/info.json', info)
    conv.write_jsonl(root / 'meta/episodes.jsonl', episodes)
    conv.write_jsonl(root / 'meta/tasks.jsonl', tasks)
    return info, episodes, tasks


class TaskMappingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw, self.out = self.root / 'raw', self.root / 'converted'
        self.info, self.episodes, self.tasks = raw_fixture(self.raw, split_first=True)

    def plan(self, task_text=TASK, confirm=False, episodes=None):
        with redirect_stdout(io.StringIO()):
            return conv.plan_tasks(self.raw, self.info, self.episodes if episodes is None else episodes,
                                   self.tasks, task_text, confirm)

    def args(self, task_text=TASK, confirm=False, start=0, limit=None):
        return SimpleNamespace(task_text=task_text, confirm_same_task=confirm, start_episode=start,
                               limit=limit, max_gap=.25)

    def test_multiple_actual_labels_require_separate_confirmation_before_encoding(self):
        with patch.object(conv, 'write_video') as video, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, '--confirm-same-task'):
                conv.convert(self.raw, self.out, self.args())
        video.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_unused_historical_label_does_not_need_confirmation(self):
        mapping, target, usage = self.plan(episodes=self.episodes[1:])
        self.assertEqual(mapping, {1: 0})
        self.assertEqual(target, [{'task_index': 0, 'task': TASK}])
        self.assertEqual(usage, [{'source_task_index': 1, 'source_task': 'flip_the_pillow', 'recordings': 1}])

    def test_explicit_confirmation_maps_both_labels_to_one_output_id(self):
        mapping, target, usage = self.plan(confirm=True)
        self.assertEqual(mapping, {0: 0, 1: 0})
        self.assertEqual(target, [{'task_index': 0, 'task': TASK}])
        self.assertEqual([row['recordings'] for row in usage], [1, 1])

    def test_keep_labels_preserves_distinct_tasks_and_compacts_subset(self):
        mapping, target, _ = self.plan(task_text=None)
        self.assertEqual(mapping, {0: 0, 1: 1})
        self.assertEqual(target, self.tasks)
        mapping, target, _ = self.plan(task_text=None, episodes=self.episodes[1:])
        self.assertEqual(mapping, {1: 0})
        self.assertEqual(target, [{'task_index': 0, 'task': 'flip_the_pillow'}])

    def test_confirmation_without_instruction_is_invalid(self):
        with self.assertRaisesRegex(ValueError, 'requires an explicit --task-text'):
            self.plan(task_text=None, confirm=True)

    def test_confirmation_does_not_hide_inconsistent_episode_task_ids(self):
        path = conv.ep_path(self.raw, conv.DATA, 1, 1000)
        table = pq.read_table(path)
        table = table.set_column(table.schema.get_field_index('task_index'), 'task_index',
                                 pa.array([0] + [1] * 25, type=pa.int64()))
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, 'consistent source task'):
            self.plan(confirm=True)

    def test_metadata_must_match_actual_task_references(self):
        self.episodes[0]['tasks'] = ['flip_the_pillow']
        with self.assertRaisesRegex(ValueError, 'consistent source task'):
            self.plan(confirm=True)

    def test_invalid_task_column_still_rejected_after_confirmation(self):
        path = conv.ep_path(self.raw, conv.DATA, 1, 1000)
        original = pq.read_table(path)
        for values, dtype in [([None] + [1] * 25, pa.int64()), ([1] * 26, pa.float64()),
                              ([99] * 26, pa.int64())]:
            with self.subTest(values=values[:1], dtype=dtype):
                table = original.set_column(original.schema.get_field_index('task_index'), 'task_index',
                                            pa.array(values, type=dtype))
                pq.write_table(table, path)
                with self.assertRaises(ValueError):
                    self.plan(confirm=True)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'real video test needs FFmpeg')
    def test_real_conversion_split_provenance_alignment_and_source_unchanged(self):
        before = {p.relative_to(self.raw): hashlib.sha256(p.read_bytes()).digest()
                  for p in self.raw.rglob('*') if p.is_file()}
        with redirect_stdout(io.StringIO()):
            info = conv.convert(self.raw, self.out, self.args(confirm=True))
            check_data(self.out)
            for ep in range(info['total_episodes']):
                compare(SimpleNamespace(input=self.raw, output=self.out, episode=ep, samples=2, max_mae=12))
        self.assertEqual(info['total_tasks'], 1)
        self.assertEqual(info['total_episodes'], 3)  # First raw recording has a pause boundary.
        self.assertEqual(conv.read_jsonl(self.out / 'meta/tasks.jsonl'), [{'task_index': 0, 'task': TASK}])
        origins = conv.read_jsonl(self.out / 'meta/source_episodes.jsonl')
        self.assertEqual([r['source_task_index'] for r in origins], [0, 0, 1])
        self.assertEqual([r['source_task'] for r in origins], ['pick_cup', 'pick_cup', 'flip_the_pillow'])
        for ep in range(3):
            output = pq.read_table(conv.ep_path(self.out, conv.DATA, ep, 1000))
            self.assertEqual(np.unique(output['task_index'].to_numpy()).tolist(), [0])
        import json
        manifest = json.loads((self.out / 'meta/conversion.json').read_text())
        self.assertTrue(manifest['confirmed_same_task'])
        self.assertEqual([r['task_index'] for r in manifest['task_mapping']], [0, 0])
        after = {p.relative_to(self.raw): hashlib.sha256(p.read_bytes()).digest()
                 for p in self.raw.rglob('*') if p.is_file()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
