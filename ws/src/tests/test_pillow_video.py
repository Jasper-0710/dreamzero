"""CPU regression tests: no DreamZero weights, GPU, network, robot or private recordings."""
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from fractions import Fraction
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np
import torch
import yaml

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ws.src.common.pillow import TASK
from ws.src.evaluation import export_pillow_video as video
from ws.src.evaluation.pillow_open_loop import Holdout
from ws.src.evaluation.pillow_video_report import TOP, side_by_side
from ws.src.tests.test_pillow_open_loop import dump, fake_checkpoint, fixture


class FakeVAE:
    def decode(self, z, *, tiled, tile_size, tile_stride):
        assert not torch.is_grad_enabled() and tile_size == (34, 34) and tile_stride == (18, 16)
        self.tiled = tiled
        # Deliberately unlike the recorded gray images: even frame 0 must come from the VAE.
        x = torch.full((1, 3, 9, 16, 16), -1.)
        x[0, 2] = 1.
        x[0, 0] = torch.linspace(-1, 1, 9)[:, None, None]
        return x


def fake_head():
    return SimpleNamespace(vae=FakeVAE(), tiled=False, tile_size_height=34, tile_size_width=34,
                           tile_stride_height=18, tile_stride_width=16)


def video_cfg(ckpt):
    cfgpath = ckpt/'experiment_cfg/conf.yaml'
    cfg = yaml.safe_load(cfgpath.read_text())
    cfg['train_dataset'] = {'dataset_class': 'groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID'}
    cfg['modality_configs'] = {'yam': {'video': {'modality_keys': video.CAMERAS, 'eval_delta_indices': [0]}}}
    cfg['transforms'] = {'yam': {'transforms': [{'_target_': 'groot.vla.data.transform.ConcatTransform', 'video_concat_order': video.CAMERAS}]}}
    cfgpath.write_text(yaml.safe_dump(cfg))
    f = ckpt/'config.json'; m = json.loads(f.read_text()); inner = m['action_head_cfg']['config']
    inner['num_frame_per_block'] = 2
    inner['diffusion_model_cfg']['num_action_per_block'] = 24
    inner['vae_cfg']['_target_'] = 'groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE'
    dump(f, m)


class VideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.train, _ = fixture(cls.root)
        cls.data = Holdout(cls.train)
        cls.ckpt = fake_checkpoint(cls.root, cls.train)
        video_cfg(cls.ckpt)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def opts(self, dest, *more):
        return video.parse_args(['--run', str(self.ckpt.parent), '--train-data', str(self.train), '--step', '1200',
                                 '--source-id', '91', '--anchors', '0', '4', '--output', str(dest), *more])

    def test_selection_uses_source_coordinates_no_tail_or_pause(self):
        ep = replace(self.data.episodes[0], source_start=100)
        self.assertEqual(video.select_windows([ep], 91, [100, 104]), [(ep, [0, 4])])
        for source, anchors in [(90, [0]), (91, [5]), (91, [43]), (91, [1,0]), (91, [0,0])]:
            with self.subTest(source=source, anchors=anchors), self.assertRaises(ValueError):
                video.select_windows(self.data.episodes, source, anchors)
        self.assertEqual(video.parse_args([]).anchors, (168, 192, 216, 240, 264))
        self.assertEqual(video.OFFSETS, (0, 3, 6, 9, 12, 15, 18, 21, 24))

    def test_decode_layout_scale_and_first_frame(self):
        head = fake_head()
        z = torch.zeros(1, 16, 3, 2, 2)
        x = video.decode_prediction(head, z, True)
        self.assertTrue(head.vae.tiled)
        self.assertEqual((x.shape, x.dtype), ((9,16,16,3), np.uint8))
        np.testing.assert_array_equal(x[[0,4,8], 0, 0], [[0,0,255],[127,0,255],[255,0,255]])
        for wrong in [None, z[:, :, 1:], z.transpose(1,2)]:
            with self.assertRaisesRegex(ValueError, 'latent'):
                video.decode_prediction(head, wrong)
        for wrong in [torch.zeros(1,3,8,16,16), torch.full((1,3,9,16,16), float('nan'))]:
            with patch.object(head.vae, 'decode', return_value=wrong), self.assertRaisesRegex(ValueError, 'VAE'):
                video.decode_prediction(head, z)
        with self.assertRaisesRegex(ValueError, '几何'):
            side_by_side(x[0,:8], x[1], 1200, 91, 0, 3)

    def test_check_only_checks_alignment_and_recipe_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, clear=True), \
                patch.object(video, 'independent_policy', side_effect=AssertionError), redirect_stdout(StringIO()):
            dest = Path(tmp)/'unused'
            video.main(self.opts(dest, '--check-only'))
            self.assertFalse(dest.exists())
            m = json.loads((self.ckpt/'config.json').read_text())
            m['action_head_cfg']['config']['num_frame_per_block'] = 1
            with self.assertRaisesRegex(ValueError, '对齐'):
                video.check_video_recipe(self.ckpt, SimpleNamespace(config=m))

    def test_official_wan_decoder_conditioning_frame_is_not_a_forecast(self):
        # Use official causal layers at reduced spatial/channel size, no pretrained weights or CUDA.
        from groot.vla.model.dreamzero.modules.wan_video_vae import VideoVAE_
        nthreads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.inference_mode():
                vae = VideoVAE_(dim=8, num_res_blocks=1).eval()
                z = torch.randn(1,16,3,2,2)
                scale = [torch.zeros(16), torch.ones(16)]
                full, condition = vae.decode(z, scale), vae.decode(z[:,:,:1], scale)
                self.assertEqual(full.shape, (1,3,9,16,16))
                torch.testing.assert_close(full[:,:,:1], condition, rtol=1e-5, atol=1e-5)
        finally:
            torch.set_num_threads(nthreads)

    def test_end_to_end_future_never_enters_forecast_and_mp4_pts_match(self):
        calls = []
        head = fake_head()
        @contextmanager
        def policy(ckpt, verified, seed):
            self.assertEqual((ckpt,seed), (self.ckpt,1140))
            def joint(obs):
                t = int(obs['state.left_joint_pos'][0,0])
                self.assertEqual(set(obs), set(video.CAMERAS) | {'state.'+s for s in self.data.state} | {'annotation.task'})
                self.assertEqual(obs['annotation.task'], TASK)
                for k in video.CAMERAS:
                    self.assertEqual(obs[k].shape, (1,16,16,3))
                    self.assertLess(abs(float(obs[k].mean())-3*t), 5)  # only the anchor image
                calls.append(('forecast', t))
                return None, torch.zeros(1,16,3,2,2)
            yield SimpleNamespace(trained_model=SimpleNamespace(action_head=head)), joint

        def reference(_, obs):
            t = int(obs['state.left_joint_pos'][0,0])
            pic = obs[video.CAMERAS[0]][0]
            calls.append(('reference', t, float(pic.mean())))
            return pic

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}, clear=True), \
                patch.object(video, 'independent_policy', policy), patch.object(video, 'recorded_grid', reference), \
                redirect_stdout(StringIO()):
            dest = Path(tmp)/'export'
            opts = self.opts(dest)
            video.main(opts)
            self.assertEqual(set(p.name for p in dest.iterdir()), {'windows.json','index.html','comparison.mp4'})
            self.assertEqual(len(calls), 20)
            for j, a in enumerate((0,4)):
                self.assertEqual(calls[j*10], ('forecast',a))
                for d, row in zip(video.OFFSETS, calls[j*10+1:j*10+10]):
                    self.assertEqual(row[:2], ('reference', a))
                    self.assertLess(abs(row[2]-3*(a+d)),5)
            meta = json.loads((dest/'windows.json').read_text())
            self.assertEqual([w['first_output_frame'] for w in meta['windows']], [0,9])
            self.assertEqual(meta['windows'][1]['source_frames'], list(range(4,29,3)))
            html = (dest/'index.html').read_text()
            self.assertIn('src="comparison.mp4"', html)
            self.assertNotIn('__RATE__', html)
            with av.open(str(dest/'comparison.mp4')) as c:
                s, = c.streams.video
                self.assertEqual((s.codec_context.name, s.average_rate), ('h264',Fraction(50,3)))
                rgb = list(c.decode(s))
                self.assertEqual([f.pts*f.time_base for f in rgb], [Fraction(i*3,50) for i in range(18)])
                for i in [0,8,9,17]:
                    anchor, d = (0,4)[i//9], video.OFFSETS[i%9]
                    im = rgb[i].to_ndarray(format='rgb24')
                    self.assertTrue(np.allclose(im[TOP+8,8], (3*(anchor+d),)*3, atol=8))
                    self.assertTrue(np.allclose(im[TOP+8,24], (round((i%9)*255/8),0,255), atol=8))
            with self.assertRaises(FileExistsError):
                video.main(opts)
            with patch.object(video, 'recorded_grid', side_effect=ValueError('reference failed')), self.assertRaisesRegex(ValueError,'failed'):
                video.main(self.opts(Path(tmp)/'failed'))
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['export'])


if __name__ == '__main__':
    unittest.main()
