"""Restore the same base -> PEFT injection -> tuned parameters as the training run."""
import gc
import os
import random
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ws.src.common.pillow import WAN_FILES, check_weights, read_json, require_file
from .pillow_open_loop import H, FPS, WIDTHS


@dataclass
class Recipe:
    config: dict
    base_files: list
    base_keys: set
    tuned_files: list
    tuned_keys: set
    tokenizer: Path
    wan: Path

    @property
    def protected_dirs(self):
        return (self.base_files[0].parent, self.wan, self.tokenizer)


def weights_at(root):
    """Read only safetensors headers; verify shard indexes before allocating a model."""
    from safetensors import safe_open
    index = root / 'model.safetensors.index.json'
    mapping = read_json(index)['weight_map'] if index.is_file() else None
    files = sorted(set(mapping.values())) if mapping is not None else ['model.safetensors']
    if not files:
        raise ValueError(f'Empty weights index: {index}')
    keys = set()
    for name in files:
        path = root / name
        require_file(path)
        with safe_open(str(path), framework='np') as f:
            names = set(f.keys())
        if (not names or keys.intersection(names) or
                (mapping is not None and names != {k for k, v in mapping.items() if v == name})):
            raise ValueError(f'Invalid shard index or duplicate keys: {path}')
        keys.update(names)
    return [root / name for name in files], keys


def recipe(ckpt, train, data, base_override=None):
    import yaml
    cfg = yaml.safe_load((ckpt/'experiment_cfg/conf.yaml').read_text())
    if (cfg['model']['_target_'] != 'groot.vla.model.dreamzero.base_vla.VLA' or not cfg['save_lora_only']
            or cfg['train_architecture'] != 'lora' or cfg['action_horizon'] != H or cfg['fps']['yam'] != FPS
            or not cfg['relative_action'] or cfg['relative_action_per_horizon']
            or set(cfg['relative_action_keys']) != set(WIDTHS)):
        raise ValueError(f'Not the expected yam_relative AgiBot training recipe: {ckpt}')
    # Do not derive inference transforms from the full/held-out dataset. Check the
    # actual saved q99 normalizer against this training view, including relative actions.
    stats = read_json(ckpt/'experiment_cfg/metadata.json')['yam']['statistics']
    raw = read_json(train/'meta/stats.json')['observation.state']
    rel = read_json(train/'meta/relative_stats_dreamzero.json')
    for k in WIDTHS:
        for q in ('q01','q99'):
            for saved, reference in [(stats['state'][k][q], raw[q][data.state[k]]),
                                     (stats['action'][k][q], rel[k][q])]:
                if np.shape(saved) != np.shape(reference) or not np.allclose(saved, reference, rtol=1e-5, atol=1e-6):
                    raise ValueError(f'{ckpt}: {k}.{q} differs from training stats; refusing mixed datasets')
    base = Path(base_override or os.environ.get('DZ_CKPT_DIR') or cfg['pretrained_model_path']).expanduser().resolve(strict=True)
    model_cfg = read_json(ckpt/'config.json')
    inner = model_cfg['action_head_cfg']['config']
    if not (inner['defer_lora_injection'] and inner['skip_component_loading'] and inner['train_architecture'] == 'lora'):
        raise ValueError('Expected deferred LoRA injection and AgiBot initialization')
    wan = Path(os.environ.get('WAN_CKPT_DIR') or cfg['dit_version']).resolve(strict=True)
    tok = Path(os.environ.get('TOKENIZER_DIR') or cfg['tokenizer_path']).resolve(strict=True)
    check_weights(base, wan, tok)
    for group, key, name in zip(('text_encoder_cfg','image_encoder_cfg','vae_cfg'),
                                ('text_encoder_pretrained_path','image_encoder_pretrained_path','vae_pretrained_path'), WAN_FILES):
        inner[group][key] = str(wan/name)
    inner['diffusion_model_cfg']['diffusion_model_pretrained_path'] = str(wan)
    base_files, base_keys = weights_at(base)
    tuned_files, tuned_keys = weights_at(ckpt)
    if not any(k.startswith('action_head.model.blocks.') for k in base_keys):
        raise ValueError(f'Base is not an unwrapped DreamZero-AgiBot DiT: {base}')
    for group in ('.lora_A.', '.lora_B.', '.action_encoder.', '.action_decoder.', '.state_encoder.'):
        if not any(group in k for k in tuned_keys):
            raise ValueError(f'{ckpt}: checkpoint does not contain {group} parameters')
    if (ckpt/'trainer_state.json').is_file() and read_json(ckpt/'trainer_state.json')['global_step'] != int(ckpt.name.split('-')[-1]):
        raise ValueError(f'{ckpt}: trainer_state.global_step disagrees with directory')
    return Recipe(model_cfg, base_files, base_keys, tuned_files, tuned_keys, tok, wan)


def _require_keys(required, actual, name, *, exact):
    missing = required - actual
    extra = actual - required if exact else set()
    if missing or extra:
        raise ValueError(f'{name}: missing={len(missing)} {sorted(missing)[:6]}; extra={len(extra)} {sorted(extra)[:6]}')


def restore(cls, r):
    from safetensors.torch import load_file
    model = cls(cls.config_class(**r.config))
    # External Wan T5/CLIP/VAE were already initialized by WANPolicyHead from the
    # same component files as training. The frozen DiT must come from AgiBot, not random init.
    overwritten = ('action_head.model.state_encoder.', 'action_head.model.action_encoder.',
                   'action_head.model.action_decoder.')
    required = {k for k, _ in model.named_parameters() if k.startswith('action_head.model.') and not k.startswith(overwritten)}
    if not required:
        raise ValueError('No DiT parameters to restore')
    _require_keys(required, r.base_keys, 'AgiBot base', exact=False)
    for f in r.base_files:
        model.load_state_dict(load_file(str(f), device='cpu'), strict=False)  # other shards legitimately absent
    model.action_head.inject_lora_after_loading()
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    _require_keys(expected, r.tuned_keys, 'LoRA + trainable encoders/decoder', exact=True)
    for f in r.tuned_files:
        status = model.load_state_dict(load_file(str(f), device='cpu'), strict=False)
        if status.unexpected_keys:
            raise ValueError(f'Unexpected tuned weights in {f}: {status.unexpected_keys[:5]}')
    gc.collect()
    print(f'Restored {len(required)} frozen DiT parameter keys and {len(expected)} tuned keys', flush=True)
    return model


def load_policy(ckpt, verified):
    import torch
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.dreamzero.base_vla import VLA
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    # GrootSimPolicy has no model-factory parameter. Override only its LoRA loader
    # during this single-process constructor; retain official merge, transforms, and unapply().
    called = []
    def loader(cls, path):
        if called or Path(path).resolve() != ckpt:
            raise ValueError(f'Unexpected model path: {path}')
        called.append(True)
        return restore(cls, verified)
    with patch.object(VLA, 'load_lora', classmethod(loader)):
        policy = GrootSimPolicy(embodiment_tag=EmbodimentTag.YAM, model_path=str(ckpt), device='cuda:0',
                                tokenizer_path_override=str(verified.tokenizer))
    if (not called or list(policy.video_delta_indices) != [0] or list(policy.state_delta_indices) != [0]
            or list(policy.modality_configs.action.eval_delta_indices) != list(range(H))):
        raise ValueError('Unexpected policy factory or observation horizon')
    torch.cuda.synchronize()
    return policy


def single_gpu_mask():
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not mask or ',' in mask or mask == '-1' or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('请直接用 python 启动，并设置 CUDA_VISIBLE_DEVICES 为一张已分配的 GPU；不需要 torchrun。')
    if 'LOAD_TRT_ENGINE' in os.environ:
        raise ValueError('请先 unset LOAD_TRT_ENGINE，以免复用其他模型的 TensorRT 权重')
    return mask


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_window_cache(head):
    head.current_start_frame = 0
    for name in ('language', 'clip_feas', 'ys', 'kv_cache1', 'kv_cache_neg', 'crossattn_cache', 'crossattn_cache_neg'):
        setattr(head, name, None)


@contextmanager
def independent_policy(ckpt, verified, seed):
    """One GPU, no server. Both consumers get exactly the same fresh joint prediction at each anchor."""
    single_gpu_mask()
    import torch._dynamo
    import torch.distributed as dist
    from tianshou.data import Batch

    torch._dynamo.config.disable = True
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one allocated CUDA device; set CUDA_VISIBLE_DEVICES to one card')
    torch.cuda.set_device(0)
    seed_all(seed, torch)
    with tempfile.TemporaryDirectory(prefix='pillow-eval-') as tmp:
        dist.init_process_group('gloo', init_method=(Path(tmp)/'rdzv').as_uri(), rank=0, world_size=1)
        try:
            policy = load_policy(ckpt, verified)
            head = policy.trained_model.action_head

            @torch.inference_mode()
            def predict(obs):
                clear_window_cache(head)
                seed_all(seed, torch)
                head.seed = seed
                return policy.lazy_joint_forward_causal(Batch(obs=obs))  # (actions, BCTHW video latent)

            yield policy, predict
        finally:
            dist.destroy_process_group()
