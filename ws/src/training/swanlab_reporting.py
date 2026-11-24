"""SwanLab scalar reporting without HF's version-sensitive native integration.

The factory is invoked by Hydra *inside* each torchrun worker. It keeps the
official DreamZero VLATrainer, adding only a logging callback. Neither get_run()
nor implicit SDK run detection is used: both old (None) and new (RuntimeError)
uninitialized-run semantics are deliberately irrelevant here.
"""

import math
import os
from numbers import Real
from pathlib import Path

from transformers.trainer_callback import TrainerCallback


class DreamZeroSwanLabCallback(TrainerCallback):
    """One explicitly initialized scalar-only run on world-process-zero."""

    def __init__(self):
        self._sdk = None
        self._active = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self._active:
            return
        # Clean legacy scalar SWANLAB_PROJECT before the SDK reads its nested
        # settings, including when this callback is invoked without our launcher.
        from ws.src.training.train_flip_pillow_lora import configure_swanlab

        project = configure_swanlab(args.output_dir)
        # Import and initialize only in rank zero, never in the torchrun parent.
        import swanlab

        mode = os.environ.get('SWANLAB_MODE', 'online').strip().lower()
        if mode not in ('online', 'offline'):
            raise ValueError('SwanLab mode must be online or offline')
        output = Path(args.output_dir)
        # Do not serialize args.to_dict()/os.environ: they may contain credentials.
        names = ('max_steps', 'learning_rate', 'per_device_train_batch_size',
                 'gradient_accumulation_steps', 'world_size', 'seed', 'bf16',
                 'save_steps', 'logging_steps', 'warmup_ratio', 'weight_decay')
        config = {key: getattr(args, key) for key in names if hasattr(args, key)}
        batch = config.get('per_device_train_batch_size', 1)
        accum = config.get('gradient_accumulation_steps', 1)
        world = config.get('world_size', 1)
        config['effective_batch_size'] = batch * accum * world
        options = dict(
            project=project,
            experiment_name=getattr(args, 'run_name', None) or output.name,
            mode=mode, logdir=str(output / 'swanlog'), config=config,
        )
        workspace = os.environ.get('SWANLAB_WORKSPACE', '').strip()
        if workspace:
            options['workspace'] = workspace
        swanlab.init(**options)
        self._sdk = swanlab
        self._active = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        if not self._active:
            self.on_train_begin(args, state, control)
        # Use HF-compatible curve names, including DreamZero's two extra losses.
        metrics = {}
        for key, value in logs.items():
            if not isinstance(value, Real) or not math.isfinite(value):
                continue
            if key.startswith('eval_'):
                name = 'eval/' + key[5:]
            elif key.startswith('test_'):
                name = 'test/' + key[5:]
            else:
                name = 'train/' + key
            metrics[name] = float(value)
        if metrics:
            metrics['train/global_step'] = state.global_step
            self._sdk.log(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self._active:
            try:
                self._sdk.finish()
            finally:
                self._active = False


def create_swanlab_trainer(**kwargs):
    """Hydra trainer factory: preserve all original LoRA/DeepSpeed/save behavior."""
    if kwargs['args'].report_to:
        raise ValueError('SwanLab factory requires report_to=none to prevent duplicate callbacks')
    from groot.vla.experiment import VLATrainer

    trainer = VLATrainer(**kwargs)
    trainer.add_callback(DreamZeroSwanLabCallback())
    return trainer
