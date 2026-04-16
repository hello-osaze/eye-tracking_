from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from src.configs.main_config import Args, ModelFactory
from src.models.cec_gaze_model import CECGazeModel
from src.run.single_run.utils import get_checkpoint_path, get_config


def get_fold_paths(base_path: Path) -> list[Path]:
    fold_paths = []
    for path in base_path.glob('fold_index=*'):
        if not path.is_dir():
            continue
        try:
            int(path.name.split('=')[1])
        except (IndexError, ValueError):
            logger.warning(f'Skipping unexpected directory {path.name}')
            continue
        fold_paths.append(path)
    return sorted(fold_paths, key=lambda path: int(path.name.split('=')[1]))


def choose_device(device_name: str = 'auto') -> torch.device:
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA but it is not available.')
        return torch.device('cuda')
    if device_name == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('Requested MPS but it is not available.')
        return torch.device('mps')
    if device_name == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def move_batch_to_device(batch: list, device: torch.device) -> list:
    feature_dict = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch[0].items()
    }
    labels = batch[1].to(device) if isinstance(batch[1], torch.Tensor) else batch[1]
    return [feature_dict, labels, batch[2], batch[3]]


def load_cec_gaze_model(
    fold_path: Path,
    checkpoint_template: str,
    error_context: str,
) -> tuple[Args, CECGazeModel]:
    cfg = get_config(config_path=fold_path)
    checkpoint_path = get_checkpoint_path(
        fold_path,
        checkpoint_template,
    )
    logger.info(f'Loading checkpoint from {checkpoint_path}')

    model_class = ModelFactory.get(cfg.model.base_model_name)
    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location='cpu',
    )
    data_args = model.hparams['data_args']
    trainer_args = model.hparams['trainer_args']
    model_args = model.hparams['model_args']
    model_args.is_training = False
    cfg = Args(
        data=data_args,
        model=model_args,
        trainer=trainer_args,
    )

    model = model_class.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location='cpu',
        model_args=model_args,
    )
    if not isinstance(model, CECGazeModel):
        raise TypeError(
            f'{error_context} expects CECGazeModel, got {type(model).__name__}.'
        )
    model.eval()
    return cfg, model
