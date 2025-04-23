import logging
import pickle
from typing import Dict, List, Tuple

import torch

from hython.config import Config
from hython.datasets import get_dataset
from hython.datasets.wflow_sbm import WflowSBM
from hython.scaler import Scaler
from itwinai.components import DataSplitter, monitor_exec

py_logger = logging.getLogger(__name__)


class RNNDatasetGetterAndPreprocessor(DataSplitter):
    def __init__(
        self,
        # == common ==
        hython_trainer: str,
        dataset: str,
        data_lazy_load: bool,
        scaling_variant: str,
        scaling_use_cached: bool,
        experiment_name: str,
        experiment_run: str,
        data_source: dict,
        work_dir: str,
        dynamic_inputs: List[str] | None = None,
        static_inputs: List[str] | None = None,
        target_variables: List[str] | None = None,
        scaling_static_range: Dict | None = None,
        mask_variables: List[str] | None = None,
        static_inputs_mask: List[str] | None = None,
        head_model_inputs: List[str] | None = None,
        train_temporal_range: List[str] | None = None,
        valid_temporal_range: List[str] | None = None,
        train_downsampler: Dict | None = None,
        valid_downsampler: Dict | None = None,
        downsampling_temporal_dynamic: bool | None = None,
        min_sample_target: int | None = None,
        seq_length: int | None = None
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(self) -> Tuple[WflowSBM, WflowSBM, None]:
        cfg = Config()

        for i in self.parameters:
            setattr(cfg, i, self.parameters[i])

        scaler = Scaler(cfg, cfg.scaling_use_cached) # type: ignore

        train_dataset = get_dataset(cfg.dataset)(cfg, scaler, True, "train") # type: ignore
        # check pickled dataset size
        py_logger.info(
            "pickled train_dataset_size: "
            f"{len(pickle.dumps(train_dataset)) / (1024 * 1024 * 1024):.2f} GB"
            )

        val_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "valid") # type: ignore
        return train_dataset, val_dataset, None


def ray_cluster_is_running() -> bool:
    """Detect if code is running inside a Ray cluster."""
    try:
        from ray import train
        train.get_context()
        return True
    except (ImportError, RuntimeError):
        return False


def prepare_batch_for_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Move a batch dictionary to the appropriate device based on strategy.

    This function specially handles Ray distributed training by detecting the current worker's
    device and moving tensors appropriately.

    Args:
        batch: A dictionary containing tensor data
        device: The default device to use (non-Ray case)
        strategy: The distributed strategy object (optional)

    Returns:
        The prepared batch with tensors on correct devices
    """
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }
