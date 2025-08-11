import logging
from typing import Dict, List, Tuple

import torch
import xarray as xr
from torch.utils.data._utils.collate import default_collate

from hython.config import Config
from hython.datasets import get_dataset
from hython.datasets.wflow_sbm import WflowSBM
from hython.scaler import Scaler
from itwinai.components import DataSplitter, monitor_exec

py_logger = logging.getLogger(__name__)


def xarray_collate_fn(batch):
    """Custom collate function to handle xarray DataArrays with zero-copy tensor views.
    Converts xarray DataArrays to PyTorch tensors using views (no copying).

    Args:
        batch: List of samples from the dataset

    Returns:
        Collated batch with tensor views instead of DataArrays
    """
    def convert_xarray_to_tensor_view(item):
        """Recursively convert xarray DataArrays to PyTorch tensor views (zero-copy)."""
        if isinstance(item, xr.DataArray):
            # Use as_tensor for zero-copy view instead of from_numpy which copies
            return torch.as_tensor(item.values)
        elif isinstance(item, dict):
            return {k: convert_xarray_to_tensor_view(v) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return type(item)(convert_xarray_to_tensor_view(v) for v in item)
        else:
            return item

    converted_batch = [convert_xarray_to_tensor_view(sample) for sample in batch]

    return default_collate(converted_batch)


class RNNDatasetGetterAndPreprocessor(DataSplitter):
    def __init__(
        self,
        # == common ==
        hython_trainer: str,
        dataset: str,
        data_lazy_load: bool,
        preprocessor: dict,
        scaler: dict,
        scaling_use_cached: bool,
        experiment_name: str,
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
        seq_length: int | None = None,
    ) -> None:
        self.save_parameters(**self.locals2params(locals()))

    @monitor_exec
    def execute(self) -> Tuple[WflowSBM, WflowSBM, None]:
        cfg = Config()

        for i in self.parameters:
            setattr(cfg, i, self.parameters[i])

        # Enable lazy loading to prevent loading entire dataset into memory
        cfg.data_lazy_load = getattr(cfg, "data_lazy_load", True)
        scaler = Scaler(cfg, cfg.scaling_use_cached)  # type: ignore

        train_dataset = get_dataset(cfg.dataset)(cfg, scaler, True, "train")  # type: ignore

        if py_logger.isEnabledFor(logging.DEBUG):
            try:
                dataset_len = (
                    len(train_dataset) if hasattr(train_dataset, "__len__") else "unknown"
                )
                py_logger.debug(f"train_dataset length: {dataset_len}")
            except Exception as e:
                py_logger.debug(f"Could not determine dataset size: {e}")

        val_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "valid")  # type: ignore
        return train_dataset, val_dataset, None


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
