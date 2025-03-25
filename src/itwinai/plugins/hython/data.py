from typing import Dict, List, Tuple
import xarray as xr
from itwinai.components import DataSplitter, monitor_exec

from hython.scaler import Scaler
from hython.datasets import get_dataset
from hython.datasets.wflow_sbm import WflowSBM
from hython.config import Config

def calculate_hython_constant(n: int):
    import time
        
    start = time.time()

    c = 0
    for i in range(n): 
        c += i**120
    end = time.time()
    # print(f"slow func took: {end - start:.2f} seconds to do {n} iterations")
    return c


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
        train_temporal_range: List[str] = None,
        valid_temporal_range: List[str] = None,
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

        scaler = Scaler(cfg, cfg.scaling_use_cached)
        
        # train_dataset = get_dataset(cfg.dataset)(cfg, scaler, True, "train")
        # val_dataset = get_dataset(cfg.dataset)(cfg, scaler, False, "valid")

        ################## Deliberate Slow-Down ##################
        original_getitem = WflowSBM.__getitem__

        def new_get_item(self, idx: int):
            calculate_hython_constant(n=10**4)
            return original_getitem(self, idx)

        WflowSBM.__getitem__ = new_get_item
        ##########################################################

        training_dataset = WflowSBM(cfg=cfg, scaler=scaler, is_train=True, period="train")
        val_dataset = WflowSBM(cfg=cfg, scaler=scaler, is_train=False, period="valid")


        return training_dataset, val_dataset, None
