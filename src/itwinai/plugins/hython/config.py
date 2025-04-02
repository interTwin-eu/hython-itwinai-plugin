# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Linus Eickhoff
#
# Credit:
# - Linus Eickhoff <linus.maximilian.eickhoff@cern.ch> - CERN
# --------------------------------------------------------------------------------------

from typing import Any, Dict, List, Literal, Union

from itwinai.torch.config import TrainingConfiguration


class HythonConfiguration(TrainingConfiguration):
    temporal_downsampling: bool
    temporal_subset: List[int]
    seq_length: int
    target_variables: List[str]
    predict_steps: int
    target_weights: Union[str, Dict[str, float]] | None = None
    gradient_clip: Dict[str, Any] | None = None
    model_head_layer: str | None = None
    loss_fn: Any
    metric_fn: Any
    model_path: str | None = None
    model_head_activation: str
    model_head_kwargs: Dict[str, Any] = {}
    dynamic_inputs: List[str]
    static_inputs: List[str]
    hython_trainer: str
    lr_scheduler_hython: Dict[str, Any] | None = None
    lr_scheduler: Literal[
        "step", "multistep", "constant", "linear", "exponential", "polynomial"
    ] | None = None

    lstm_layers: int
    lstm_batch_norm: bool
    hidden_size: int
    dropout: float

    find_unused_parameters: bool = False
