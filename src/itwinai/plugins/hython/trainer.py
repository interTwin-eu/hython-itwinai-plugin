import os
from pathlib import Path
from timeit import default_timer
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from ray import train
from ray.train import get_context
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hython.metrics import MetricCollection
from hython.models import ModelLogAPI
from hython.models import get_model_class as get_hython_model
from hython.sampler import SamplerBuilder
from hython.utils import get_lr_scheduler, get_optimizer, get_temporal_steps
from itwinai.distributed import ray_cluster_is_running, suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.distributed import (
    DeepSpeedStrategy,
    HorovodStrategy,
    NonDistributedStrategy,
    RayDDPStrategy,
    TorchDDPStrategy,
)
from itwinai.torch.trainer import TorchTrainer

from .config import HythonConfiguration


class RNNDistributedTrainer(TorchTrainer):
    """Trainer class for RNN model using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (nn.Module | None, optional): model to train.
            Defaults to None.
        strategy (Literal['ddp', 'deepspeed', 'horovod'], optional):
            distributed strategy. Defaults to 'ddp'.
        validation_every (int | None, optional): run a validation epoch
            every ``validation_every`` epochs. Disabled if None. Defaults to 1.
        test_every (int | None, optional): run a test epoch
            every ``test_every`` epochs. Disabled if None. Defaults to None.
        random_seed (int | None, optional): set random seed for
            reproducibility. If None, the seed is not set. Defaults to None.
        logger (Logger | None, optional): logger for ML tracking.
            Defaults to None.
        metrics (Dict[str, Metric] | None, optional): map of torch metrics
            metrics. Defaults to None.
        checkpoints_location (str): path to checkpoints directory.
            Defaults to "checkpoints".
        checkpoint_every (int | None): save a checkpoint every
            ``checkpoint_every`` epochs. Disabled if None. Defaults to None.
        name (str | None, optional): trainer custom name. Defaults to None.
    """
    config: HythonConfiguration
    lr_scheduler: ReduceLROnPlateau | None = None
    model: nn.Module

    def __init__(
        self,
        config: Dict[str, Any],
        epochs: int,
        model: str,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        test_every: int | None = None,
        random_seed: int | None = None,
        logger: Logger | None = None,
        metrics: Dict[str, Callable] | None = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: int | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        # convert config to HythonConfiguration
        config = HythonConfiguration(**config)  # type: ignore
        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            metrics=metrics,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            **kwargs,
        )
        self.epoch_preds = None
        self.epoch_targets = None
        self.epoch_valid_masks = None
        self.save_parameters(**self.locals2params(locals()))
        # returns the model class def in config
        self.model_class = get_hython_model(model)
        self.model_class_name = model
        self.model_dict = {}

    @suppress_workers_print
    # @profile_torch_trainer
    def execute(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> Tuple[Dataset, Dataset, Dataset, Any]:
        """Execute the trainer.

        Args:
            train_dataset (Dataset): training dataset
            validation_dataset (Dataset | None, optional): validation dataset.
                Defaults to None.
            test_dataset (Dataset | None, optional): test dataset. Defaults to None.

        Raises:
            ValueError: if model could not be instantiated
            NotImplementedError: if hython_trainer is not implemented

        Returns:
            Tuple[Dataset, Dataset, Dataset, Any]: training, validation, test datasets and
                any additional information
        """

        self.config.loss_fn = instantiate({"loss_fn": self.config.loss_fn})["loss_fn"]
        self.config.metric_fn = instantiate({"metric_fn": self.config.metric_fn})["metric_fn"]
        self.model_api = ModelLogAPI(self.config)

        if self.config.hython_trainer == "rnntrainer":
            # LOAD MODEL
            self.model_logger = self.model_api.get_model_logger("model")
            self.model = self.model_class(self.config)
            if self.model is None:
                raise ValueError("Model could not be instantiated")

        # TODO: adjust caltrainer to work with Ray TorchTrainer
        # elif self.config.hython_trainer == "caltrainer":
        #     # LOAD MODEL HEAD/SURROGATE
        #     self.model_logger = self.model_api.get_model_logger("head")

        #     # TODO: to remove if condition, delegate logic to model api
        #     if self.model_logger == "mlflow":
        #         surrogate = self.model_api.load_model("head")
        #     else:
        #         # FIXME: There is a clash in "static_inputs" semantics between training and
        #         # calibration
        #         # In the training the "static_inputs" are used to train the CudaLSTM model
        #         # (main model - the surrogate -)
        #         # In the calibration the "static_inputs" are other input features that are
        #         # used to train the TransferNN model.
        #         # Hence during calibration, when loading the weights of the surrogate,
        #         # I need to replace the CudaLSTM (now the head model) "static_inputs" with
        #         # the correct "head_model_inputs" in order to avoid clashes with the
        #         # TransferNN model
        #         # I think that if I used more modular config files, thanks to hydra, then I
        #         # could import a surrogate_model.yaml into both...
        #         config = deepcopy(self.config)
        #         config.static_inputs = config.head_model_inputs
        #         surrogate = get_hython_model(self.config.model_head)(config)

        #         surrogate = self.model_api.load_model("head", surrogate)

        #     transfer_nn = get_hython_model(self.config.model_transfer)(
        #         self.config.head_model_inputs,
        #         len(self.config.static_inputs),
        #         self.config.mt_output_dim,
        #         self.config.mt_hidden_dim,
        #         self.config.mt_n_layers,
        #     )

        #     self.model = self.model_class(
        #         transfernn=transfer_nn,
        #         head_layer=surrogate,
        #         freeze_head=self.config.freeze_head,
        #         scale_head_input_parameter=self.config.scale_head_input_parameter,
        #     )

        #     self.hython_trainer = CalTrainer(self.config)
        else:
            raise NotImplementedError

        self.optimizer = get_optimizer(self.model, self.config)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)
        self._set_target_weights()

        return super().execute(train_dataset, validation_dataset, test_dataset)

    def _set_target_weights(self) -> None:
        """Set the target weights for the model.

        If target_weights is None or "even", the target weights are set to the inverse of the
        number of target variables.
        """
        if self.config.target_weights is None or self.config.target_weights == "even":
            self.target_weights = {
                t: 1 / len(self.config.target_variables) for t in self.config.target_variables
            }
        else:
            raise NotImplementedError

    def create_model_loss_optimizer(self) -> None:
        """Create the model, loss, optimizer and lr scheduler.

        Args:
            self (RNNDistributedTrainer): self
        """
        distribute_kwargs = {}
        if isinstance(self.strategy, DeepSpeedStrategy):
            # Batch size definition is not optional for DeepSpeedStrategy!
            distribute_kwargs = {
                "config_params": {"train_micro_batch_size_per_gpu": self.config.batch_size}
            }
        # if RayDDPStrategy, find_unused_parameters is not supported
        elif isinstance(self.strategy, TorchDDPStrategy) and not isinstance(
                self.strategy, RayDDPStrategy
                ):
            if "find_unused_parameters" not in self.config.model_fields:
                self.config.find_unused_parameters = False
            distribute_kwargs = {"find_unused_parameters": self.config.find_unused_parameters}

        if self.optimizer is None:
            raise ValueError("Optimizer is not set")
        if self.lr_scheduler is None:
            raise ValueError("Lr scheduler is not set")
        if self.model is None:
            raise ValueError("Model is not set")

        (
            self.model,
            self.optimizer,
            self.lr_scheduler,  # type: ignore
        ) = self.strategy.distributed(
            model=self.model,
            optimizer=self.optimizer,
            # not LRScheduler (metric dependent step function)
            lr_scheduler=self.lr_scheduler,  # type: ignore
            **distribute_kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler.

        Args:
            epoch (int): epoch
        """
        if self.profiler is not None:
            self.profiler.step()

        if self.strategy.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
            self.val_loader.sampler.set_epoch(epoch)

    def train_valid_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Train and validate the model.

        Args:
            model (nn.Module): model
            train_loader (DataLoader): training loader
            val_loader (DataLoader): validation loader
            device (torch.device): device

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor],
            torch.Tensor, Dict[str, torch.Tensor]]:
                training loss, training metric, validation loss, validation metric
        """
        model.train()
        # set time indices for training
        self._set_dynamic_temporal_downsampling([train_loader, val_loader])
        train_loss, train_metric = self.epoch_step(
            model, train_loader, device, opt=self.optimizer
        )
        model.eval()
        with torch.no_grad():
            # ? why is this running twice?
            # set time indices for validation
            self._set_dynamic_temporal_downsampling([train_loader, val_loader])
            val_loss, val_metric = self.epoch_step(model, val_loader, device, opt=None)

        return train_loss, train_metric, val_loss, val_metric

    def target_step(self, target: torch.Tensor, steps: int = 1) -> torch.Tensor:
        selection = get_temporal_steps(steps)

        return target[:, selection]

    def predict_step(
        self,
        prediction: Dict[str, torch.Tensor],
        steps: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Return the n steps that should be predicted"""
        selection = get_temporal_steps(steps)
        output = {}
        if self.config.model_head_layer == "regression":
            output["y_hat"] = prediction["y_hat"][:, selection]
        elif self.config.model_head_layer == "distr_normal":
            for k in prediction:
                output[k] = prediction[k][:, selection]
        return output

    def _concatenate_result(
        self,
        prediction: Dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Concatenate results for reporting and computing the metrics

        Args:
            prediction (Dict[str, torch.Tensor]): prediction
            target (torch.Tensor): target
            mask (torch.Tensor | None, optional): mask. Defaults to None.
        """

        # prediction can be probabilistic
        if self.config.model_head_layer == "regression":
            pred_cpu = prediction["y_hat"].detach().cpu().numpy()
        elif self.config.model_head_layer == "distr_normal":
            pred_cpu = prediction["mu"].detach().cpu().numpy()

        target_cpu = target.detach().cpu().numpy()
        if mask is not None:
            mask_cpu = mask.detach().cpu().numpy()
        else:
            mask_cpu = mask

        if self.epoch_preds is None:
            self.epoch_preds = pred_cpu
            self.epoch_targets = target_cpu
            self.epoch_valid_masks = mask_cpu
        else:
            arrays = [self.epoch_preds, pred_cpu]
            self.epoch_preds = np.concatenate(arrays, axis=0)

            arrays = [self.epoch_targets, target_cpu]
            self.epoch_targets = np.concatenate(arrays, axis=0)

            if mask is not None:
                if self.epoch_valid_masks is None:
                    self.epoch_valid_masks = mask_cpu
                else:
                    arrays = [self.epoch_valid_masks, mask_cpu]
                    self.epoch_valid_masks = np.concatenate(arrays, axis=0)

    def _compute_batch_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        target_weight: Dict[str, float] = {},
    ) -> torch.Tensor:
        """Compute the loss for the batch.

        Args:
            prediction (Dict[str, torch.Tensor]): prediction
            target (torch.Tensor): target
            valid_mask (torch.Tensor | None, optional): mask. Defaults to None.
            target_weight (Dict[str, float], optional): target weight. Defaults to {}.

        Returns:
            torch.Tensor: loss
        """
        # Compute targets weighted loss. In case only one target, weight is 1
        # pred and target can be (N, C) or (N, T, C) depending on how the model is trained.
        loss = torch.tensor(0.0, device=target.device)
        for i, target_name in enumerate(target_weight):

            iypred = {}

            if valid_mask is not None:
                imask = valid_mask[..., i]
            else:
                imask = Ellipsis

            # target
            iytrue = target[..., i][imask]

            if self.config.model_head_layer == "regression":
                iypred["y_pred"] = prediction["y_hat"][..., i][imask]
                n = torch.ones_like(iypred["y_pred"])
            elif self.config.model_head_layer == "distr_normal":
                iypred["mu"] = prediction["mu"][..., i][imask]
                iypred["sigma"] = prediction["sigma"][..., i][imask]
                n = torch.ones_like(iypred["mu"])

            w: float = target_weight[target_name]
            # ! TODO: This is a hack to move the data to the correct device, when using Ray
            # check if ray is running before
            if ray_cluster_is_running():
                for key, value in iypred.items():
                    iypred[key] = value.to(f"cuda:{get_context().get_local_rank()}")
                iytrue.to(f"cuda:{get_context().get_local_rank()}")

            loss_tmp: torch.Tensor = self.config.loss_fn(iytrue, **iypred)

            # in case there are missing observations in the batch
            # the loss should be weighted to reduce the importance
            # of the loss on the update of the NN weights
            if valid_mask is not None:
                # fraction valid samples per batch
                scaling_factor = torch.sum(imask) / torch.sum(n)  # type: ignore
                # scale by number of valid samples in a mini-batch
                loss_tmp = loss_tmp * scaling_factor

            loss += loss_tmp * w

        self._set_regularization()

        return loss

    def _set_regularization(self) -> None:
        """Set the regularization for the model.

        Returns:
            None
        """
        self.add_regularization = {}

        # return a dictionary of { reg_func1: weight1, reg_func2: weight2, ...   }

        # actually regularization should access any data in the trainig loop not only pred,
        # target

    def _backprop_loss(self, loss: torch.Tensor, opt: Optimizer) -> None:
        """Backpropagate the loss.

        Args:
            loss (torch.Tensor): loss
            opt (Optimizer): optimizer
        """
        if opt is not None:
            opt.zero_grad()
            loss.backward()

            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), **self.config.gradient_clip  # type: ignore
                )

            opt.step()

    def _compute_metric(self) -> Dict[str, Any]:
        """Compute the metric by the object's metric function.

        Returns:
            Dict[str, Any]: metric
        """
        if isinstance(self.config.metric_fn, MetricCollection):
            metric = self.config.metric_fn(
                self.epoch_targets,
                self.epoch_preds,
                self.config.target_variables,
                self.epoch_valid_masks,
            )
        else:
            metric_or = self.config.metric_fn(
                self.epoch_targets,
                self.epoch_preds,
                self.config.target_variables,
                self.epoch_valid_masks,
            )
            metric = {}
            for itarget in metric_or:
                metric[itarget] = {
                    self.config.metric_fn.__class__.__name__: metric_or[itarget]
                }

        return metric

    def epoch_step(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        opt: Optimizer | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run one epoch of the model.

        Args:
            model (nn.Module): model
            dataloader (DataLoader): dataloader
            device (torch.device): device
            opt (Optimizer | None, optional): optimizer. Defaults to None.

        Returns:
            Tuple[Any, Dict[str, Any]]: loss, metric
        """
        running_batch_loss = 0
        data_points = 0
        for data in dataloader:
            batch_temporal_loss = 0

            for t in self.time_index:  # time_index could be a subset of time indices
                # filter sequence
                dynamic_bt = data["xd"][:, t : (t + self.config.seq_length)].to(device)
                targets_bt = data["y"][:, t : (t + self.config.seq_length)].to(device)

                # static --> dynamic size (repeat time dim)
                static_bt = (
                    data["xs"].unsqueeze(1).repeat(1, dynamic_bt.size(1), 1).to(device)
                )

                x_concat = torch.cat(
                    (dynamic_bt, static_bt),
                    dim=-1,
                )

                pred = model(x_concat)

                output = self.predict_step(pred, steps=self.config.predict_steps)
                target = self.target_step(targets_bt, steps=self.config.predict_steps)

                self._concatenate_result(output, target)

                batch_sequence_loss = self._compute_batch_loss(
                    prediction=output,
                    target=target,
                    valid_mask=None,
                    target_weight=self.target_weights,
                )

                if opt is not None:
                    self._backprop_loss(batch_sequence_loss, opt)

                batch_temporal_loss += batch_sequence_loss.detach()

            data_points += data["xd"].size(0)

            running_batch_loss += batch_temporal_loss

        epoch_loss = running_batch_loss / data_points

        metric = self._compute_metric()

        return epoch_loss, metric

    def _set_dynamic_temporal_downsampling(
        self,
        data_loaders: List[DataLoader],
        opt: Optimizer | None = None,
    ) -> None:
        """Return the temporal indices of the timeseries, it may be a subset of the time
            indices.

        Args:
            data_loaders (List[DataLoader]): data loaders
            opt (Optimizer | None, optional): optimizer. Defaults to None.
        """

        try:
            temporal_downsampling = self.config.temporal_downsampling
        except Exception:
            temporal_downsampling = False

        if temporal_downsampling:
            if len(self.config.temporal_subset) > 1:
                # use different time indices for training and validation
                if opt is None:
                    # validation
                    time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
                    temporal_subset = self.config.temporal_subset[-1]
                else:
                    time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                    temporal_subset = self.config.temporal_subset[0]

                self.time_index = np.random.randint(
                    0, time_range - self.config.seq_length, temporal_subset
                )
            else:
                # use same time indices for training and validation, time indices are from
                # train_loader
                time_range = next(iter(data_loaders[0]))["xd"].shape[1]
                self.time_index = np.random.randint(
                    0, time_range - self.config.seq_length, self.config.temporal_subset[-1]
                )

        else:
            if opt is None:
                # validation
                time_range = next(iter(data_loaders[-1]))["xd"].shape[1]
            else:
                time_range = next(iter(data_loaders[0]))["xd"].shape[1]

            self.time_index = np.arange(0, time_range)

    # @profile_torch_trainer
    # @measure_gpu_utilization
    def train(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Override train_val version of hython to support distributed strategy.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: loss history, metric history
        """
        # self.enable_remote_debugging()
        # Tracking epoch times for scaling test
        if self.strategy.is_main_worker:
            # get number of nodes, defaults to unknown (unk)
            num_nodes = int(os.environ.get("SLURM_NNODES", "unk"))
            epoch_time_output_dir = Path("scalability-metrics/epoch-time")
            epoch_time_file_name = f"epochtime_{self.strategy.name}_{num_nodes}N.csv"
            epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name

            epoch_time_tracker = EpochTimeTracker(
                strategy_name=self.strategy.name,
                save_path=epoch_time_output_path,
                num_nodes=num_nodes,
            )

        device = self.strategy.device()
        loss_history = {"train": [], "val": []}
        metric_history = {f"train_{target}": [] for target in self.config.target_variables}
        metric_history.update({f"val_{target}": [] for target in self.config.target_variables})

        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            epoch_start_time = default_timer()
            self.set_epoch(epoch)
            if self.model is None:
                raise ValueError("Model is not set")
            # run train_valid epoch step of hython trainer
            (
                train_loss,
                train_metric,
                val_loss,
                val_metric,
            ) = self.train_valid_epoch(
                self.model, self.train_loader, self.val_loader, device
            )

            # gather losses from each worker and place them on the main worker.
            worker_val_losses = self.strategy.gather(val_loss, dst_rank=0)

            if worker_val_losses is None:
                raise ValueError("Worker val losses are None")
            # check if all worker_val_losses are tensors
            if not all(isinstance(loss, torch.Tensor) for loss in worker_val_losses):
                raise ValueError("Worker val losses are not all tensors")

            if not self.strategy.is_main_worker:
                continue

            # Moving them all to the cpu() before performing calculations
            avg_val_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(avg_val_loss)  # type: ignore ->
            loss_history["train"].append(train_loss)
            loss_history["val"].append(avg_val_loss)

            self.log(
                item=train_loss.item(),
                identifier="train_loss_per_epoch",
                kind="metric",
                step=epoch,
            )
            self.log(
                item=avg_val_loss.item(),
                identifier="val_loss_per_epoch",
                kind="metric",
                step=epoch,
            )

            for target in self.config.target_variables:
                metric_history[f"train_{target}"].append(train_metric[target])
                metric_history[f"val_{target}"].append(val_metric[target])

            # Aggregate and log metrics
            metric_history_ = {}
            for period in metric_history:
                for target in metric_history[period]:
                    for imetric, metric_value in target.items():
                        metric_key = imetric.lower().split("metric")[0]
                        new_metric_key = period + "_" + metric_key
                        metric_history_[new_metric_key] = [metric_value]

            avg_metrics = pd.DataFrame(metric_history_).mean().to_dict()
            for m_name, m_val in avg_metrics.items():
                self.log(
                    item=m_val,
                    identifier=m_name + "_epoch",
                    kind="metric",
                    step=epoch,
                )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = self.model.state_dict()
                # self.hython_trainer.save_weights(self.model)

            epoch_time = default_timer() - epoch_start_time
            epoch_time_tracker.add_epoch_time(epoch + 1, epoch_time)
            train.report({"loss": avg_val_loss.item(), "train_loss": train_loss.item()})

        if self.strategy.is_main_worker:
            epoch_time_tracker.save()
            self.model.load_state_dict(best_model)

            # MODEL LOGGING
            model_log_names = self.model_api.get_model_log_names()
            for module_name, model_class_name in model_log_names.items():
                item = (
                    self.model
                    if module_name == "model"
                    else self.model.get_submodule(module_name)
                )
                if self.model_logger == "mlflow":
                    self.log(
                        item=item,
                        identifier=model_class_name,
                        kind="model",
                        registered_model_name=model_class_name,
                    )
                else:
                    self.model_api.log_model(module_name, item)

        return loss_history, metric_history

    def create_dataloaders(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset | None,  # ? TODO: Why not used?
    ) -> None:
        """Create the dataloaders.

        Args:
            train_dataset (Dataset): training dataset
            validation_dataset (Dataset): validation dataset
            test_dataset (Dataset | None): test dataset (not used currently)
        """
        sampling_kwargs = {}
        if isinstance(self.strategy, HorovodStrategy):
            sampling_kwargs["num_replicas"] = self.strategy.global_world_size()
            sampling_kwargs["rank"] = self.strategy.global_rank()

        if isinstance(self.strategy, NonDistributedStrategy):
            processing = "single-gpu"
        else:
            processing = "multi-gpu"

        train_sampler_builder = SamplerBuilder(
            train_dataset,
            sampling="random",
            processing=processing,
            sampling_kwargs=sampling_kwargs,
        )

        val_sampler_builder = SamplerBuilder(
            validation_dataset,
            sampling="sequential",
            processing=processing,
            sampling_kwargs=sampling_kwargs,
        )

        train_sampler = train_sampler_builder.get_sampler()
        val_sampler = val_sampler_builder.get_sampler()

        batch_size = self.config.batch_size // self.strategy.global_world_size()
        self.train_loader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_gpu_memory,
            generator=self.torch_rng,
            sampler=train_sampler,
            drop_last=True,
        )  # INFO: drop_last=True, throws errors for samples < batch size, as empty then
        # (can happen for strong downsampling)

        if validation_dataset is not None:
            self.val_loader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler,
                drop_last=False,
            )  # INFO: drop_last=True, throws errors for samples < batch size
            # (can happen for strong downsampling)
