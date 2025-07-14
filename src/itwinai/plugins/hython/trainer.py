import logging
import os
from functools import partial
from pathlib import Path
from timeit import default_timer
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from hython.models import get_model_class as get_hython_model
from hython.sampler import SamplerBuilder
from hython.utils import get_lr_scheduler, get_optimizer, get_temporal_steps
from itwinai.components import monitor_exec
from itwinai.distributed import suppress_workers_print
from itwinai.loggers import EpochTimeTracker, Logger
from itwinai.torch.distributed import (
    HorovodStrategy,
    NonDistributedStrategy,
)
from itwinai.torch.monitoring.monitoring import measure_gpu_utilization
from itwinai.torch.profiling.profiler import profile_torch_trainer
from itwinai.torch.trainer import TorchTrainer, _get_tuning_metric_name
from itwinai.utils import time_and_log

from .config import HythonConfiguration
from .data import prepare_batch_for_device

py_logger = logging.getLogger(__name__)


class RNNDistributedTrainer(TorchTrainer):
    """Trainer class for RNN model using pytorch.

    Args:
        config (Union[Dict, TrainingConfiguration]): training configuration
            containing hyperparameters.
        epochs (int): number of training epochs.
        model (str): model to train.
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
    lr_scheduler: LRScheduler
    model: str
    loss: _Loss
    optimizer: Optimizer
    train_time_range: int = 0
    val_time_range: int = 0

    def __init__(
        self,
        config: Dict[str, Any],  # type: ignore
        epochs: int,
        model: str,
        strategy: Literal["ddp", "deepspeed", "horovod"] | None = "ddp",
        test_every: int | None = None,
        logger: Logger | None = None,
        checkpoints_location: str = "checkpoints",
        checkpoint_every: int | None = None,
        name: str | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        # Save for serialization
        self.save_parameters(**self.locals2params(locals()))
        config: HythonConfiguration = HythonConfiguration(**config)
        metrics = {}
        # setup hython mtrics
        metrics[f"{config.metric_name}"] = instantiate({"metric_fn": config.metric_fn})[
            "metric_fn"
        ]

        super().__init__(
            config=config,
            epochs=epochs,
            model=model,
            strategy=strategy,
            test_every=test_every,
            random_seed=random_seed,
            logger=logger,
            checkpoints_location=checkpoints_location,
            checkpoint_every=checkpoint_every,
            name=name,
            metrics=metrics,
            **kwargs,
        )
        py_logger.info(f"random_seed: {random_seed}")
        self.model_class = get_hython_model(model)
        self.model_class_name = model

        self.epoch_preds = None
        self.epoch_targets = None
        self.epoch_valid_masks = None

    @suppress_workers_print
    @monitor_exec
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
        # else:
        #     raise NotImplementedError

        # print(f"PARAMS:{sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        return super().execute(train_dataset, validation_dataset, test_dataset)

    def _setup_metrics(self) -> None:
        # TODO: Hython: Please make Metrics based on torchmetrics so they can be computed on
        # TODO: GPU
        # TODO: remove this override method when done
        pass

    def _set_lr_scheduler_from_config(self) -> None:
        """Parse Lr scheduler from training config"""
        if not self.config.lr_scheduler:
            return
        if not self.optimizer:
            raise ValueError("Trying to instantiate a LR scheduler but the optimizer is None!")

        self.lr_scheduler = get_lr_scheduler(self.optimizer, self.config)

    def _set_optimizer_from_config(self) -> None:
        """Set the optimizer from the configuration.
        Creates an optimizer instance using the configuration parameters and assigns it to
        self.optimizer.
        """
        self.optimizer = get_optimizer(self.model, self.config)

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
        if self.config.hython_trainer == "rnntrainer":
            # LOAD MODEL
            py_logger.info(f"instantiating model {self.model_class_name}")
            self.model = self.model_class(self.config)
            if self.model is None:
                raise ValueError("Model could not be instantiated")
            py_logger.info("Model instantiated")
        else:
            raise NotImplementedError

        self._set_optimizer_from_config()
        self._set_lr_scheduler_from_config()

        if self.optimizer_state_dict:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if self.lr_scheduler_state_dict and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(self.lr_scheduler_state_dict)

        # set target weights for hython model
        self._set_target_weights()
        self._set_loss_from_config()

        # IMPORTANT: model, optimizer, and scheduler need to be distributed from here on
        # Distributed model, optimizer, and scheduler
        distribute_kwargs = self.get_default_distributed_kwargs()
        py_logger.info("Distributing model, optimizer, and scheduler")
        (self.model, self.optimizer, lr_scheduler) = self.strategy.distributed(
            self.model,
            self.optimizer,
            # ! -> not _LRScheduler but LRScheduler
            self.lr_scheduler,  # type: ignore
            **distribute_kwargs,
        )
        py_logger.info("Model, optimizer, and scheduler distributed")

    def _set_loss_from_config(self) -> None:
        """Set the loss function from the configuration.

        Instantiates the loss function specified in the configuration and assigns it to
        self.loss.
        """
        self.loss = instantiate({"loss_fn": self.config.loss_fn})["loss_fn"]

    def set_epoch(self) -> None:
        """Set the epoch for the sampler.

        Args:
            epoch (int): epoch
        """
        if self.profiler is not None and self.current_epoch > 0:
            self.profiler.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()
        self._set_epoch_dataloaders(self.current_epoch)

    def train_epoch(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train the model.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                training loss, training metric
        """
        self.model.train()
        if self.train_dataloader is None:
            raise ValueError("Train dataloader is None")
        train_loss, train_metric = self.epoch_step(
            self.model, self.train_dataloader, opt=self.optimizer
        )
        return train_loss, train_metric

    def validation_epoch(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run one validation epoch.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Tuple containing
                (validation_loss, validation_metrics)
        """
        self.model.eval()
        with torch.no_grad():
            if self.validation_dataloader is None:
                raise ValueError("Validation dataloader is None")
            val_loss, val_metric = self.epoch_step(
                self.model, self.validation_dataloader, opt=None
            )

        return val_loss, val_metric

    def target_step(self, target: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Process target tensor for a specified number of steps.

        Args:
            target (torch.Tensor): The target tensor to process
            steps (int, optional): Number of temporal steps to select. Defaults to 1.

        Returns:
            torch.Tensor: Processed target tensor with selected temporal steps
        """
        selection = get_temporal_steps(steps)
        return target[:, selection]

    def predict_step(
        self,
        prediction: Dict[str, torch.Tensor],
        steps: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """Process model predictions for a specified number of steps.

        Args:
            prediction (Dict[str, torch.Tensor]): Dictionary containing model predictions
            steps (int, optional): Number of temporal steps to select. Defaults to -1.

        Returns:
            Dict[str, torch.Tensor]: Processed predictions with selected temporal steps
        """
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
        else:
            raise ValueError(f"Unknown model head layer: {self.config.model_head_layer}")

        target_cpu = target.detach().cpu().numpy()
        if mask is not None:
            mask_cpu = mask.detach().cpu().numpy()
        else:
            mask_cpu = mask

        if self.epoch_preds is None:
            # Pre-allocate memory for results
            batch_size = (
                prediction["y_hat"].size(0)
                if "y_hat" in prediction
                else prediction["mu"].size(0)
            )
            if self.train_dataloader is None:
                raise ValueError("Train dataloader is None")
            expected_samples = len(self.train_dataloader) * batch_size
            self.epoch_preds = torch.zeros((expected_samples, *prediction["y_hat"].shape[1:]))
            self.epoch_targets = torch.zeros((expected_samples, *target.shape[1:]))
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
        loss = torch.tensor(0.0, device=self.strategy.device())
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
            loss_tmp: torch.Tensor = self.loss(iytrue, **iypred)

            # in case there are missing observations in the batch
            # the loss should be weighted to reduce the importance
            # of the loss on the update of the NN weights
            if valid_mask is not None:
                # fraction valid samples per batch
                scaling_factor = torch.sum(imask) / torch.sum(n)  # type: ignore
                # scale by number of valid samples in a mini-batch
                loss_tmp = loss_tmp * scaling_factor

            loss += loss_tmp * w

        return loss

    def _backprop_loss(self, loss: torch.Tensor, opt: Optimizer) -> None:
        """Backpropagate the loss.

        Args:
            loss (torch.Tensor): loss
            opt (Optimizer): optimizer
        """
        opt.zero_grad()
        loss.backward()

        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                **self.config.gradient_clip,  # type: ignore
            )

        opt.step()

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute the metric by the object's metric function.

        Returns:
            Dict[str, Any]: metric in format {'target_name': {'metric_name': value}}
        """
        new_metrics = {}
        for name, metric in self.metrics.items():
            computed_metrics = metric(
                self.epoch_targets,  # type: ignore
                self.epoch_preds,
                self.config.target_variables,
                self.epoch_valid_masks,
            )
            # computed_metrics will have format {'target_name': value}
            for target_name, metric_value in computed_metrics.items():
                # Initialize nested dict if target_name not present
                if target_name not in new_metrics:
                    new_metrics[target_name] = {}
                # Add the metric value with the metric name as key
                new_metrics[target_name][name] = metric_value

        py_logger.debug(f"Metrics: {new_metrics}")
        # reset
        self.epoch_preds = None
        self.epoch_targets = None
        self.epoch_valid_masks = None

        return new_metrics

    def epoch_step(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        opt: Optimizer | None = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run one epoch of training or validation.

        Args:
            model (nn.Module): The model to train/evaluate
            dataloader (DataLoader): DataLoader providing the batches
            opt (Optimizer | None, optional): Optimizer for training. Defaults to None.

        Returns:
            Tuple[Any, Dict[str, Any]]: Tuple containing (epoch_loss, metrics)
        """
        running_batch_loss = 0

        for batch in dataloader:
            batch = prepare_batch_for_device(batch, self.strategy.device())
            dynamic_bt = batch["xd"]
            # process targets
            targets_bt = batch["y"]
            # process static data
            static_bt = batch["xs"].unsqueeze(1).expand(-1, dynamic_bt.size(1), -1)
            # concatenate dynamic and static data
            x_concat = torch.cat((dynamic_bt, static_bt), dim=-1)

            # process predictions
            pred = model(x_concat)
            output = self.predict_step(pred, steps=self.config.predict_steps)
            target = self.target_step(targets_bt, steps=self.config.predict_steps)
            self._concatenate_result(output, target)

            batch_loss = self._compute_batch_loss(
                prediction=output,
                target=target,
                valid_mask=None,
                target_weight=self.target_weights,
            )

            # Only do backprop during training
            if opt is not None:
                self._backprop_loss(batch_loss, opt)

            running_batch_loss += batch_loss.detach()

        epoch_loss = running_batch_loss / len(dataloader)
        metric = self.compute_metrics()

        return epoch_loss, metric

    @profile_torch_trainer
    @measure_gpu_utilization
    def train(self) -> None:
        # Tracking epoch times for scaling test
        epoch_time_tracker: EpochTimeTracker | None = None
        if self.strategy.is_main_worker and self.strategy.is_distributed:
            if "SLURM_NNODES" not in os.environ:
                raise OSError(
                    "'SLURM_NNODES' is not present in 'os.environ', but is required when "
                    "running distributed training!"
                )
            num_nodes = int(os.environ["SLURM_NNODES"])
            py_logger.warning(
                f"Running distributed training with {num_nodes} nodes. "
                "Epoch times will be tracked for scalability metrics."
            )
            epoch_time_output_dir = Path(f"scalability-metrics/{self.run_id}/epoch-time")
            epoch_time_file_name = f"epochtime_{self.strategy.name}_{num_nodes}N.csv"
            epoch_time_output_path = epoch_time_output_dir / epoch_time_file_name

            epoch_time_tracker = EpochTimeTracker(
                strategy_name=self.strategy.name,
                save_path=epoch_time_output_path,
                num_nodes=num_nodes,
                should_log=self.measure_epoch_time,
            )

        metric_history = {f"train_{target}": [] for target in self.config.target_variables}
        # add empty validation metrics
        metric_history.update({f"val_{target}": [] for target in self.config.target_variables})

        best_loss = float("inf")

        progress_bar = tqdm(
            range(self.current_epoch, self.epochs),
            desc="Epochs",
            disable=self.disable_tqdm or not self.strategy.is_main_worker,
        )
        for self.current_epoch in progress_bar:
            epoch_start_time = default_timer()
            progress_bar.set_description(f"Epoch {self.current_epoch + 1}/{self.epochs}")
            self.set_epoch()
            # run train and validation epoch step of hython trainer
            train_loss, train_metric = self.train_epoch()
            val_loss, val_metric = self.validation_epoch()

            # ! TODO: In hython, make loss serializable, before being able to save checkpoint
            # periodic_ckpt_path = None
            # periodic_ckpt_path = self.save_checkpoint(name=f"epoch_{self.current_epoch}")
            # best_ckpt_path = None

            # gather losses from each worker and place them on the main worker.
            worker_val_losses = time_and_log(
                func=partial(self.strategy.gather, tensor=val_loss, dst_rank=0),
                logger=self.logger,
                identifier="gather_loss_time_s_per_epoch",
                step=self.current_epoch,
            )
            if self.strategy.is_main_worker:
                avg_val_loss = torch.mean(torch.stack(worker_val_losses)).detach().cpu()
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    # ! TODO: In hython, make loss serializable
                    # best_ckpt_path = self.save_checkpoint(
                    #     name="best_model",
                    #     best_validation_metric=avg_val_loss,
                    #     force=True,
                    # )
                    self.best_validation_metric = avg_val_loss

                self.log(
                    item=avg_val_loss.item(),
                    identifier="avg_val_loss_per_epoch",
                    kind="metric",
                    step=self.current_epoch,
                )

            # Report validation metrics to Ray (useful for tuning!)
            metric_name = _get_tuning_metric_name(self.ray_tune_config)
            if metric_name is None:
                raise ValueError("Could not find a metric in the TuneConfig")

            if self.time_ray:
                # time and log the ray_report call
                time_and_log(
                    func=partial(
                        self.ray_report,
                        metrics={"loss": val_loss.item(), "train_loss": train_loss.item()},
                    ),
                    logger=self.logger,
                    identifier="ray_report_time_s_per_epoch",
                    step=self.current_epoch,
                )
            else:
                self.ray_report(
                    metrics={"loss": val_loss.item(), "train_loss": train_loss.item()},
                )

            if self.test_every and (self.current_epoch + 1) % self.test_every == 0:
                self.test_epoch()

            # only main worker and for distributed
            if self.strategy.is_main_worker and self.strategy.is_distributed:
                assert epoch_time_tracker is not None
                epoch_time = default_timer() - epoch_start_time
                epoch_time_tracker.add_epoch_time(self.current_epoch + 1, epoch_time)  # type: ignore

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
                    step=self.current_epoch,
                )

            self.log(
                item=train_loss.item(),
                identifier="train_loss_per_epoch",
                kind="metric",
                step=self.current_epoch,
            )

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
            self.config,
            train_dataset,
            sampling="temporal-downsampling-random",
            processing=processing,
            sampling_kwargs=sampling_kwargs,
        )
        val_sampler_builder = SamplerBuilder(
            self.config,
            validation_dataset,
            sampling="temporal-downsampling-sequential",
            processing=processing,
            sampling_kwargs=sampling_kwargs,
        )

        train_sampler = train_sampler_builder.get_sampler()
        val_sampler = val_sampler_builder.get_sampler()

        batch_size = self.config.batch_size // self.strategy.global_world_size()
        self.train_dataloader = self.strategy.create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,  # Keep workers alive between iterations
            generator=self.torch_rng,
            sampler=train_sampler,
            drop_last=True,
        )  # ! drop_last=True, throws errors for samples < batch size, as empty then
        # (can happen for strong downsampling)
        # check if train_dataset has different time ranges for different batches

        self.train_time_range = train_dataset[0]["xd"].shape[0]

        # Get sequence length from configuration
        if validation_dataset is not None:
            self.validation_dataloader = self.strategy.create_dataloader(
                dataset=validation_dataset,
                batch_size=batch_size,
                num_workers=self.config.num_workers_dataloader,
                pin_memory=self.config.pin_gpu_memory,
                generator=self.torch_rng,
                sampler=val_sampler,
                drop_last=True,
            )  # ! drop_last=True, throws errors for samples < batch size
            # (can happen for strong downsampling)
            self.val_time_range = validation_dataset[0]["xd"].shape[0]
