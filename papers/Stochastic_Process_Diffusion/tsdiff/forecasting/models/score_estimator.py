from typing import Any, Callable, List, Optional

import torch
from torch.utils.data import DataLoader

from typing import NamedTuple, Optional
import torch
import torch.nn as nn
from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import SelectFields, Transformation
from gluonts.itertools import maybe_len
from pts.model import get_module_forward_input_names
from pts.dataset.loader import TransformedIterableDataset

from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.model.predictor import Predictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from gluonts.core.component import validated

from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)
from pts.model import PyTorchEstimator
from pts.model.utils import get_module_forward_input_names

from tsdiff.utils import TrainerForecasting

class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor

class ScoreEstimator(PyTorchEstimator):
    def __init__(
        self,
        training_net: Callable,
        prediction_net: Callable,
        noise: str,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: TrainerForecasting = TrainerForecasting(),
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "GRU",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        hidden_dim: int = 100,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_end=0.1,
        beta_schedule="linear",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        scaling: bool = True,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        old: bool = False,
        time_feat_dim: int = 2, #change acc to frequency
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.training_net = training_net
        self.prediction_net = prediction_net
        self.noise = noise

        self.old = old

        self.freq = freq
        self.context_length = context_length if context_length is not None else prediction_length

        self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension

        self.conditioning_length = hidden_dim
        self.diff_steps = diff_steps
        self.loss_type = loss_type
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.residual_layers = residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def create_training_network(self, device: torch.device):
        return self.training_net(
            noise=self.noise,
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
            time_feat_dim=self.time_feat_dim,
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: Any,
        device: torch.device,
    ) -> Predictor:
        prediction_network = self.prediction_net(
            noise=self.noise,
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            conditioning_length=self.conditioning_length,
            num_parallel_samples=self.num_parallel_samples,
            time_feat_dim=self.time_feat_dim,
        ).to(device)

        # Clarify: parameters are copied from training network to predicction network
        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )

    def train_model(
            self,
            training_data: Dataset,
            validation_data: Optional[Dataset] = None,
            dataset_test: Optional[ListDataset] = None,
            num_workers: int = 0,
            prefetch_factor: int = 2,
            shuffle_buffer_length: Optional[int] = None,
            cache_data: bool = False,
            mean: float = 0,
            std: float =0,
            **kwargs,
        ) -> TrainOutput:
            transformation = self.create_transformation()

            trained_net = self.create_training_network(self.trainer.device)

            input_names = get_module_forward_input_names(trained_net)

            with env._let(max_idle_transforms=maybe_len(training_data) or 0):
                training_instance_splitter = self.create_instance_splitter("training")
            training_iter_dataset = TransformedIterableDataset(
                ## Eg:- [8,5471] for exchange rate
                dataset=training_data,
                transform=transformation
                + training_instance_splitter
                + SelectFields(input_names),
                is_train=True,
                shuffle_buffer_length=shuffle_buffer_length,
                cache_data=cache_data,
            )

            training_data_loader = DataLoader(
                training_iter_dataset,
                batch_size=self.trainer.batch_size,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                worker_init_fn=self._worker_init_fn,
                **kwargs,
            )

            validation_data_loader = None
            if validation_data is not None:
                with env._let(max_idle_transforms=maybe_len(validation_data) or 0):
                    validation_instance_splitter = self.create_instance_splitter("validation")
                validation_iter_dataset = TransformedIterableDataset(
                    dataset=validation_data,
                    transform=transformation
                    + validation_instance_splitter
                    + SelectFields(input_names),
                    is_train=True,
                    cache_data=cache_data,
                )
                validation_data_loader = DataLoader(
                    validation_iter_dataset,
                    batch_size=self.trainer.batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    pin_memory=True,
                    worker_init_fn=self._worker_init_fn,
                    **kwargs,
                )
            ## Call to TrainerForecasting
            self.trainer(
                net=trained_net,
                train_iter=training_data_loader,
                validation_iter=validation_data_loader,
                create_pred = self.create_predictor,
                dataset_test=dataset_test,
                transformation=transformation,
                mean=mean,
                std=std,
            )

            return TrainOutput(
                transformation=transformation,
                trained_net=trained_net,
                predictor=self.create_predictor(
                    transformation, trained_net, self.trainer.device
                ),
            )

