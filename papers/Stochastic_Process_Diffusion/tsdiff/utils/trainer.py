from typing import Optional, Union

from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from gluonts.dataset.common import Dataset
from gluonts.model.predictor import Predictor
from gluonts.transform import Transformation
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.core.component import validated
from pts import Trainer
from tsdiff.forecasting.metrics import get_crps
from tsdiff.forecasting.plot import plotCRPS


class TrainerForecasting(Trainer):
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: float = 1e-2,
        clip_gradient: Optional[float] = None,
        patience: int = None,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.maximum_learning_rate = maximum_learning_rate
        self.clip_gradient = clip_gradient
        self.patience = patience
        self.device = device

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
        dataset_test: Optional[Dataset] = None,
        create_pred: Optional[Predictor] = None,
        transformation: Optional[Transformation] = None,
        mean: float = 0,
        std: float = 0,
    ) -> None:

        intermediate_net = net

        optimizer = Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )

        # Early stopping setup
        best_loss = float('inf')
        waiting = 0
        best_net = deepcopy(net.state_dict())
        crps_scores=[]
        # Training loop
        for epoch_no in range(self.epochs):
            # mark epoch start time
            cumm_epoch_loss = 0.0
            total = self.num_batches_per_epoch - 1

            # training loop
            with tqdm(train_iter, total=total) as it:
                for batch_no, data_entry in enumerate(it, start=1):

                    optimizer.zero_grad()

                    inputs = [v.to(self.device) for v in data_entry.values()]

                    loss = net(*inputs)

                    if isinstance(loss, (list, tuple)):
                        loss = loss[0]

                    loss.backward()

                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(net.parameters(), self.clip_gradient)

                    optimizer.step()
                    lr_scheduler.step()

                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    it.set_postfix(
                        {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                        },
                        refresh=False,
                    )

                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()

            # validation loop
            if validation_iter is not None:
                cumm_epoch_loss_val = 0.0
                with tqdm(validation_iter, total=total, colour="green") as it:

                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device) for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                        if isinstance(output, (list, tuple)):
                            loss = output[0]
                        else:
                            loss = output

                        cumm_epoch_loss_val += loss.item()
                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        it.set_postfix(
                            {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                            },
                            refresh=False,
                        )

                        if self.num_batches_per_epoch == batch_no:
                            break
                it.close()

                # Early stopping logic
                if avg_epoch_loss_val < best_loss:
                    best_loss = avg_epoch_loss_val
                    best_net = deepcopy(net.state_dict())
                    waiting = 0
                elif waiting > self.patience:
                    print(f'Early stopping at epoch {epoch_no}')
                    break
                else:
                    waiting += 1

            print(f'Epoch {epoch_no} finished')
            if(dataset_test is not None):
                print(f'Calculating CRPS of epoch {epoch_no}')
                intermediate_net.load_state_dict(net.state_dict())
                predictor = create_pred(transformation,intermediate_net,self.device)
                forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test, predictor=predictor, num_samples=100)
                forecasts = list(forecast_it)
                targets = list(ts_it)
                crps = get_crps(forecast=(np.array([x.samples for x in forecasts]) * std) + mean,
                test_truth=(np.array([x[-30:] for x in targets])[:,None,...] * std) + mean, print_results=False )
                crps_scores.append(crps)


            # mark epoch end time and log time cost of current epoch
        plotCRPS(crps_scores)
        net.load_state_dict(best_net)

# python -m tsdiff.train_forecasting --seed 1 --dataset electricity_nips --network timegrad_rnn --noise ou --epochs 100
