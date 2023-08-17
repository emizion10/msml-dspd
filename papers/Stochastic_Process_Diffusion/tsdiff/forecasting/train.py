import argparse
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator,Evaluator
from pts.feature import (
    lags_for_fourier_time_features_from_frequency,
)
from tsdiff.forecasting.plot import ( generate_plots)
from tsdiff.forecasting.metrics import ( get_crps )
from tsdiff.forecasting.models import (
    ScoreEstimator,
    TimeGradTrainingNetwork_AutoregressiveOld, TimeGradPredictionNetwork_AutoregressiveOld,
    TimeGradTrainingNetwork_Autoregressive, TimeGradPredictionNetwork_Autoregressive,
    TimeGradTrainingNetwork_All, TimeGradPredictionNetwork_All,
    TimeGradTrainingNetwork_RNN, TimeGradPredictionNetwork_RNN,
    TimeGradTrainingNetwork_Transformer, TimeGradPredictionNetwork_Transformer,
    TimeGradTrainingNetwork_CNN, TimeGradPredictionNetwork_CNN,
)
from tsdiff.utils import NotSupportedModelNoiseCombination, TrainerForecasting

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def energy_score(forecast, target):
    obs_dist = np.mean(np.linalg.norm((forecast - target), axis=-1))
    pair_dist = np.mean(
        np.linalg.norm(forecast[:, np.newaxis, ...] - forecast, axis=-1)
    )
    return obs_dist - pair_dist * 0.5


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    lag = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception(
                                    "Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception(
                                    "Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(
                                    strtobool(line_content[1]))
                            elif line.startswith("@lag"):
                                lag = int(line_content[1])

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception(
                                "Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(
                                    replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
            lag,
        )


def train(
    seed: int,
    dataset: str,
    network: str,
    noise: str,
    diffusion_steps: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    num_cells: int,
    hidden_dim: int,
    residual_layers: int,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # TODO: Find the reason for this logic
    covariance_dim = 4 if dataset != 'exchange_rate_nips' else -4

    # Load data
    # dataset = get_dataset(dataset, regenerate=False)

    uni_dataset = convert_tsf_to_dataframe(
        './papers/Stochastic_Process_Diffusion/tsdiff/data/saugeenday_dataset.tsf')

    univariate_data = uni_dataset[0]['series_value'].values[0]
    forecast_horizon =  uni_dataset[2] #24
    start_timestamp =  uni_dataset[0]['start_timestamp'][0]
    train_index = len(univariate_data) - (forecast_horizon*5)
    data_entry_train = {
        FieldName.START: start_timestamp,  # Replace with appropriate start timestamp
        FieldName.TARGET: univariate_data[:train_index],
    }
    dataset_train = ListDataset([data_entry_train], freq="D")

    test_start_timestamp = start_timestamp + np.timedelta64(train_index, 'D')
    data_entry_test = []
    for i in range(1,6):
        data_entry_test.append({
        FieldName.START: start_timestamp,  # Replace with appropriate start timestamp
        FieldName.TARGET:univariate_data[:(train_index+i*forecast_horizon)],
    })

    dataset_test = ListDataset(data_entry_test, freq="D")

    target_dim = 1

    train_grouper = MultivariateGrouper(max_target_dim=min(2000, target_dim))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset_test) / len(dataset_train)), max_target_dim=min(2000, target_dim))
    dataset_train = train_grouper(dataset_train)
    dataset_test = test_grouper(dataset_test)

    # Slicing original dataset to training & validation arrays
    # val_window = 20 * dataset.metadata.prediction_length
    val_window = int(0.2 * len(univariate_data))
    dataset_train = list(dataset_train)
    dataset_val = []
    for i in range(len(dataset_train)):
        x = deepcopy(dataset_train[i])
        x['target'] = x['target'][:, -val_window:]
        # Eg:- For exchange_rate, Dim - [8,600]
        dataset_val.append(x)
        # Eg:- For exchange_rate, Dim - [8,5471]
        dataset_train[i]['target'] = dataset_train[i]['target'][:, :-val_window]

    min_y = np.min(dataset_train[0]['target'])
    max_y = np.max(dataset_train[0]['target'])
    y_buffer = 0.2 * (max_y-min_y)  

    end_timestamp = start_timestamp + np.timedelta64(dataset_train[0]['target'].shape[1], 'D')
    timestamps = np.arange(start_timestamp, end_timestamp, dtype='datetime64[D]')
    plt.figure(1)
    for feature_idx in range(target_dim):
        plt.plot(timestamps, dataset_train[0]['target'][feature_idx, :], label=f'Feature {feature_idx + 1}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_data.png')
    plt.show()

    # Load model
    if network == 'timegrad':
        if noise != 'normal':
            raise NotSupportedModelNoiseCombination
        training_net, prediction_net = TimeGradTrainingNetwork_Autoregressive, TimeGradPredictionNetwork_Autoregressive
    elif network == 'timegrad_old':
        if noise != 'normal':
            raise NotSupportedModelNoiseCombination
        training_net, prediction_net = TimeGradTrainingNetwork_AutoregressiveOld, TimeGradPredictionNetwork_AutoregressiveOld
    elif network == 'timegrad_all':
        training_net, prediction_net = TimeGradTrainingNetwork_All, TimeGradPredictionNetwork_All
    elif network == 'timegrad_rnn':
        training_net, prediction_net = TimeGradTrainingNetwork_RNN, TimeGradPredictionNetwork_RNN
    elif network == 'timegrad_transformer':
        training_net, prediction_net = TimeGradTrainingNetwork_Transformer, TimeGradPredictionNetwork_Transformer
    elif network == 'timegrad_cnn':
        training_net, prediction_net = TimeGradTrainingNetwork_CNN, TimeGradPredictionNetwork_CNN

    estimator = ScoreEstimator(
        training_net=training_net,
        prediction_net=prediction_net,
        noise=noise,
        target_dim=target_dim,
        #  Eg:- For exchange_rate, prediction_length = 30
        prediction_length=forecast_horizon,
        context_length=forecast_horizon,
        cell_type='GRU',
        num_cells=num_cells,
        hidden_dim=hidden_dim,
        residual_layers=residual_layers,
        # input_size=target_dim * 4 + covariance_dim,
        input_size=target_dim * 4 + 2,
        freq='D',
        loss_type='l2',
        scaling=True,
        diff_steps=diffusion_steps,
        beta_end=20 / diffusion_steps,
        beta_schedule='linear',
        num_parallel_samples=100,
        pick_incomplete=True,
        trainer=TrainerForecasting(
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            num_batches_per_epoch=100,
            batch_size=batch_size,
            patience=10,
        ),
    )
    # Training
    predictor = estimator.train(dataset_train, dataset_val, num_workers=8)

    # Evaluation
    forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    targets = list(ts_it)

    score = energy_score(
        #(5,100,30,1)
        forecast=np.array([x.samples for x in forecasts]),
        #(5,1,30,1)
        target=np.array([x[-forecast_horizon:] for x in targets])[:,None,...],
    )

    lags_seq =  lags_for_fourier_time_features_from_frequency(freq_str='D')
    history_length = forecast_horizon + max(lags_seq)

    generate_plots(forecast=np.array([x.samples for x in forecasts]),
                   test_truth=np.array([x[-(forecast_horizon+history_length):] for x in targets])[:,None,...],
                   history_length=history_length,
                   forecast_horizon=forecast_horizon,
                   target_dim=target_dim,
                   dataset='ER',
                   max_y=max_y,min_y=min_y,y_buffer=y_buffer)

    generate_plots(forecast=np.array([x.samples for x in forecasts]),
                   test_truth=np.array([x[-(forecast_horizon+history_length):] for x in targets])[:,None,...],
                   history_length=history_length,
                   forecast_horizon=forecast_horizon,
                   target_dim=target_dim,
                   dataset='ER_Sampled',
                   sample_length=5,
                   max_y=max_y,min_y=min_y,y_buffer=y_buffer)

    get_crps(forecast=np.array([x.samples for x in forecasts]),
             test_truth=np.array([x[-forecast_horizon:] for x in targets])[:,None,...],)


    evaluator = Evaluator(quantiles=(np.arange(20)/20.0)[1:])
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    metrics = dict(
        CRPS=agg_metric['mean_wQuantileLoss'],
        ND=agg_metric['ND'],
        NRMSE=agg_metric['NRMSE'],
        # CRPS_sum=agg_metric['m_sum_mean_wQuantileLoss'],
        # ND_sum=agg_metric['m_sum_ND'],
        # NRMSE_sum=agg_metric['m_sum_NRMSE'],
        energy_score=score,
    )
    metrics = {k: float(v) for k, v in metrics.items()}

    return metrics
    # return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train forecasting model.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--network', type=str, choices=[
        'timegrad', 'timegrad_old', 'timegrad_all', 'timegrad_rnn', 'timegrad_transformer', 'timegrad_cnn'
    ])
    parser.add_argument('--noise', type=str, choices=['normal', 'ou', 'gp'])
    parser.add_argument('--diffusion_steps', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_cells', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--residual_layers', type=int, default=8)
    args = parser.parse_args()

    metrics = train(**args.__dict__)

    for key, value in metrics.items():
        print(f'{key}:\t{value:.4f}')

# Example:
# python -m tsdiff.forecasting.train --seed 1 --dataset electricity_nips --network timegrad_rnn --noise ou --epochs 100
