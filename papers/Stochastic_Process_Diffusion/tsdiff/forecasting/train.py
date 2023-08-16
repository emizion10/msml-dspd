import argparse
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from pts.feature import (
    lags_for_fourier_time_features_from_frequency,
)
from tsdiff.forecasting.plot import ( generate_dimension_plots, generate_plots)
from tsdiff.forecasting.metrics import ( get_crps )

import matplotlib.cm as cm
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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def energy_score(forecast, target):
    obs_dist = np.mean(np.linalg.norm((forecast - target), axis=-1))
    pair_dist = np.mean(
        np.linalg.norm(forecast[:, np.newaxis, ...] - forecast, axis=-1)
    )
    return obs_dist - pair_dist * 0.5

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
    ## exchange_rate_nips have daily freq, rest everything has hourly
    covariance_dim = 4 if dataset != 'exchange_rate_nips' else -4

    # Load data
    dataset = get_dataset(dataset, regenerate=False)
    ## Eg:- For exchange_rate, target_dim = 8
    target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)

    train_grouper = MultivariateGrouper(max_target_dim=min(2000, target_dim))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test) / len(dataset.train)), max_target_dim=min(2000, target_dim))
    dataset_train = train_grouper(dataset.train)
    dataset_test = test_grouper(dataset.test)

    val_window = 20 * dataset.metadata.prediction_length
    dataset_train = list(dataset_train)
    dataset_val = []
    for i in range(len(dataset_train)):
        x = deepcopy(dataset_train[i])
        x['target'] = x['target'][:,-val_window:]
        # Eg:- For exchange_rate, Dim - [8,600]
        dataset_val.append(x)
        # Eg:- For exchange_rate, Dim - [8,5471]
        dataset_train[i]['target'] = dataset_train[i]['target'][:, :-val_window]

    min_y = np.min(dataset_train[0]['target'])
    max_y = np.max(dataset_train[0]['target'])
    y_buffer = 0.2 * (max_y-min_y)  

    start_timestamp =  dataset_train[0]['start']
    end_timestamp = start_timestamp + np.timedelta64(dataset_train[0]['target'].shape[1], 'D')
    timestamps = np.arange(start_timestamp, end_timestamp, dtype='datetime64[D]')
    plt.figure(1)
    for feature_idx in range(target_dim):
        plt.plot(timestamps, dataset_train[0]['target'][feature_idx, :], label=f'Feature {feature_idx + 1}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series')
    plt.ylim(min_y-y_buffer, max_y+y_buffer)
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
        prediction_length=dataset.metadata.prediction_length,
        context_length=dataset.metadata.prediction_length,
        cell_type='GRU',
        num_cells=num_cells,
        hidden_dim=hidden_dim,
        residual_layers=residual_layers,
        input_size=target_dim * 4 + covariance_dim,
        freq=dataset.metadata.freq,
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
    # forecasts[0].samples[0] has first prediction sample
    score = energy_score(
        #(5, 100, 30, 8)
        forecast=np.array([x.samples for x in forecasts]),
        #(5, 1, 30, 8)
        target=np.array([x[-dataset.metadata.prediction_length:] for x in targets])[:,None,...],
    )
    forecast_horizon = dataset.metadata.prediction_length
    lags_seq =  lags_for_fourier_time_features_from_frequency(freq_str=dataset.metadata.freq)
    history_length = forecast_horizon + max(lags_seq)

    generate_dimension_plots(forecast=np.array([x.samples for x in forecasts]),
                             test_truth=np.array([x[-(forecast_horizon+history_length):] for x in targets])[:,None,...],
                             history_length=history_length,
                             forecast_horizon=forecast_horizon,
                             target_dim=target_dim,
                             dataset='ER_Dimension',
                             max_y=max_y,min_y=min_y,y_buffer=y_buffer)
    
    generate_dimension_plots(forecast=np.array([x.samples for x in forecasts]),
                             test_truth=np.array([x[-(forecast_horizon+history_length):] for x in targets])[:,None,...],
                             history_length=history_length,
                             forecast_horizon=forecast_horizon,
                             target_dim=target_dim,
                             dataset='ER_Dim_Sampled',
                             sample_length=5,
                             max_y=max_y,min_y=min_y,y_buffer=y_buffer)
    
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
             test_truth=np.array([x[-dataset.metadata.prediction_length:] for x in targets])[:,None,...],)


    evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

    metrics = dict(
        CRPS=agg_metric['mean_wQuantileLoss'],
        ND=agg_metric['ND'],
        NRMSE=agg_metric['NRMSE'],
        CRPS_sum=agg_metric['m_sum_mean_wQuantileLoss'],
        ND_sum=agg_metric['m_sum_ND'],
        NRMSE_sum=agg_metric['m_sum_NRMSE'],
        energy_score=score,
    )
    metrics = { k: float(v) for k,v in metrics.items() }

    return metrics


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
