import numpy as np

# The grand truth (actual) is expected to be the shape of (batch_size,length,channels) and forecast be the form of (number_of_smaples,batch_size,length,channels). for each forecast, 100 samples should be sufficent.

def calc_crps_sample(actual, forecast):
    rng = np.random.default_rng()
    shuffled_forecast = rng.permutation(forecast)
    return np.absolute(forecast - actual).mean(axis=0) - 0.5 * np.absolute(forecast - shuffled_forecast).mean(axis=0)

# target_dim=1

# test_truth = np.random.rand(7,1,32,target_dim)
# forecast = np.random.rand(7,100,32,target_dim)

def get_crps(forecast,test_truth,print_results=True):
    num_samples = len(test_truth)
    target_dim = test_truth.shape[-1] 
    forecast_horizon = test_truth.shape[-2]   
    forecast_reshaped = forecast.transpose(1, 0, 2, 3).reshape(100, num_samples, forecast_horizon, target_dim)
    test_truth_reshaped = test_truth.reshape(num_samples,-1,target_dim)
    crps_array= calc_crps_sample(test_truth_reshaped,forecast_reshaped)
    crps = np.mean(crps_array)
    crps_mean_sample = np.mean(crps_array,axis=1)
    if(print_results):
        print('Custom CRPS Score', crps)
        print('CRPS along samples',crps_mean_sample)
    return crps


# get_crps(forecast=forecast,test_truth=test_truth)