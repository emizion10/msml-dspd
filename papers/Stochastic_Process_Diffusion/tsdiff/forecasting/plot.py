import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def generate_plots(test_truth,forecast,history_length,forecast_horizon,target_dim,dataset,min_y,max_y,y_buffer):
    generateForecastPlots(forecast=forecast,
                   test_truth=test_truth,
                   history_length=history_length,
                   forecast_horizon=forecast_horizon,
                   target_dim=target_dim,
                   dataset=dataset,
                   max_y=max_y,min_y=min_y,y_buffer=y_buffer)
    
    generateForecastPlots(forecast=forecast,
                   test_truth=test_truth,
                   history_length=history_length,
                   forecast_horizon=forecast_horizon,
                   target_dim=target_dim,
                   dataset=f'{dataset}_Sampled',
                   sample_length=5,
                   max_y=max_y,min_y=min_y,y_buffer=y_buffer)
    generateQuantilePlots(forecast=forecast,test_truth=test_truth,history_length=history_length,forecast_horizon=forecast_horizon,target_dim=target_dim, dataset=dataset,max_y=max_y,min_y=min_y,y_buffer=y_buffer)
    generateForecastHistograms(forecast=forecast,test_truth=test_truth, dataset=dataset,max_y=max_y,min_y=min_y)



def generateForecastPlots(test_truth,forecast,history_length,forecast_horizon,target_dim,dataset,min_y,max_y,y_buffer,sample_length=100):
    colors = cm.get_cmap('tab10', target_dim)
    x_sample = np.arange(1,forecast_horizon+history_length+1)
    for sample_idx in range(len(test_truth)):
        plt.figure(sample_idx + 1,figsize=(15,6))  # Create a new figure for each sample
        # Plotting forecast values
        forecast_samples=forecast[sample_idx]    
        for sample in range(sample_length):
            forecast_sample = forecast_samples[sample]
            for feature_idx in range(target_dim):
                color = colors(feature_idx)
                plt.plot(
                    x_sample[history_length:],
                    forecast_sample[:, feature_idx],
                    color=color,
                    alpha=0.2
                )

        # Plotting truth values
        truth_sample=test_truth[sample_idx][0]
        for feature_idx in range(target_dim):
            color = colors(feature_idx)
            plt.plot(
                x_sample[:],
                truth_sample[:, feature_idx],
                label=f'Feature {feature_idx}',
                color=color
            )
        plt.axvline(x=history_length, color='r', linestyle='--') 
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{dataset} Time Series - Sample {sample_idx + 1}')
        plt.ylim(min_y - y_buffer, max_y + y_buffer)
        plt.grid(True)
        plt.savefig(f'{dataset}_sample_{sample_idx + 1}.png')  # Save the plot as an image
        plt.show() 
    
def generateDataPlots(target_dim, dataset_train, dataset_val, dataset_test,dataset_name):
    colors = cm.get_cmap('tab10', min(target_dim,12))
    x_sample = np.arange(0,len(dataset_test.list_data[-1]['target'][0]))
    min_y = np.min(dataset_train[0]['target'])
    max_y = np.max(dataset_train[0]['target'])
    y_buffer = 0.2 * (max_y-min_y)  
    train_length = len(dataset_train[0]['target'][0])
    val_length = len(dataset_val[0]['target'][0])
    plt.figure(100)
    for feature_idx in range(target_dim):
        color = colors(feature_idx)
        plt.plot(x_sample, dataset_test.list_data[-1]['target'][feature_idx, :], color=color, alpha=0.7)
        plt.plot(x_sample[train_length:train_length+val_length], dataset_val[0]['target'][feature_idx, :], color=color, alpha=0.5)
        plt.plot(x_sample[:train_length], dataset_train[0]['target'][feature_idx, :], label=f'Feature {feature_idx}', color=color)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'{dataset_name} Univariate Time Series')
    plt.ylim(min_y-y_buffer, max_y+y_buffer)
    plt.axvline(x=train_length, color='r', linestyle='--') 
    plt.axvline(x=train_length+val_length, color='b', linestyle='--') 
    plt.grid(True)
    plt.savefig('train_data.png')
    plt.show()
    plotHistogram(target_dim, dataset_train, dataset_val, dataset_test,dataset_name)

def plotHistogram(target_dim, dataset_train, dataset_val, dataset_test,dataset_name):
        colors = cm.get_cmap('tab10', target_dim) 
        train_data = dataset_train[0]['target']
        val_data = dataset_val[0]['target']
        test_data = dataset_test.list_data[-1]['target']
        train_and_val_limit = train_data.shape[-1]+ val_data.shape[-1]
        plt.figure()
        fig, axes = plt.subplots(1,3, figsize=(6, 2))
        for feature_idx in range(target_dim):
            min_y = np.min(train_data[feature_idx, :])
            max_y = np.max(train_data[feature_idx, :])
            axes[0].hist(train_data[feature_idx, :],bins=20, color='blue', alpha=0.7) 
            axes[0].set_xlim([min_y, max_y])
            axes[0].set_title(f'Feature {feature_idx}')

            axes[1].hist(val_data[feature_idx, :],bins=20, color='green', alpha=0.7) 
            axes[1].set_xlim([min_y, max_y])
            # axes[1].set_title('Validation Data')

            axes[2].hist(test_data[feature_idx,train_and_val_limit:],bins=20, color='red', alpha=0.7)  
            axes[2].set_xlim([min_y, max_y])
            # axes.set_title('Test Data')


        plt.suptitle(f'Time Series Data Distribution')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{dataset_name}_Histogram.png')  # Save the plot as an image
        plt.show() 


def generateQuantilePlots(test_truth,forecast,history_length,forecast_horizon,target_dim,dataset,min_y,max_y,y_buffer):
    x_sample = np.arange(1,forecast_horizon+history_length+1)
    for sample_idx in range(len(test_truth)):
        plt.figure(sample_idx + 1,figsize=(10,4))  # Create a new figure for each sample
        # Plotting forecast values
        forecast_samples=forecast[sample_idx]  

        # Calculate quantiles (e.g., 10th, 50th, and 90th percentiles)
        quantiles = np.percentile(forecast_samples, [25, 50, 75], axis=0)

        # Calculate the median and mean
        median = np.median(forecast_samples, axis=0)
        mean = np.mean(forecast_samples, axis=0)  
        min_values = np.min(forecast_samples, axis=0)
        max_values = np.max(forecast_samples, axis=0)

        plt.plot(x_sample[history_length:], quantiles[0, :, 0], label='25th Percentile', color='red', linestyle='--')
        plt.plot(x_sample[history_length:], quantiles[2, :, 0], label='75th Percentile', color='green', linestyle='--')
        plt.fill_between(x_sample[history_length:], min_values[:, 0], max_values[:, 0], alpha=0.2, color='gray', label='Min-Max Range')
        plt.plot(x_sample[history_length:], median[:, 0], label='Median', color='orange')
        plt.plot(x_sample[history_length:], mean[:, 0], label='Mean', color='purple')

        # Plotting truth values
        truth_sample=test_truth[sample_idx][0]
        for feature_idx in range(target_dim):
            plt.plot(
                x_sample[:],
                truth_sample[:, feature_idx],
                label=f'Truth Value',
                color='blue'
            )
        plt.axvline(x=history_length, color='r', linestyle='--') 
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{dataset} Forecast(Quantiles) - Sample {sample_idx + 1}')
        plt.ylim(min_y - y_buffer, max_y + y_buffer)
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{dataset}_Quantiles_Sample_{sample_idx + 1}.png')  # Save the plot as an image
        plt.show() 

def generateForecastHistograms(test_truth,forecast,dataset,min_y,max_y):
    plt.figure()  
    for sample_idx in range(len(test_truth)):
        forecast_samples=forecast[sample_idx]  
        mean = np.mean(forecast_samples, axis=0)  
        plt.hist(mean[:], color='orange', alpha=0.7) 

    plt.xlim([min_y, max_y])
    plt.title(f'{dataset} Forecast Histogram')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{dataset}_Forecast_Histogram.png')  # Save the plot as an image
    plt.show() 


# forecast_horizon=30
# history_length=32
# target_dim=1

# x_sample = np.arange(0,forecast_horizon+history_length)
# test_truth = np.random.rand(5,1,62,target_dim)
# forecast = np.random.rand(5,100,30,target_dim)

# min_y = np.min(test_truth)
# max_y = np.max(test_truth)
# y_buffer = 0.2 * (max_y-min_y)  

# generateForecastHistograms(forecast=forecast,test_truth=test_truth,dataset='ER Multivariate',max_y=max_y,min_y=min_y)
