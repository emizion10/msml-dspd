import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



def generate_plots(test_truth,forecast,history_length,forecast_horizon,target_dim,dataset,min_y,max_y,y_buffer,sample_length=100):
    colors = cm.get_cmap('tab10', target_dim)
    x_sample = np.arange(1,forecast_horizon+history_length+1)
    for sample_idx in range(len(test_truth)):
        plt.figure(sample_idx + 1,figsize=(15,6))  # Create a new figure for each sample
        truth_sample=test_truth[sample_idx][0]
        for feature_idx in range(target_dim):
            color = colors(feature_idx)
            plt.plot(
                x_sample[:],
                truth_sample[:, feature_idx],
                label=f'Feature {feature_idx + 1}',
                color=color
            )
        forecast_samples=forecast[sample_idx]    
        # for sample in range(len(forecast_samples)):
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
        plt.axvline(x=history_length, color='r', linestyle='--') 
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(f'{dataset} Time Series - Sample {sample_idx + 1}')
        plt.ylim(min_y - y_buffer, max_y + y_buffer)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{dataset}_sample_{sample_idx + 1}.png')  # Save the plot as an image
        plt.show() 



def generate_dimension_plots(test_truth,forecast,history_length,forecast_horizon,target_dim,dataset,min_y,max_y,y_buffer,sample_length=100):
    x_sample = np.arange(1,forecast_horizon+history_length+1)
    colors = cm.get_cmap('tab10', target_dim)

    # Set low opacity value
    opacity = 0.1
    ## Looping over the 5 different test sets
    for sample_idx in range(len(test_truth)):
        plt.figure(sample_idx + 1)
        forecast_samples = forecast[sample_idx]
        fig, axes = plt.subplots(target_dim, 1, figsize=(15, 2*target_dim), sharex=True)
        for dim in range(target_dim):
            color = colors(dim)
            ax = axes[dim]
            ax.set_ylabel(f"Dim {dim+1}")

            # Plotting truth values
            truth_sample=test_truth[sample_idx][0]
            ax.plot(
                x_sample[:],
                truth_sample[:, dim],
                color=color
            )
            ## Plotting 100 forecasts per sample
            # for forecast_idx in range(len(forecast_samples)):
            for forecast_idx in range(sample_length):
                forecast_sample = forecast_samples[forecast_idx, :, dim]
                ax.plot(x_sample[history_length:],forecast_sample, color=color, alpha=opacity)

            ax.axvline(x=history_length, color='r', linestyle='--') 
            ax.set_ylim(min_y - y_buffer, max_y + y_buffer)
            ax.grid(True)
            ax.set_xlabel("Time")
            ax.set_xticks(x_sample)
            ax.set_xticklabels(x_sample)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f'{dataset} Time Series - Sample {sample_idx + 1}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{dataset}_sample_{sample_idx + 1}.png')  # Save the plot as an image
        plt.show() 

def generateDataPlots(target_dim, dataset_train, dataset_val, dataset_test):
    colors = cm.get_cmap('tab10', min(target_dim,12))
    x_sample = np.arange(0,len(dataset_test.list_data[-1]['target'][0]))
    min_y = np.min(dataset_train[0]['target'])
    max_y = np.max(dataset_train[0]['target'])
    y_buffer = 0.2 * (max_y-min_y)  
    train_length = len(dataset_train[0]['target'][0])
    val_length = len(dataset_val[0]['target'][0])
    plt.figure(100)
    for dim in range(target_dim):
        if(target_dim>10):
            feature_idx = (dim+1) * (target_dim//10)
        else:
            feature_idx = dim
        if feature_idx< target_dim:
            color = colors(dim)
            plt.plot(x_sample, dataset_test.list_data[-1]['target'][feature_idx, :], color=color, alpha=0.7)
            plt.plot(x_sample[train_length:train_length+val_length], dataset_val[0]['target'][feature_idx, :], color=color, alpha=0.5)
            plt.plot(x_sample[:train_length], dataset_train[0]['target'][feature_idx, :], label=f'Feature {feature_idx + 1}', color=color)
        else:
            break
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Multivariate Time Series')
    plt.ylim(min_y-y_buffer, max_y+y_buffer)
    plt.axvline(x=train_length, color='r', linestyle='--') 
    plt.axvline(x=train_length+val_length, color='b', linestyle='--') 
    plt.legend()
    plt.grid(True)
    plt.savefig('train_data.png')
    plt.show()
    plotHistogram(target_dim, dataset_train, dataset_val, dataset_test)

def plotHistogram(target_dim, dataset_train, dataset_val, dataset_test):
        colors = cm.get_cmap('tab10', target_dim) 
        for dim in range(target_dim):
            if(target_dim>10):
                feature_idx = (dim+1) * (target_dim//10)
            else:
                feature_idx = dim
            if feature_idx< target_dim:
                plt.figure(dim + 1)
                fig, axes = plt.subplots(1,3, figsize=(6, 2))
                # color = colors(dim)
                min_y = np.min(dataset_train[0]['target'][feature_idx, :])
                max_y = np.max(dataset_train[0]['target'][feature_idx, :])
                axes[0].hist(dataset_train[0]['target'][feature_idx, :],bins=20, color='blue', alpha=0.7) 
                axes[0].set_xlim([min_y, max_y])
                axes[0].set_title('Training Data')

                axes[1].hist(dataset_val[0]['target'][feature_idx, :],bins=20, color='green', alpha=0.7) 
                axes[1].set_xlim([min_y, max_y])
                axes[1].set_title('Validation Data')

                axes[2].hist(dataset_test.list_data[-1]['target'][feature_idx, :],bins=20, color='red', alpha=0.7)  
                axes[2].set_xlim([min_y, max_y])
                axes[2].set_title('Test Data')

                plt.suptitle(f'Time Series ')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                # plt.savefig(f'{dataset}_sample_{sample_idx + 1}.png')  # Save the plot as an image
                plt.show() 


# forecast_horizon=30
# history_length=32
# target_dim=4

# x_sample = np.arange(0,forecast_horizon+history_length)
# test_truth = np.random.rand(5,1,62,target_dim)
# forecast = np.random.rand(5,100,30,target_dim)

# min_y = np.min(test_truth)
# max_y = np.max(test_truth)
# y_buffer = 0.2 * (max_y-min_y)  

# generate_dimension_plots(forecast=forecast,test_truth=test_truth,history_length=history_length,forecast_horizon=forecast_horizon,target_dim=target_dim, dataset='ER_Dimension',max_y=max_y,min_y=min_y,y_buffer=y_buffer)
# generate_plots(forecast=forecast,test_truth=test_truth,history_length=history_length,forecast_horizon=forecast_horizon,target_dim=target_dim, dataset='ER Multivariate',max_y=max_y,min_y=min_y,y_buffer=y_buffer)
