import numpy as np
from matplotlib import pyplot as plt

def get_events(rowcol_id, is_target):
    event_sample = np.array(np.where(np.diff(rowcol_id) > 0)) + 1
    is_target_event = np.array([True if is_target[ind] else False for ind in event_sample[0]])
    return event_sample, is_target_event

def epoch_data (eeg_time, eeg_data, event_sample, epoch_start_time = -0.5, epoch_end_time=1):

    samples_per_second = 0
    seconds_per_epoch = epoch_end_time - epoch_start_time

    while eeg_time[samples_per_second] <= 1:
        samples_per_second += 1

    samples_per_epoch = (samples_per_second - 1)* seconds_per_epoch

    event_before = int(round((abs(epoch_start_time) / seconds_per_epoch) * samples_per_epoch))
    event_after = int(round((abs(epoch_end_time) / seconds_per_epoch) * samples_per_epoch))

    eeg_epochs = np.empty((event_sample.shape[1], int(samples_per_epoch), eeg_data.shape[0]), dtype = object)
    erp_times = np.linspace(epoch_start_time,epoch_end_time,int(samples_per_epoch))

    for ind, event in enumerate(event_sample[0]):
        epoch = []

        for eeg_time_ind in range(event - event_before, event + event_after):
            epoch.append(eeg_data.T[eeg_time_ind])
        eeg_epochs[ind] = np.array(epoch)

    return eeg_epochs, erp_times

def get_erps (eeg_epochs, is_target_event):
    target_erp = np.mean(eeg_epochs[is_target_event], axis = 0)
    nontarget_erp = np.mean(eeg_epochs[~is_target_event], axis = 0)
    return target_erp, nontarget_erp


def plot_erps(target_erp, nontarget_erp, erp_times):
    # Reshape the data for subplot arrangement
    target_erp = target_erp.T  # Transpose to have shape (8, 150)
    nontarget_erp = nontarget_erp.T  # Transpose to have shape (8, 150)

    # Create a 3x3 subplot grid for 8 plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))

    # Plot each subplot
    for i in range(3):
        for j in range(3):
            plot_index = i * 3 + j
            ax = axes[i, j]
            # Check if there are fewer than 8 plots
            if plot_index < 8:

                
                # Plot target ERP
                ax.plot(erp_times, target_erp[plot_index], label='Target')
                
                # Plot nontarget ERP
                ax.plot(erp_times, nontarget_erp[plot_index], label='Nontarget')
                
                # Add a dotted line at x=0 and y=0
                ax.axvline(x=0, linestyle='--', color='black', linewidth=1)
                ax.axhline(y=0, linestyle='--', color='black', linewidth=1)

                # Set y-axis ticks
                ax.set_yticks([-2, 0, 2])
                
                # Set x-axis ticks
                ax.set_xticks(np.linspace(erp_times[0], erp_times[-1], 4))                

                # Set subplot title and labels
                ax.set_title(f'Channel {plot_index}', fontsize=14)
                ax.set_xlabel('time from flash onset( s)')
                ax.set_ylabel('Voltage (uV)')
                
                # Add legend to the last subplot
                if plot_index == 7:
                    ax.legend()

            # Remove empty subplots
            else:
                fig.delaxes(ax)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()