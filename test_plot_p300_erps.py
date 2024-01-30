#%%
import load_p300_data

# define file path
data_directory = "P300Data/"

# call load and plot functions from load_p300_data module
eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                            subject = 3)

#%%
import plot_p300_erps
event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)

# %%
eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
# %%
target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)

# %%
plot_p300_erps.plot_erps(target_erp,nontarget_erp,erp_times)

# %%
for subject in range(3, 11):
    eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                            subject = subject)
    event_sample, is_target_event = plot_p300_erps.get_events(rowcol_id, is_target)
    eeg_epochs, erp_times = plot_p300_erps.epoch_data(eeg_time, eeg_data, event_sample)
    target_erp, nontarget_erp= plot_p300_erps.get_erps(eeg_epochs, is_target_event)
    plot_p300_erps.plot_erps(target_erp,nontarget_erp,erp_times)
# %%
# 1. The repeated up-and-down patterns represent 