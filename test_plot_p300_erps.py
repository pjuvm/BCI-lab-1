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
print(event_sample.shape)
# %%
print(is_target_event.shape)
# %%
