#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_plot_p300_erps.py
test plot p300 eprs

Created on Thu Jan 18 15:25:31 2024
@author: marszzibros
"""

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
"""
1.	The up and down patterns reflect neural activations involved in processing the P300 speller task, a visual oddball task. 
First, there is a C1 component, which is a positive deflection, followed by the P1 (positive deflection) and N1 (negative deflection) wave forms, before the P3a and P3b wave forms that the BCI relies on. 
This results in the up-and-down patterns in the EEG epochs.                
2.	There are certain EEG channels that show early response to visual stimuli, such as the occipital and temporal lobe channels, which reflect activity in the primary visual cortex. 
Also, some EEG channel might not be involved as much as other channels which are used the most during the P300 speller test
3.	The positive peak half a second after the target flash corresponds to the P3b component.
This reflects higher level cognitive processes like target detection, context updating, and decision making. 
4.	The channels that have often been shown to show strong P300 responses include Cz, Pz, Oz, CPz, because of their proximity to brain regions involved in attention, evaluation, and memory, which is important for the P300 speller task.
"""
