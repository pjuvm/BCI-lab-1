#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_load_p300_data.py
tester for the module load_p300_data

Created on Thu Jan 18 15:25:31 2024
@author: marszzibros
"""

#%%
# define file path
data_directory = "P300Data/"
subject = 3
data_file = f"{data_directory}s{subject}.mat"


#%%
# import modules
import numpy as np
from matplotlib import pyplot as plt

import loadmat

# load training data into the dict
data = dict()
data = loadmat.loadmat(data_file)


train_data = data[f's{subject}']['train']

#%%
"""
Data Structure

egg_time    Row 0   : time (s)
egg_data    Row 1-8 : EEG data
rowcol_id   Row 9   : event type (ID) 1-6 (col), 7-12 (row)
is_target   Row 10  : 1 if current event is in target 0 otherwise
"""

eeg_time  = np.array(train_data[0])
eeg_data  = np.array(train_data[1:9])
rowcol_id = np.array(train_data[9], dtype = int)
is_target = np.array(train_data[10], dtype = bool)


#%%
# Create a figure with 1 column and 3 rows of subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
fig.suptitle(f'P300 Speller subject {subject} Raw Data', fontsize=12)

count = 0
found = False
while not found:
    if rowcol_id[count] == 0:
        count += 1
    else:
        found = True
        count = eeg_time[count] - 1

count = count // 1
# Plot data on each subplot
axs[0].plot(eeg_time, rowcol_id, label='rowcol_id')
axs[0].set_xlim(count, count + 5)
axs[0].set_ylim(-1,11)
axs[0].set_yticks(np.arange(0, 16, 5))
axs[0].set_xlabel('time (s)')
axs[0].set_ylabel('row/col ID')
axs[0].grid(True)

axs[1].plot(eeg_time, is_target, label='is_target')
axs[1].set_xlim(count, count + 5)
axs[1].set_ylim(-0.1,1.1)
axs[1].set_yticks(np.arange(0, 1.1, 0.5))
axs[1].set_xlabel('time (s)')
axs[1].set_ylabel('Target ID')
axs[1].grid(True)

axs[2].plot(eeg_time,eeg_data.T, label='egg_data')
axs[2].set_xlim(count, count + 5)
axs[2].set_ylim(-30, 30)
axs[2].set_yticks(np.arange(-25, 26, 25))
axs[2].set_xlabel('time (s)')
axs[2].set_ylabel('Voltage (uV)')
axs[2].grid(True)

plt.tight_layout()
plt.savefig(f'P300_S{subject}_training_rawdata.png')

#%%
import load_p300_data

# call load and plot functions from load_p300_data module
eeg_time, eeg_data, rowcol_id, is_target = load_p300_data.load_training_eeg(data_directory = data_directory , 
                                                                            subject = 3)
load_p300_data.plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, 3)

#%%
# call load_and_plot_all from subject 3 to 10 
load_p300_data.load_and_plot_all(data_directory=data_directory, subjects=np.arange(3,11,1))

#%%
print(load_p300_data.__doc__)
print(load_p300_data.load_training_eeg.__doc__)
print(load_p300_data.plot_raw_eeg.__doc__)
print(load_p300_data.load_and_plot_all.__doc__)

print(load_p300_data.get_word_from_subject.__doc__)
print(load_p300_data.get_intersections.__doc__)
print(load_p300_data.calculate_word_per_min.__doc__)

print(load_p300_data.ordered_intersection.__doc__)
print(load_p300_data.unique.__doc__)


#%%
# guess the single subject's word
print("subject 3")
print(load_p300_data.get_word_from_subject(3))
print()

# guess another subject's word
print("subject 4")
print(load_p300_data.get_word_from_subject(4))
print()

print("prediction of word of subject 3 and 4")
print(load_p300_data.get_intersections(subject1=3,subject2=4))
print()

# word per minute
for ind in range(3,11):
    print(f"subject {ind}: {load_p300_data.calculate_word_per_min(subject=ind):.2f} words per minute")


# a) The documentation says it uses the word "WATER" to train before the test.
#    Then, they are asked to type "LUKAS". Also, the documentations mentioned
#    how to load the data, and what information we will see in a specific row.
#    For example, explanation of is_target and rowcol_id gave me an idea of how 
#    I should utilize this information to guess what words subjects are asked to
#    type.
         
# b) The documentation gave me the matrix of p300, so I could use that information,
#    so at first, when I get the sequences of rowcol_id where is_target is 1, I
#    try to coordinate in the matrix to find what letters it is pointing to.