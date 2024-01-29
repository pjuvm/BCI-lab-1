#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_p300_data.py
loads and plots p300 data

Created on Thu Jan 18 15:25:31 2024
@author: marszzibros
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import reduce

import loadmat

def load_training_eeg(data_directory = "P300Data/", subject = 3):
    """
    load_training_eeg

    load p300 eeg data and return four arrays eeg_time, eeg_data, rowcol_id, is_target

    data_directory (string) : directory path to the p300 data  
    subject           (int) : subject of the data

    """

    # define file path
    data_file = f"{data_directory}s{subject}.mat"

    # load training data into the dict
    data = dict()
    data = loadmat.loadmat(data_file)

    train_data = data[f's{subject}']['train']

    
    # Data Structure

    # egg_time    Row 0   : time (s)
    # egg_data    Row 1-8 : EEG data
    # rowcol_id   Row 9   : event type (ID) 1-6 (col), 7-12 (row)
    # is_target   Row 10  : 1 if current event is in target 0 otherwise


    # save data into seperate variables 
    eeg_time  = np.array(train_data[0])
    eeg_data  = np.array(train_data[1:9])
    rowcol_id = np.array(train_data[9], dtype = int)
    is_target = np.array(train_data[10], dtype = bool)

    # return four numpy arrays
    return eeg_time, eeg_data, rowcol_id, is_target

def plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject):
    """
    plot_raw_eeg

    plot p300 eeg data

    eeg_time    (narray) : eegtime
    eeg_data    (narray) : eeg data
    rowcol_id   (narray) : (int) row/col id
    is_target   (narray) : (bool) target
    subject        (int) : subject to the p300 data

    """

    # find first flash of rowcol_id
    count = 0
    found = False
    while not found:
        if rowcol_id[count] == 0:
            count += 1
        else:
            found = True
            # one second before
            count = eeg_time[count] - 1

    # Create a figure with 1 column and 3 rows of subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f'P300 Speller subject {subject} Raw Data', fontsize=12)

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

def load_and_plot_all(data_directory, subjects):
    """
    load_and_plot_all

    load and plot given subjects collectively

    data_directory (string) : directory path to the p300 data  
    subjects          (int) : subjects of the data

    """
    # run multiple subjects using for loop
    for subject in subjects:
        eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(data_directory = data_directory , 
                                                                                    subject = subject)
        plot_raw_eeg(eeg_time, eeg_data, rowcol_id, is_target, subject=subject)



# p300 speller setup
matrix = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "0", "1", "2", "3"],
    ["4", "5", "6", "7", "8", "9"]]

matrix = np.array(matrix)
coord = np.array([matrix[:,0].T,
                  matrix[:,1].T,
                  matrix[:,2].T,
                  matrix[:,3].T,
                  matrix[:,4].T,
                  matrix[:,5].T,
                  matrix[0,:],
                  matrix[1,:],
                  matrix[2,:],
                  matrix[3,:],
                  matrix[4,:],
                  matrix[5,:],])


# https://stackoverflow.com/questions/9792664/converting-a-list-to-a-set-changes-element-order
def unique(sequence):
    """
    remove repeated words
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def ordered_intersection(arr1, arr2):
    """
    Custom function to find the ordered intersection
    """
    return [elem for elem in arr1 if elem in arr2]


def get_word_from_subject(subject = 3):
    """
    extract words from eeg data
    """
    # get data
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject=subject)
    temp_dict = {"eeg_time" : eeg_time,
                 "rowcol_id" : rowcol_id,
                 "is_target" : is_target}
    data = pd.DataFrame(temp_dict)               

    # get eeg data with is_target == 1            
    data_processed = data[data['is_target'] == 1]

    # process rowcol_id 
    # remove repeated rowcol_id
    prev_id = data_processed["rowcol_id"].iloc[0]
    rowcol_id_list = []
    rowcol_id_list.append(prev_id)

    for ind in range(1, len(data_processed["rowcol_id"])):
        if data_processed["rowcol_id"].iloc[ind] != prev_id:
            prev_id = data_processed["rowcol_id"].iloc[ind]
            rowcol_id_list.append(prev_id)


    # rowcol_id to coordinate id index conversion        
    rowcol_id_list = np.array(rowcol_id_list) - 1


    # get word from rowcol_id
    word_list = []

    previous_1 = rowcol_id_list[0]
    previous_2 = rowcol_id_list[1]    

    for ind in range(0, len(rowcol_id_list), 2):
        if ind + 2 >= len(rowcol_id_list):
            pass
        
        # get row and col in the matrix and extract the word
        # check if repeated rowcol_id is matching with previous one to prevent the error
        # find the word combination
        elif rowcol_id_list[ind] == previous_1 and rowcol_id_list[ind + 1] == previous_2 and ((rowcol_id_list[ind] > 5 and rowcol_id_list[ind + 1] < 6) or (rowcol_id_list[ind + 1] > 5 and rowcol_id_list[ind] < 6)):
            word_list.append(np.intersect1d(coord[rowcol_id_list[ind]],coord[rowcol_id_list[ind + 1]])[0])
        # if not skip several rowcol_id
        else:
            ind += 4
            previous_1 = rowcol_id_list[ind]
            previous_2 = rowcol_id_list[ind+1]            

    word_list = np.array(word_list)
    word_list = unique(word_list)

    return word_list

def get_intersections (subject1 = 3, subject2 = 4):
    """
    get intersection of two subjects to guess the word they are asked to type
    """
    subject1_word = get_word_from_subject(subject1)
    subject2_word = get_word_from_subject(subject2)

    return reduce(ordered_intersection, [subject1_word, subject2_word])

def calculate_word_per_min (subject = 3):
    """
    calculate word per minute using the eeg time and word count
    """
    eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject=subject)
    word = get_word_from_subject(subject=subject)

    word_length = len(word)
    time = eeg_time[-1] / 60

    return word_length / time
