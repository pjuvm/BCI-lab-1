#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:00:05 2024

filter_ssvep_data.py

This module contains functions to perform filtering and freqeuncy analysis on SSVEP EEG data. 

Useful abbreviations:
    EEG: electroencephalography
    SSVEP: steady-state visual evoked potentials
    fs: sampling frequency
    FFT: Fast Fourier Transform
    FIR: finite impulse response

@authors: Peijin Chen and Marit Scott

Sources:
    - Used Chat GPT to inform subplot plotting on linked axes
    - Used Chat GPT to help with makign the conditionals to account for channel_to_plot and ssvep inputs
    - Used Chat GPT to improve the efficiency of the part 6 function 

"""

# import packages
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import * 

#TODO: DOCSTRINGS


#%% Part 2: Design a Filter
def make_bandpass_filter(low_cutoff, high_cutoff, filter_type = "hann", filter_order=10, fs=1000):
    """
    Create a bandpass filter and plot its impulse and frequency response.

    Parameters:
        low_cutoff (float): The low cutoff frequency of the bandpass filter.
        high_cutoff (float): The high cutoff frequency of the bandpass filter.
        filter_type (str, optional): The window type to use for filter design. Defaults to "hann".
        filter_order (int, optional): The order of the filter. Defaults to 10.
        fs (int, optional): The sampling frequency. Defaults to 1000.

    Returns:
        filter_coefficients (1D array of float, length order + 1): The filter coefficients from the FIR filter
    """

    # create an FIR filter with 
    filter_coefficients = firwin(numtaps = filter_order + 1, 
                                 cutoff = [low_cutoff,high_cutoff], 
                                 window = filter_type, 
                                 pass_zero = False,
                                 fs = fs)
    
    # Calculate frequency response (h)
    w, h = freqz(filter_coefficients)
    magnitude_db = 20 * np.log10(np.abs(h))
    
    # Time axis for impulse response
    t = np.arange(len(filter_coefficients)) / fs
    
    # Frequency axis for frequency response
    f = w * fs / (2 * np.pi)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot impulse response
    ax1.plot(t, filter_coefficients)
    
    # annotate subplot
    ax1.set_title('Impulse Response')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Gain')
    ax1.grid()
    
    # Plot frequency response
    ax2.plot(f, magnitude_db)
    
    # annotate subplot
    ax2.set_title('Frequency Response')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude (dB)')
    ax2.grid()
    
    plt.tight_layout()
    plt.show()
    
    # save figure
    figure_name = f"{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}"
    plt.savefig(figure_name)
    
    
    return filter_coefficients

#%% Part 3: Filter the EEG Signals

def filter_data(data, b):
    """
    Apply a forward-backward filter to each channel of EEG data.

    Parameters:
        data (dict): A dictionary containing raw EEG data.
        b (1D array of float, length order + 1): The filter coefficients from the FIR filter

    Returns:
        filtered_data (2D array of float, size channels x samples): Filtered EEG data in microvolts.
    """
    # Extract EEG data from the dictionary
    raw_data = data['eeg']
    
    # Number of channels and samples
    channel_count, sample_count = raw_data.shape
    
    # Initialize filtered data array
    filtered_data = np.zeros_like(raw_data)
    
    # Apply filter to each channel
    for channel_index in range(channel_count):
        # Apply forward-backward filter to the current channel
        filtered_data[channel_index] = filtfilt(b, 1, raw_data[channel_index])
    
    return filtered_data



#%% Part 4: Calculate the envelope

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency=None):
    """
    Compute the envelope of the filtered EEG data and plot it.

    Parameters:
        data (dict): A dictionary containing raw EEG data and other information about the dataset.
        filtered_data (channels x samples array of floats): The filtered EEG data.
        channel_to_plot (str, optional): The channel to plot. Defaults to None.
        ssvep_frequency (float, optional): The SSVEP frequency being isolated. Defaults to None.

    Returns:
        envelope (channels x samples 2D array of floats): The amplitude of oscillations on every channel at every time point.
    """
    # Extract EEG data from the data dictionary
    raw_data = data['eeg']
    
    # Get list of channels from the data dictionary 
    channel_list = list(dict(data)['channels'])
    
    # get fs from the data_dictionary
    fs = data['fs']
    
    
    # Compute the Hilbert transform to get the envelope
    envelope = np.abs(hilbert(filtered_data))
    
    # used chat gpt to help with makign the conditionals to account for channel_to_plot and ssvep inputs
    if channel_to_plot is not None:
        # Find the channel index
        channel_index = channel_list.index(channel_to_plot)
        
        
        # Create a time axis 
        t = np.arange(len(raw_data[channel_index])) / fs
        
        # Plot the channel's data and envelope
        plt.figure()
        plt.plot(t, filtered_data[channel_index], label='Filtered Signal')
        plt.plot(t, envelope[channel_index], label='Envelope')
        #plt.plot(t, raw_data[channel_index], label='Raw Signal')
        
        
        # annotate plot
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (uV)')
        plt.title(f'Channel {channel_to_plot} with Envelope')
        plt.legend()
        plt.show()
    
        # Set title for SSVEP frequency
        if ssvep_frequency is None:
            plt.title(f"Unknown SSVEP Frequency {channel_to_plot} data")
        else:
            plt.title(f"{ssvep_frequency} Hz {channel_to_plot} data")
    
  
    
    return envelope

#%% Part 5: Plot the Amplitudes

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    """
    Plot the envelopes of oscillations for two SSVEP frequencies and event start & end times.

    Parameters:
        data (dict):  A dictionary containing raw EEG data and other information about the dataset.
        envelope_a (channels x samples 2D array of floats): The amplitude of oscillations on every channel at every time point for the first frequency.
        envelope_b (channels x samples 2D array of floats): The amplitude of oscillations on every channel at every time point for the second frequency.
        channel_to_plot (str): The EEG channel to plot.
        ssvep_freq_a (int): The first SSVEP frequency being isolated.
        ssvep_freq_b (int): The second SSVEP frequency being isolated.
        subject (int): The subject number.

    Returns:
        None
    """
    # Get channel index
    channel_list = list(dict(data)['channels'])
    channel_index = channel_list.index(channel_to_plot)
  
    # Extract variables from data dictionary 
    fs = data['fs']
    event_durations = data['event_durations']
    event_samples = data['event_samples']
    event_types = data['event_types']
    
    # Establish time axis
    t = np.arange(len(envelope_a[channel_index])) / fs 
  
    
    # find event samples and times
    event_intervals = np.zeros([len(event_samples),2]) # array to contain interval times
    # find the end sample indices of the events
    event_ends = event_samples + event_durations # event_samples contains the start samples
    # for each event (:), 0 is the interval start time, and 1 is the interval end time
    event_intervals[:,0] = event_samples/fs # convert start samples to times
    event_intervals[:,1] = event_ends/fs # convert end samples to times
    
    
    # initialize figure
    figure, axes = plt.subplots(2, sharex=True)
    
    # top subplot containing flash frequency over span of event
    for event_number, interval in enumerate(event_intervals):
    
        # determine the event frequency to plot
        if event_types[event_number] == "12hz":
            event_frequency = 12
    
        else: 
            event_frequency = 15
        
        # plottting the event frequency
        axes[0].hlines(xmin=interval[0], xmax=interval[1], y=event_frequency, label='o') # line
        axes[0].plot([interval[0], interval[1]], [event_frequency,event_frequency], 'bo') # start and end markers
   
    # annotate subplot
    axes[0].set_title(f"Subject {subject} SSVEP Amplitudes")
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Flash Frequency (Hz)')

    # Plot envelopes
    axes[1].plot(t, envelope_a[channel_index,:], label=f'{ssvep_freq_a} Hz')
    axes[1].plot(t, envelope_b[channel_index,:], label=f'{ssvep_freq_b} Hz')
    
    # annotate subplot 
    axes[1].set_title('Envelope Comparison')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Voltage (uV)')
    axes[1].legend()

    # annotate figure
    plt.tight_layout()
    plt.show()

#%% Part 6: 
  
# Used Chat GPT to imporve the efficiency and keep track of all the things to plot 


def plot_filtered_spectra(data, filtered_data, envelope, channels=['Fz', 'Oz']):
    """
    Plot the average power spectra across epochs for the specified channels at three stages of analysis:
    raw data, filtered data, and the envelope of the filtered data.

    Parameters:
        data (dict): A dictionary containing raw EEG data.
        filtered_data (channels x samples 2D array of floats): The filtered EEG data.
        envelope (channels x samples 2D array of floats): The amplitude of oscillations on every channel at every time point.
        channels (list): A list of EEG electrode channel names to plot. Default is ['Fz', 'Oz'].

    Returns:
        None
    """
    # Get channel indices
    channel_list = list(dict(data)['channels'])
    channel_indices = [channel_list.index(channel) for channel in channels]

    # Define the frequency axis
    sample_count = len(filtered_data[0])  # Length of the signal
    fs = data['fs']  # Sampling frequency
    f = np.fft.rfftfreq(sample_count, 1/fs)  # Frequency axis (one-sided)

    # establish figure
    fig, axes = plt.subplots(len(channels), 3, figsize=(15, 4 * len(channels)), sharex=True)

    for channel_index, channel_number in enumerate(channel_indices):
        
        # get name of the electrode
        channel_name = channel_list[channel_number]

        # Compute power spectra for raw data
        raw_power = np.abs(np.fft.rfft(data['eeg'][channel_index]))**2 / sample_count
        raw_power_norm = raw_power / np.max(raw_power)
        raw_power_db = 10 * np.log10(raw_power_norm)

        # Compute power spectra for filtered data
        filtered_power = np.abs(np.fft.rfft(filtered_data[channel_index]))**2 / sample_count
        filtered_power_norm = filtered_power / np.max(filtered_power)
        filtered_power_db = 10 * np.log10(filtered_power_norm)

        # Compute power spectra for envelope data
        envelope_power = np.abs(np.fft.rfft(envelope[channel_index]))**2 / sample_count
        envelope_power_norm = envelope_power / np.max(envelope_power)
        envelope_power_db = 10 * np.log10(envelope_power_norm)

        # Plot raw data power spectra
        axes[channel_index, 0].plot(f, raw_power_db)
        axes[channel_index, 0].set_title(f'{channel_name} - Raw Data')
        axes[channel_index, 0].set_ylabel('Power (dB)')
        axes[channel_index, 0].set_xlabel('Frequency (Hz)')

        # Plot filtered data power spectra
        axes[channel_index, 1].plot(f, filtered_power_db)
        axes[channel_index, 1].set_title(f'{channel_name} - Filtered Data')
        axes[channel_index, 1].set_ylabel('Power (dB)')
        axes[channel_index, 1].set_xlabel('Frequency (Hz)')

        # Plot envelope data power spectra
        axes[channel_index, 2].plot(f, envelope_power_db)
        axes[channel_index, 2].set_title(f'{channel_name} - Envelope Data')
        axes[channel_index, 2].set_ylabel('Power (dB)')
        axes[channel_index, 2].set_xlabel('Frequency (Hz)')

    # annotate plot 
    plt.tight_layout()
    plt.show()
    
    # save figure
    figure_name = f"Filtered_Spectra_{channels}"
    plt.savefig(figure_name)
    
