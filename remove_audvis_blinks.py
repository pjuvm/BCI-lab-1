#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:00:30 2024

@author: Alaina Birney

A module that contains functions to load raw EEG data and associated metadata,
plot scalp maps (topomaps) for specified components by calling plot_topo from
https://github.com/djangraw/BCIs-S24/blob/main/plot_topo.py, transform EEG
data into source space, remove artifacts and transform data back to electrode
space, and plot the raw, reconstructed, and cleaned eeg data in one set of subplots.
"""
import matplotlib.pyplot as plt
import numpy as np
import plot_topo as plot_topo_maps

#%% Part 1: Load Data

def load_data(data_directory='./AudVisData', channels_to_plot=None):
    """
    Loads the raw data from an .npy file and plots the raw eeg data for specified
    channels if channels are specified. It is assumed that the name of the raw 
    data file is "AudVisData.npy"
    
    Parameters
    ----------
    data_directory : str, optional
        The directory where the EEG data is stored. The default is './AudVisData'.
    channels_to_plot: List of str, optional
        The channels to plot raw EEG data for. The default is None. If the default
        is kept, no plots will be created.

    Returns
    -------
    data : dict
        Dictionary containing the EEG data and metadata. The structure is as follows:
        - 'eeg': 2D array of float. Size (channels x samples).
            EEG data in uV.
        - 'channels': 1D array of str.
            The names of EEG channels. Order matches the EEG matrix such that
            EEG[0,:] is EEG data from channels[0]
        - 'fs': float
            Sampling frequency in Hz.
        - 'event_samples': 1D array of int. Size (E,) where E is the number of
        distinct events.
            The samples where each event occurred.
        - 'event_types': 1D array of int. Size (E,) where E is the number of 
        distinct events.
            The event code for each event. 1 = left ear auditory tone, 2 = right
            ear auditory tone, 3 = left side visual checkerboard, 4 = right side
            visual checkerboard, 5 = smiley face, 32 = button press in response
            to smiley face. Indices correspond to event samples such that 
            event_types[i] is the event type that occured at sample event_sample[i]

    """
    data = np.load(f"{data_directory}/AudVisData.npy", allow_pickle=True).item()
    eeg = data['eeg']
    channels = data['channels']
    fs = data["fs"]
    event_samples = data["event_samples"]
    event_types = data["event_types"]
    unmixing_matrix = data["unmixing_matrix"]
    mixing_matrix = data["mixing_matrix"]
    
    data = {
        'eeg': eeg,
        'channels': channels,
        'fs': fs,
        'event_samples': event_samples,
        'event_types': event_types,
        "unmixing_matrix": unmixing_matrix,
        "mixing_matrix": mixing_matrix
    }
        
    if channels_to_plot is None:
        # return dataset in dictionary if channels_to_plot is empty
        return data
    else:
        # if channels_to_plot not empty, plot raw data from each electrode in a
        # separate subplot
        # get time
        T = np.arange(eeg.shape[1]) / fs # time axis (seconds)
        # initialize variable for indexing axs to make subplots
        axs_idx = 0

        # initialize figure with as many rows as channels_to_plot and 1 column
        fig, axs = plt.subplots(len(channels_to_plot),1, figsize=(10,5), sharex = True)
        fig.suptitle("Raw AudVis EEG Data")
        
        for channel_to_plot in channels_to_plot:
            channel_to_plot_index = None
            # find index of current channel in channels_to_plot
            for channel_index, channel_value in enumerate(channels):
                if channel_value == channel_to_plot:
                    channel_to_plot_index = channel_index
                    break
            # use that index to get raw data
            eeg_to_plot = eeg[channel_to_plot_index]
            # plot
            axs[axs_idx].plot(T, eeg_to_plot)
            axs[axs_idx].set_xlabel("time (s)")
            axs[axs_idx].set_ylabel(f"Voltage on {channel_to_plot} (uV)")
            axs_idx +=1 # incrememnt index for axs
        
        plt.tight_layout()
        plt.savefig(f"Raw_eeg_data_channels_{channels_to_plot}.png")
            
        

        return data
    
#%% Part 2: Plot the Components

def plot_components(mixing_matrix, channels, components_to_plot):
    """
    Plots each specified component in components_to_plot in a separate subplot.
    Calls plot_topo (https://github.com/djangraw/BCIs-S24/blob/main/plot_topo.py)
    to produce topomaps.
    
    Parameters
    ----------
    mixing_matrix : Array of float. Size (C, C) where C is the number of channels
    in the EEG data.
        ICA components. Rows are weights used to combine across electrodes to 
        get activity for a single source. Each column is the impact of a source
        on all electrodes.
    channels : Array of str. Size (C,) where C is the number of channels in the
    EEG data.
        The names of EEG channels.
    components_to_plot : Array of int. Size (Co,) where Co is the number of 
    components to plot. 
        ICA components to produce topomaps for. 

    Returns
    -------
    None.

    """
    # plot each component in the list in a separate subplot

    # convert channels to list
    channels = list(channels)
        
    # plot two rows of subplots with len(components_to_plot)//2 +len(components_to_plot)%2 subplots in each row
    fig, axs = plt.subplots(2,int(len(components_to_plot)//2 +len(components_to_plot)%2), figsize=(10,5), sharex = True)
    # flatten axs array
    axs = axs.flatten()
    for component_index, component_value in enumerate(components_to_plot):
        # increment ax based on component
        ax = axs[component_index]
        # set current ax
        plt.sca(ax)
        # column of mixing matrix matching component to plot is the component to plot
        im, cbar = plot_topo_maps.plot_topo(channel_names=channels, channel_data=mixing_matrix[:,component_value],
                  title=f"ICA component {component_value}", cbar_label="", montage_name="standard_1005")
        
    # get rid of unused axs
    for unused_ax in range(len(components_to_plot), len(axs)):
        fig.delaxes(axs[unused_ax])
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"ICA_first_{len(components_to_plot)}_components.png")
    
#%% Part 3: Transform into Source Space

def get_sources(eeg, unmixing_matrix, fs, sources_to_plot):
    """
    Transforms EEG data into source space using unmixing_matrix, plots the
    specified sources, and returns the source activation timecourses 
    (source_activations). If no sources are specified to plot (meaning [] is
    specified for sources_to_plot), no plotting occcurs and the source
    activation timecourses are returned.

    Parameters
    ----------
    eeg : Array of float. Size (C,S) where C is the number of channels for 
    which we have data and S is the number of samples.
        EEG data in uV.
    unmixing_matrix : Array of float. Size (Co,C) where Co is the number of 
    independent components and C is the number of channels.
        ICA source transformation. Rows are weights used to combine across
        electrodes to get activity for a single source. 
    fs : float.
        Sampling frequency in Hz.
    sources_to_plot : list
        The indices of the sources to plot.

    Returns
    -------
    source_activations : Array of float. Size (Co, C) where Co is the number
    of independent components and C is the number of channels.
        Source activation timecourses.

    """
    # transform EEG data into source space with unmixing_matrix
    # unmixing matrix * X = U
    # columns of unmixing matrix should match rows of eeg
    source_activations = np.matmul(unmixing_matrix, eeg) # unmixed source activity

    # if user enters sources_to_plot = [], don't plot anything, just return source activations
    if sources_to_plot == []:
        return source_activations
    # otherewise, plot the specified sources
    else:
        # check that source_activations contains sources_to_plot, raise error if not
        num_sources = source_activations.shape[1] # sources are columns in source_activations
        for source_to_plot in sources_to_plot:
            if source_to_plot >= num_sources:
                raise ValueError("Source index to plot {source_to_plot} is out of range. Only {num_sources} sources are available.")
            else:
                # if source_activations contains source_to_plot, plot it
                # set up time
                T = np.arange(source_activations.shape[1]) / fs # time axis (seconds)
                # initialize axs index
                axs_idx = 0
                # initialize figure with as many rows as sources_to_plot and 1 column
                fig, axs = plt.subplots(len(sources_to_plot),1, figsize=(10,5), sharex = True)
                fig.suptitle("AudVis EEG Data in ICA Source Space")
                
                for source_to_plot in sources_to_plot:
                    # use source_to_plot index to get source data- rows are sources
                    data_to_plot = source_activations[source_to_plot,:]
                    # plot
                    axs[axs_idx].plot(T, data_to_plot, label="reconstructed")
                    axs[axs_idx].set_xlabel("time (s)")
                    axs[axs_idx].set_ylabel(f"Source {source_to_plot} (uV)")
                    axs[axs_idx].legend(loc="upper right")
                    axs_idx +=1 # incrememnt index for axs
                
                plt.tight_layout()
                plt.savefig(f"AudVis_EEG_Data_ICA_Source_Space_Sources_{sources_to_plot}.png")
                # return activations after plotting
                return source_activations

#%% Part 4: Remove Artifact Components

def remove_sources(source_activations, mixing_matrix, sources_to_remove):
    """
    Removes sources if specified (intended to be artifact sources) and transforms
    source data back to electrode space. If sources to remove are not specified,
    data is just transformed back to the electrode space with no source removal.

    Parameters
    ----------
    source_activations : Array of float. Size (Co, C) where Co is the number
    of independent components and C is the number of channels.
        Source activation timecourses.
    mixing_matrix : Array of float. Size (C, C) where C is the number of channels
    in the EEG data.
        ICA components. Rows are weights used to combine across electrodes to 
        get activity for a single source. Each column is the impact of a source
        on all electrodes.
    sources_to_remove : List of size (So,) where So is the number of sources to 
    remove.
        The indices of the sources to remove from the data.

    Returns
    -------
    cleaned_eeg : Array of float. Size (C,S) where C is the number of channels
    and S is the number of samples.
        EEG data in electrode space. If sources to remove were specified, they 
        were removed before transforming data back to electrode space and returning
        this variable.

    """
    # make a copy of source_activations to mitigate unexpected behavior
    source_activations_copy = source_activations.copy()
    # save num sources for checking if source exists later
    num_sources = source_activations.shape[1] # sources are columns in source_activations

    if sources_to_remove == []:
        # if nothing to remove, just transform back
        cleaned_eeg = np.matmul(mixing_matrix, source_activations_copy)
        # return variable cleaned_eeg
        return cleaned_eeg
    else:
        # zero out specified sources if they are specified
        # rows of source_activations align with sources
        for source in sources_to_remove:
            # check that source_activations contains source and raise error otherwise
            for source_to_remove in sources_to_remove:
                if source_to_remove >= num_sources:
                    raise ValueError("Source index to remove {source_to_remove} is out of range. Only {num_sources} sources are available.")
            source_activations_copy[source,:] = 0
        # transform results back into electrode space
        cleaned_eeg = np.matmul(mixing_matrix, source_activations_copy)
        # return variable cleaned_eeg
        return cleaned_eeg
    
#%% Part 5: Transform Back into Electrode Space
def compare_reconstructions(eeg, reconstructed_eeg, cleaned_eeg, fs, channels,
                            channels_to_plot):
    """
    Plots raw, reconstructed, and cleaned eeg data in a series of subplots.

    Parameters
    ----------
    eeg : Array of float. Size (C,S) where C is the number of channels for 
    which we have data and S is the number of samples.
        EEG data in uV.
    reconstructed_eeg : Array of float. Size (C,S) where C is the number of channels
    and S is the number of samples.
        EEG data, transformed to source space and back to electrode space.
    cleaned_eeg : Array of float. Size (C,S) where C is the number of channels
    and S is the number of samples.
        EEG data, transformed to source space and back to electrode space with
        specified sources removed. 
    fs : Float.
        Sampling frequency in Hz.
    channels : Array of str. Size (C,) where C is the number of channels in the
    EEG data.
        The names of EEG channels.
    channels_to_plot: List of str
        The channels to plot raw EEG data for.

    Returns
    -------
    None.

    """
    # set up time
    T = np.arange(eeg.shape[1]) / fs # time axis (seconds)
    # initialize axs index
    axs_idx = 0
    # initialize figure with as many rows as channels_to_plot and 1 column
    fig, axs = plt.subplots(len(channels_to_plot),1, figsize=(10,5), sharex = True)
    fig.suptitle("AudVis EEG Data reconstructed & cleaned after ICA")
    
    for channel_to_plot in channels_to_plot:
        # get channel index from channel name
        channel_to_plot_index = None
        for channel_index, channel_value in enumerate(channels):
            if channel_value == channel_to_plot:
                channel_to_plot_index = channel_index
                break
        # use source_to_plot index to get source data- rows are sources
        original_eeg_for_channel = eeg[channel_to_plot_index,:]
        reconstructed_eeg_for_channel = reconstructed_eeg[channel_to_plot_index, :]
        cleaned_eeg_for_channel = cleaned_eeg[channel_to_plot_index, :] 
        # plot each
        axs[axs_idx].plot(T, original_eeg_for_channel, label="raw")
        axs[axs_idx].plot(T, reconstructed_eeg_for_channel, label="reconstructed",
                          linestyle = "dashed")
        axs[axs_idx].plot(T, cleaned_eeg_for_channel, label="cleaned",
                          linestyle = "dotted")
        axs[axs_idx].set_xlabel("time (s)")
        axs[axs_idx].set_ylabel(f"Voltage on {channel_to_plot} (uV)")
        axs[axs_idx].legend(loc="upper right")
        axs_idx +=1 # incrememnt index for axs
    
    plt.tight_layout()
    plt.savefig(f"AudVis_EEG_reconstructed_cleaned_ICA_channels_{channels_to_plot}.png")
    
    
    
    
    
    