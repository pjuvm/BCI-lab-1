#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:01:10 2024

@author: Alaina Birney

A module closely related to remove_audvis_blinks.py. Functions within 
remove_audvis_blinks.py are called to load data and plot raw EEG data for 
specified channels, plot scalp maps/ topo maps of the first 10 ICA components 
(the components that capture the most variance in the data), transform data 
into source space and plot EOG components (EOG components were manually identified 
from the topomaps of ICA components), remove artifact components, and plot raw 
eeg data, reconstructed eeg data, and cleaned eeg data (with artifact components 
removed) in sublots corresponding to different channels.
"""

import remove_audvis_blinks as remove_audvis_blinks
import numpy as np
#%% Part 1: Load Data
channels_to_plot = ["Fpz", "Cz", "Iz"]
data = remove_audvis_blinks.load_data(channels_to_plot = channels_to_plot)
"""
Eye blink artifacts typically look like huge peaks on the frontal electrodes
(100uV) (source: Dr.J Lecture 8: Noise and Confidence), these peaks are also 
reflected on other electrodes, but they are a bit less in magnitude. Based
on this, we can infer that a blink likely occured at about 14.5 seconds because
we see EEG data spike to approximately 229 uV on electrode Fpz (Fpz is 
positioned over the frontal lobe) while we also see spikes, albeit less in 
magnitude on electrodes Iz and Cz at that time (Iz is positioned over the
occipital lobe while Cz is positioned over the central midline, between the
parietal and frontal lobes).
"""

#%% Part 2: Plot the Components

# call plot_components to plot first 10 components
components_to_plot = np.arange(0,10,1)
mixing_matrix = data["mixing_matrix"]
channels = data["channels"]
remove_audvis_blinks.plot_components(mixing_matrix, channels, components_to_plot)
"""
The term EOG artifacts encompasses eye blink and eye movement artifacts. As stated
before, eye blink artifacts typically look like huge peaks on the frontal electrodes.
Eye movement artifacts typically look like increased or decreased voltage on
lateral frontal electrodes (source: Dr. J lecture 8: Noise and Confidence and
lecture 22: Combining Across Sensors). Additionally, ICA polarity does not matter
(source: lab 5 instructions). Therefore, scalp maps should show cooler or 
warmer colors over the central or lateral frontal lobe when EOG artifacts are 
present. Based on this information and the topomaps of ICA components, we can 
infer that EOG artifacts are present on ICA components 0 and 9. This is because
ICA component 0 shows cooler colors over the central frontal lobe while 
ICA component 9 shows a warmer color over the left lateral frontal lobe and a 
cooler color over the right lateral frontal lobe (the warmer and cooler colors
on the different sides of the frontal lobe correspond to the direction of 
eye movement).

"""

#%% Part 3: Transform Into Source Space
# get parameters
eeg = data["eeg"]
unmixing_matrix = data["unmixing_matrix"]
fs = data["fs"]
# plot the sources with EOG components and a third that doesn't look like EOG
sources_to_plot = [0,9,5]
source_activations = remove_audvis_blinks.get_sources(eeg, unmixing_matrix, fs, sources_to_plot)

#%% Part 4: Remove Artifact Components
sources_to_remove = [0,9]
#sources_to_remove = [5,7]
#sources_to_remove = [2,8]
# call once to remove specified sources
cleaned_eeg = remove_audvis_blinks.remove_sources(source_activations, mixing_matrix, sources_to_remove)
# call again to get clean eeg
sources_to_remove = []
reconstructed_eeg = remove_audvis_blinks.remove_sources(source_activations, mixing_matrix, sources_to_remove)

#%% Part 5: Transform Back into Electrode Space

remove_audvis_blinks.compare_reconstructions(eeg, reconstructed_eeg,
                                             cleaned_eeg, fs, channels,
                                             channels_to_plot)
"""
The plot resulting from this call to the compare_reconstructions function shows
that our ICA artifact removal was effective; the reconstructed data mirrors the 
raw eeg data, while the cleaned data closely mirrors the raw eeg data, but does
not contain the large peaks that we attributed to eye blink artifacts. For example,
if one looks closely at the data around the 14.5 second mark where we identified 
eye blink artifacts, it is clear that the cleaned data does not exhibit the same
peaks, instead it is more smooth and aligned with what we would expect to see
if these artifacts were not present.

If we did not know anything about the origin of a source ahead of time, we could
still infer that electrode Fpz would show the effects of removing components
associated with EOG artifacts because it is positioned over the frontal lobe and 
as stated prior, eye blink artifacts typically look like huge peaks on the frontal 
electrodes while eye movement artifacts typically look like increased or decreased 
voltage on lateral frontal electrodes (eye blink and eye movement artifacts are
each types of EOG artifacts). As electodes are positioned further from the 
frontal lobe, the effect of removing these components should be less clear. 
This means the difference should be less clear on electrodes Cz and Iz than Fpz, 
as Fpz is the closest electrode to the frontal lobe out of the three.

Further clarity is provided by testing removing different components. For example,
sources 5 and 7 were removed instead of sources 0 and 9 (sources 0 and 9 were
determined to correspond to contain EOG artifacts). When this was done, the same
reduction of peaks attributed to eye blink artifacts was not exhibited. The same
methodology was preformed for sources 2 and 8 and the same results (a lack 
of reduction in peaks attributed to eye blink artifacts) were observed.
"""


