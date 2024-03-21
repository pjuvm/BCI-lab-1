import numpy as np
from scipy.signal import * 
from filter_ssvep_data import * 

def load_ssvep_data(subject = 1, data_directory = '/SsvepData'):
  import os
  os.chdir(data_directory)
  data = np.load(f'SSVEP_S{subject}.npz', allow_pickle = True)
  return dict(data)

subject = 1 
data = load_ssvep_data(subject = subject, data_directory = ".")

fir12 = make_bandpass_filter(high = 13, low = 11, fs = 1000, filter_order = 1000, filter_type = "hann")
fir15 = make_bandpass_filter(high = 16, low = 14, fs = 1000, filter_order = 1000, filter_type = "hann")
filtered_data_12 = filter_data(data = dict(data), b = fir12)
filtered_data_15 = filter_data(data = dict(data), b = fir15)
envelope12 = get_envelope(data = data, filtered_data = filtered_data_12, channel_to_plot = "Oz", ssvep_frequency = 12)
envelope15 = get_envelope(data = data, filtered_data = filtered_data_15, channel_to_plot = "Oz", ssvep_frequency = 15)

plot_ssvep_amplitudes(data = data, envelope_a = envelope12, envelope_b = envelope15, channel_to_plot = "Oz")
