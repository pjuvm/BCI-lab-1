from scipy.signal import * 

def make_bandpass_filter(high, low, fs, filter_order, filter_type):
  fir = firwin(fs = fs, numtaps = filter_order + 1, cutoff = [low,high], window = filter_type, pass_zero = False)
  return fir


def filter_data(data, b):
  raw_data = data['eeg']
  return filtfilt(b = b, a = 1, x = raw_data)

def get_envelope(data, filtered_data, channel_to_plot, ssvep_frequency):
  channel_list = list(dict(data)['channels'])
  channel_idx = channel_list.index(channel_to_plot)
  envelope = np.abs(scipy.signal.hilbert(filtered_data))
  plt.figure(figsize = (20,10))
  plt.plot(envelope[channel_idx,:], alpha = 0.6, color = 'red')
  plt.plot(filtered_data[channel_idx,:], alpha = 0.2)
  
  plt.xlim(5000,8000)
  plt.show()
  return envelope

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, subject = 1):
  channel_list = list(dict(data)['channels'])
  channel_idx = channel_list.index(channel_to_plot)
  #extract data
  event_durations = data['event_durations']
  event_samples = data['event_samples']
  event_types = data['event_types']
  
  #compute event intervals, start and finish times
  event_intervals = []
  for i in range(len(event_samples)):
    event_start = event_samples[i]
    event_finish = int(event_samples[i] + event_durations[i])
    event_intervals.append((event_start, event_finish))


    
  fig, ax = plt.subplots(2)
  ax[0].set_title(f'SSVEP Subject {subject} SSVEP amplitudes')
  #subplot 1
  for event_num, interval in enumerate(event_intervals):

    if event_types[event_num] == "12hz":
      ax[0].hlines(xmin = interval[0], xmax = interval[1], y = 12,label = 'o')
      ax[0].plot([interval[0], interval[1]], [12,12], 'bo')

    else:
      ax[0].hlines(xmin = interval[0], xmax = interval[1], y = 15,label = 'o')
      ax[0].plot([interval[0], interval[1]], [15,15], 'bo')
  ax[0].set_xlabel('time(s)')
  ax[0].set_ylabel('Flash Frequency')
  ax[0].grid(alpha=0.2,color="black")
  #ax[0].vlines([x if x % 1000 == 0 else None for x in range(data['eeg'].shape[-1])], ymin = 12, ymax = 15)
  #subplot 2
  ax[1].plot(envelope_a[channel_idx,:])
  ax[1].plot(envelope_b[channel_idx,:])
  ax[1].set_xlabel('time(s)')
    #ax[1].set_title(f"Channel {channel} EEG plot")
  #the grid

  ax[1].grid(alpha=0.2,color="black")
  
  fig.tight_layout()
  
  
  #plt.xlim(5000,10000)
  plt.tight_layout()
  plt.show()
