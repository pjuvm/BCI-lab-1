def subject_epoch_groups(subject):
  eeg_time, eeg_data, rowcol_id, is_target = load_training_eeg(subject = subject)
  event_sample, is_target_event = get_events(rowcol_id, is_target)
  eeg_epochs, erp_times = epoch_data(eeg_time, eeg_data, event_sample, epoch_start_time = -0.5, epoch_end_time = 1)
  target_epochs = eeg_epochs[is_target_event]
  nontarget_epochs = eeg_epochs[~is_target_event]
  return target_epochs.astype(float), nontarget_epochs.astype(float)

def erp_difference(EEG1,EEG2):
  """ this functions finds the absolute value of the difference between the ERPs of two EEG groups
  Args:
    EEG1: 3d array of (trials, time points, channels)
    EEG2: same as above
  Returns:
    Absolute value of difference between ERP (mean EEG) of two groups
  """
  target_ERP = EEG1[:,:,:].mean(0)
  nontarget_ERP = EEG2[:,:,:].mean(0)
  return np.abs(target_ERP - nontarget_ERP)

def p_value_gen(EEG1, EEG2, n_iter = 500, channel = 0, alpha = 0.05, random_seed=888):
  """ Function that generates a array of p-values of size (time points, channels)
  relating real data to bootstrapped data; viz probabilities of real data given the bootstrapped sample data
  Args:
    EEG1 -- array of shape (samples, time, channels)
    EEG2 -- as above
    n_iter -- number of bootstrap iterations
    channel -- int. The channel to generate p-values for
  Returns:
    p-values: array that has bootstrapped p-values for each slice. These are not corrected for multiple trials.
    significance list: an array that indexes where significant p-values occur during the epoch.
  """
  np.random.seed(random_seed)
  combined_epochs = np.vstack([EEG1,EEG2])
  real_erp_diff = erp_difference(EEG1,EEG2)  #the real ERP difference between two groups or conditions
  score = np.zeros(EEG1.shape[1],)                      #zero matrix to count significance
  for _ in range(n_iter):
    inds = np.random.randint(0,len(combined_epochs),len(combined_epochs))
    bootstrapped_epochs = combined_epochs[inds]
    bootstrap_target = bootstrapped_epochs[:EEG1.shape[0]] #use same number of trials eg 150 as data
    bootstrap_nontarget = bootstrapped_epochs[EEG2.shape[0]:] #use same number of trials eg 750 as data
    #bootstrapped nontarget ERP, mean of the 750 samples
    bootstrap_erp_diff_array = erp_difference(bootstrap_target,bootstrap_nontarget) #absolute difference of the two as statistic
    #print(bootstrap_erp_diff_array.shape)

    x = (bootstrap_erp_diff_array[:,channel] > real_erp_diff[:,channel]).astype(int) #whenever a bootstrap value is larger than the real data add 1
    score += x

  return score/n_iter, np.where(score/n_iter < alpha) #calculate percentage of bootstrap samples bigger than the real data


def find_significant_times(subjects, n_channels = 8, alpha = 0.05, samples_per_epoch=384, n_iter=500, random_seed=888):
  """
  this function takes the data across the various subjects and channels, epochs the data by label (e.g. target vs nontarget),
  then generates false discovery rate corrected p-values for each time point (sample) in the epoch. This is then
  used to create a "tally" matrix that shows, given a channel, how subjects had statistically significant results 
  at that particular time.

  Args: 
    subjects: list of numbers corresponding to subject  e.g. [1,2,3]
    n_channels: int, number of channels in the data
    alpha: statistical significance threshold, default is 0.05
    samples_per_epoch: int, number of samples in an epoch
    n_iter: int, number of bootstrap samples
  Returns:
    significance_dict: a dictionary, where each key corresponds to a channel (e.g. "1") and the associated value is an array of 
    size (samples_per_epoch,) that gives counts of how many subjects had significant results at that time point in the epoch

  """
  significance_dict = {} 
  np.random.seed(random_seed)
  for chan in range(n_channels):
    score = np.zeros(samples_per_epoch,)
    for subj in subjects:
      #split data into two conditions or groups of EEG
      EEGa,EEGb = subject_epoch_groups(subj)  
      #get the uncorrected p-values
      my_raw_pvals = p_value_gen(EEG1 = EEGa, EEG2 = EEGb, n_iter = 500, channel = channel, alpha = 0.05)   
      #false discovery rate correction for the p-values, uses fdr_correction output of boolean values based on alpha threshold
      mask = fdr_correction(my_raw_pvals[0], alpha = alpha)[0]   
      #takes boolean mask converts to binary matrix of 1s and 0s
      mask = mask.astype(int)   
       #adds the above matrix to the score matrix, to tally significant time points across subjects  
      score += mask              
    #add the per channel significance matrix to a dictionary 
    significance_dict[str(channel)] = score
  return significance_dict

def rank_channel_significance(channels, fs, epoch_start, epoch_end, tstart, tend):
  """ this function ranks each channel by examining the corresponding 'significance matrix', a matrix 
  which shows how many subjects had significant results at a given time point in the epoch. The ranking is done 
  by computing an "integral", that is, treating the matrix's values as a step function. This "integral" rewards 
  greater numbers of subjects (height) as well as the number of time slices (width, temporal duration) to create a 
  rough way of ranking the channels. We can use tstart and tend to confine our "limits of integration" to a particular 
  time slice of interest, e.g. 0-500ms after stimulus onset, or 200-500ms after onset.

  Args:
    channels: int, number of channels 
    fs: float, sampling frequency
    epoch_start: float, first time point in epoch
    epoch_end: float, last time point in epoch
    tstart: float, the first time point in the window of interest
    tend: float, the last time point in the window of interest 
  Returns:
    rankarray_sorted_desc: array of shape (n_channels,1). The first axis (channels) is ordered using the values 
    in the second axis (integral), thus giving the ranking of most important channel to least important channel
    in the given time window. 

  """
  
  time_axis = np.linspace(epoch_start, epoch_end, int(fs * (epoch_end - epoch_start)))
  print(len(time_axis))
  time_mask = (time_axis >= tstart) & (time_axis <= tend)
  rankarray = np.empty((len(channels),2))
  dx = 1/fs
  for chan in range(len(channels)):
    
    rankarray[chan][0] = chan
    rankarray[chan][1] = np.sum(significance_dict[str(chan)][time_mask] * dx)

  
  rankarray_sorted_desc = rankarray[rankarray[:, 1].argsort()[::-1]]
  return rankarray_sorted_desc 

def erp_group_median(subjects):
  """
  function that takes EEG data from various subjects, finds the median per subject, and then finds the median of the medians
  for each respective EEG group (e.g. target vs nontarget)
  Args:
    subjects: list of subjects 
  Returns
    group_median_erp: array of length samples_per_epoch
  """

  target_median_ERPs = []
  nontarget_median_ERPs = []
  for subj in subjects:
    target_epochs, nontarget_epochs = subject_epoch_groups(subj)
    #print(target_epochs.shape)
    #find the median of the subject at hand
    target_median = np.median(target_epochs, 0)
    # do same for nontarget
    nontarget_median = np.median(nontarget_epochs, 0)
    #append subject's median to a list of such medians
    target_median_ERPs.append(target_median)
    #append subject nontarget median to list
    nontarget_median_ERPs.append(nontarget_median)
    #take list and created 3d array of shape (n_subjects, samples_per_epoch, channels)
  stacked_target_erps = np.stack(target_median_ERPs)
  stacked_nontarget_erps = np.stack(nontarget_median_ERPs)
    #find the median of those medians
  group_target_median_erp = np.median(stacked_target_erps, axis = 0)
  group_nontarget_median_erp = np.median(stacked_nontarget_erps, axis = 0)
  return group_target_median_erp, group_nontarget_median_erp

def window_waveform(signal, samples_per_epoch=384, start=-0.5, end=1.0, window_start = .25, window_end=0.5):
  
  """
  this function takes an epoch and slices it along the time axis given a start point and end point 
  Args:
    signal: array of size (n_samples, n_channels)
    samples_per_epoch: int number of samples in an epoch
    start: float, epoch begin time relative to stimulus onset 
    end: float, epoch end time relative to stimulus onset 
    window_start: float, time in (ms) relative to stimulus onset where window start 
    window_end: float, time in (ms) relative to stimulus onset where window ends 
  Return:
    windowed_signal: 1d array of length samples between window start and end 
  """

  time_axis = np.linspace(start, end, samples_per_epoch)
  window_mask = np.where((time_axis >= window_start) & (time_axis <= window_end))
  windowed_signal = signal[window_mask]
  windowed_signal = windowed_signal.mean(axis = 0)
  return windowed_signal
