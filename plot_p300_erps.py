import numpy as np
from matplotlib import pyplot as plt

def get_events(rowcol_id, is_target):
    event_sample = np.array(np.where(np.diff(rowcol_id) > 0)) + 1
    is_target_event = np.array(np.where(is_target[event_sample] == True))
    return event_sample, is_target_event

def epoch_data (eeg_time, eeg_data, event_sample, epoch_start_time = -0.5, epoch_end_time=1):
    eeg_epochs
    pass