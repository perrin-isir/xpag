# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import time

global_timing_time_saved = None
global_timing_first_time_saved = None


def timing():
    """
    Returns:
        interval_time (in ms): the time elapsed since the previous call to timing()
        total_time (in s): the time elapsed since the first call to timing()
    """
    global global_timing_time_saved, global_timing_first_time_saved
    if global_timing_time_saved is None:
        global_timing_first_time_saved = time.time()
        global_timing_time_saved = global_timing_first_time_saved
        interval_time = 0.0
        total_time = 0.0
    else:
        interval_time = (time.time() - global_timing_time_saved) * 1000.0
        total_time = time.time() - global_timing_first_time_saved
        global_timing_time_saved = time.time()
    return interval_time, total_time
