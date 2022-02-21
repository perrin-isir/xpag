# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

import time


def timing():
    global timing_time_saved, timing_first_time_saved
    try:
        timing_time_saved
    except NameError:
        timing_time_saved = None
    if timing_time_saved is None:
        timing_first_time_saved = time.time()
        timing_time_saved = timing_first_time_saved
        interval_time = 0.
        total_time = 0.
    else:
        interval_time = (time.time() - timing_time_saved) * 1000.
        total_time = time.time() - timing_first_time_saved
        timing_time_saved = time.time()
    return interval_time, total_time
