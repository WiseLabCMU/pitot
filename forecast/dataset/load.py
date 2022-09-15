"""Data loader."""


import os
import numpy as np
import pandas as pd


class Trace:
    """Load trace data.

    Parameters
    ----------
    path : str
        Path for this data trace; looks for npz files
        {path}_0.npz, {path}_1.npz, ...
    """

    COLUMNS = {
        "start_time": (0, "Period start timestamp (CLOCK_REALTIME)"),
        "wall_time": (1, "Wall clock time (CLOCK_MONOTONIC_RAW)"),
        "cpu_time": (2, "Thread CPU time (CLOCK_THREAD_CPUTIME_ID)"),
        "process_time": (3, "Runtime CPU time (CLOCK_PROCESS_CPUTIME_ID)"),
        "memory": (4, "Max memory usage"),
        "cpufreq": (5, "Average clock speed"),
        "ch_mqtt": (6, "Bytes read from MQTT"),
        "ch_local": (7, "Bytes read from loopback"),
        "ch_out": (8, "Bytes written to channels"),
        "utime": (9, "Userspace time"),
        "stime": (10, "Systemspace time"),
        "maxrss": (11, "Maximum resident set size (peak memory)"),
        "majflt": (12, "Major page faults"),
        "minflt": (13, "Minor page faults"),
        "nvcsv": (14, "Voluntary Context Swaps"),
        "nivcsw": (15, "InVoluntary Context Swaps"),
    }

    def __init__(self, path="data"):

        parent = os.path.dirname(path)
        self.srcs = [os.path.join(parent, d) for d in os.listdir(parent)]
        self.srcs = [s for s in self.srcs if s.startswith(path)]
        self.srcs.sort(key=os.path.getctime)
        self.data = np.vstack([np.load(s)['data'] for s in self.srcs])

    def arrays(self, keys=None):
        """Load as dict of arrays."""
        if keys is None:
            keys = list(self.COLUMNS.keys())

        return {
            k: self.data[:, self.COLUMNS[k][0]]
            for k in keys
        }

    def dataframe(self, keys=None):
        """Load as dataframe."""
        return pd.DataFrame(self.arrays(keys=keys))
