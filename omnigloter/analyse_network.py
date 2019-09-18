import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict

def spiking_per_class(indices, spikes, start_t, end_t, dt):
    neurons_per_class = {u: {} for u in np.unique(indices)}
    et = 0
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        class_idx = int((st-start_t)//dt)
        cls = int(indices[class_idx])
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            if len(whr):
                narray = neurons_per_class[cls].get(nid, None)
                if narray is None:
                    narray = times[whr]
                else:
                    narray = np.append(narray, times[whr])

                neurons_per_class[cls][nid] = narray

    return neurons_per_class


