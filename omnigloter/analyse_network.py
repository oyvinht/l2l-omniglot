import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict

def spiking_per_class(indices, spikes, start_t, end_t, dt):
    uindices = np.unique(indices)
    aggregate_per_class = {u: {} for u in uindices}
    individual_per_class = {u: {} for u in uindices}
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        class_idx = int(st // dt)
        cls = int(indices[class_idx])
        ind = {}
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            if len(whr):
                narray = aggregate_per_class[cls].get(nid, None)
                if narray is None:
                    narray = times[whr]
                else:
                    narray = np.append(narray, times[whr])

                ind[nid] = times[whr]

                aggregate_per_class[cls][nid] = narray

        individual_per_class[cls][class_idx] = ind

    return aggregate_per_class, individual_per_class
