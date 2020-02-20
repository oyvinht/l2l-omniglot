import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict

def spiking_per_class(indices, spikes, start_t, end_t, dt):
    uindices = np.unique(indices)
    aggregate_per_class = {}
    individual_per_class = {}
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        class_idx = int(st // dt)
        cls = int(indices[class_idx])
        apc = aggregate_per_class.get(cls, {})
        ipc = individual_per_class.get(cls, {})
        ind = {}
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            if len(whr):
                narray = apc.get(nid, None)
                if narray is None:
                    narray = times[whr]
                else:
                    narray = np.append(narray, times[whr])

                ind[nid] = times[whr]
                apc[nid] = narray

        aggregate_per_class[cls] = apc

        ipc[class_idx] = ind
        individual_per_class[cls] = ipc

    return aggregate_per_class, individual_per_class


def spiking_per_class_split(indices, spikes, start_t, end_t, dt):
    uindices = np.unique(indices)
    aggregate_per_class = {u: {} for u in uindices}
    individual_per_class = {u: {} for u in uindices}
    for st in np.arange(start_t, end_t, dt):
        sample_idx = int(st // dt)
        cls = int(indices[sample_idx])
        ind = {}
        for nid, ts in enumerate(spikes[sample_idx]):
            times = np.asarray(ts)
            if len(times):
                narray = aggregate_per_class[cls].get(nid, None)
                if narray is None:
                    narray = times
                else:
                    narray = np.append(narray, times)

                ind[nid] = times

                aggregate_per_class[cls][nid] = narray

        individual_per_class[cls][sample_idx] = ind

    return aggregate_per_class, individual_per_class


def split_per_dt(spikes, start_t, end_t, dt):
    spikes_dt = []
    for st in np.arange(start_t, end_t, dt):
        et = st + dt
        dt_act = []
        for nid, ts in enumerate(spikes):
            times = np.array(ts)
            whr = np.where(np.logical_and(st <= times, times < et))[0]
            new_ts = times[whr] if len(whr) else []
            dt_act.append(new_ts)

        spikes_dt.append(dt_act)
    return spikes_dt


def count_active_per_dt(split_spikes):
    count = []
    for spikes in split_spikes:
        count.append(np.sum([len(ts) for ts in spikes]))
    return count

def count_non_spiking_samples(spikes, start_t, end_t, dt):
    not_spiking = 0
    for st in np.arange(start_t, end_t, dt):
        sample_idx = int(st // dt)
        sample_not_spiking = np.sum([len(ts) > 0 for ts in spikes[sample_idx]])
        not_spiking += int(sample_not_spiking == 0)
    return not_spiking

