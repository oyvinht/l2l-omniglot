import numpy as np
import matplotlib.pyplot as plt
import operator
from collections import OrderedDict

ZERO_FLOAT = 1.0e-9

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

def overlap_score(apc, n_output):
    classes = sorted(apc.keys())
    uniques = set()
    for cls in classes:
        uniques |= set(apc[cls].keys())

    neuron_overlaps = np.zeros(n_output)
    for cls0_id, cls0 in enumerate(classes[:-1]):
        for nid in np.unique(list(apc[cls0].keys())):
            for cls1 in classes[cls0_id + 1:]:
                nids1 = list(apc[cls1].keys())
                if nid in nids1:
                    neuron_overlaps[nid] += 1

    # print("{} / {} = {}".format(np.sum(neuron_overlaps), len(uniques), np.sum(neuron_overlaps) / len(uniques)))

    return np.sum(neuron_overlaps) / len(uniques)

def individual_score(ipc, n_tests, n_classes):
    events = np.zeros(n_classes)
    for cls in sorted(ipc.keys()):
        for idx in sorted(ipc[cls].keys()):
            if len(ipc[cls][idx]):
                events[cls] += 1
    max_score = n_tests * n_classes
    return np.sum(events) / max_score


def diff_class_vectors(apc, n_output):
    dcv = [np.zeros(n_output) for _ in apc]
    for c in apc:
        nids = list(apc[c].keys())
        if len(nids):
            dcv[c][nids] += 1

    return dcv

def diff_class_dists(diff_class_vectors):
    d = []
    for ix, x in enumerate(diff_class_vectors):
        for iy, y in enumerate(diff_class_vectors):
            if iy > ix:
                xn = np.sqrt(np.sum(x ** 2))
                xx = x / xn if xn > ZERO_FLOAT else x

                yn = np.sqrt(np.sum(y ** 2))
                yy = y / yn if yn > ZERO_FLOAT else y

                # sqrt(2) == max distance
                dot = np.sqrt(np.sum((xx - yy) ** 2)) / np.sqrt(2)
                d.append(dot)

    return np.asarray(d)

def any_all_zero(apc, ipc):
    any_zero = False
    all_zero = False
    for cls in sorted(ipc.keys()):
        for idx in sorted(ipc[cls].keys()):
            if len(ipc[cls][idx]) == 0:
                any_zero = True
                break
        if any_zero:
            break

    for c in apc:
        nids = list(apc[c].keys())
        if len(nids) == 0:
            all_zero = True

    return any_zero, all_zero

def same_class_vectors(ipc, n_out):
    smc = {c: [np.zeros(n_out) for _ in ipc[c]] for c in ipc}
    for c in sorted(ipc.keys()):
        for i, x in enumerate(sorted(ipc[c].keys())):
            for nid in ipc[c][x]:
                smc[c][i][nid] = 1
    return smc

def same_class_distances(same_class_vectors):
    same_class_norms = {
        c: np.linalg.norm(np.asarray(same_class_vectors[c]), axis=1)
        for c in same_class_vectors
    }

    same_class_distances = {}
    same_class_count = 0.0
    for c in same_class_vectors:
        d = []
        for ix, x in enumerate(same_class_vectors[c]):
            for iy, y in enumerate(same_class_vectors[c]):
                if iy > ix:
                    xn = same_class_norms[c][ix]
                    yn = same_class_norms[c][iy]

                    xnorm = x / xn if xn > ZERO_FLOAT else x
                    ynorm = y / yn if yn > ZERO_FLOAT else y

                    #sqrt(2) == max distance
                    dist = np.sqrt(np.sum((xnorm - ynorm) ** 2)) / np.sqrt(2)
                    d.append(dist)
                    same_class_count += 1.0

        same_class_distances[c] = np.asarray(d)

    return same_class_distances, same_class_count

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

def get_test_region(spikes, start_time, labels, t_per_sample):
    spks = []
    lbls = []
    for times in spikes:
        ts = np.asarray(times)
        whr = np.where(ts >= start_time)
        if len(whr[0]):
            spks.append(ts[whr].tolist())
        else:
            spks.append([])

    start_idx = int(start_time // t_per_sample)
    for l in labels[start_idx:]:
        lbls.append(l)

    return spks, lbls
