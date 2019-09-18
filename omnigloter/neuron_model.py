from copy import deepcopy
from functools import partial
import numpy as np
import lazyarray as la
from pyNN.standardmodels import cells, build_translations
from pynn_genn.standardmodels.cells import tau_to_decay, genn_postsyn_defs
from pynn_genn.simulator import state
import logging
from pynn_genn.model import GeNNStandardCellType, GeNNDefinitions
from pygenn.genn_model import create_custom_neuron_class

def inv_val(val_name, **kwargs):
    return 1.0/kwargs[val_name]

def inv_tau_to_decay(val_name, **kwargs):
    return 1.0/la.exp(-state.dt / kwargs[val_name])



genn_neuron_defs = GeNNDefinitions(
    definitions = {
        "sim_code" : """
            $(I) = $(Isyn);
            if ($(RefracTime) <= 0.0) {
                scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
                $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
                $(VThreshAdapt) = fmax($(VThreshAdapt) * $(DownThresh), $(Vthresh));
            }
            else {
                $(RefracTime) -= DT;
            }
        """,

        "threshold_condition_code" : "$(RefracTime) <= 0.0 && $(V) >= $(VThreshAdapt)",

        "reset_code" : """
            $(V) = $(Vreset);
            $(RefracTime) = $(TauRefrac);
            $(VThreshAdapt) *= $(UpThresh); 
        """,

        "var_name_types" : [
            ("V", "scalar"),
            ("I", "scalar"),
            ("RefracTime", "scalar"),
            ("VThreshAdapt", "scalar"),
        ],

        "param_name_types": {
            "Rmembrane":  "scalar",  # Membrane resistance
            "ExpTC":      "scalar",  # Membrane time constant [ms]
            "Vrest":      "scalar",  # Resting membrane potential [mV]
            "Vreset":     "scalar",  # Reset voltage [mV]
            "Vthresh":    "scalar",  # Spiking threshold [mV]
            "Ioffset":    "scalar",  # Offset current
            "TauRefrac":  "scalar",
            "UpThresh":   "scalar",
            "DownThresh": "scalar",
        }
    },
    translations = (
        ("v_rest",      "Vrest"),
        ("v_reset",     "Vreset"),
        ("cm",          "Rmembrane",     "tau_m / cm", ""),
        ("tau_m",       "ExpTC",         partial(tau_to_decay, "tau_m"), None),
        ("tau_refrac",  "TauRefrac"),
        ("v_thresh",    "Vthresh"),
        ("i_offset",    "Ioffset"),
        ("v",           "V"),
        ("i",           "I"),
        ("mult_thresh", "UpThresh", partial(inv_val, "mult_thresh"), None),
        ("tau_thresh",  "DownThresh",    partial(inv_tau_to_decay, "tau_thresh"), None),
        ("v_thresh_adapt",    "VThreshAdapt"),
    ),
    extra_param_values = {
        "RefracTime" : 0.0,
    })


class IF_curr_exp_i(cells.IF_curr_exp, GeNNStandardCellType):
    __doc__ = cells.IF_curr_exp.__doc__

    default_parameters = {
        'v_rest': -65.0,  # Resting membrane potential in mV.
        'cm': 1.0,  # Capacity of the membrane in nF
        'tau_m': 20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E': 5.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I': 5.0,  # Decay time of inhibitory synaptic current in ms.
        'i_offset': 0.0,  # Offset current in nA
        'v_reset': -65.0,  # Reset potential after a spike in mV.
        'v_thresh': -50.0,  # Spike threshold in mV. STATIC, MIN
        'i': 0.0, #nA total input current

        ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
        # 'tau_thresh': 80.0,
        # 'mult_thresh': 1.8,
        ### https://www.frontiersin.org/articles/10.3389/fncom.2018.00074/full
        'tau_thresh': 120.0,
        'mult_thresh': 1.8,
        'v_thresh_adapt': -50.0,  # Spike threshold in mV.

    }

    recordable = ['spikes', 'v', 'i', 'v_thresh_adapt']

    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'isyn_exc': 0.0,
        'isyn_inh': 0.0,
        'i': 0.0,
    }

    units = {
        'v': 'mV',
        'isyn_exc': 'nA',
        'isyn_inh': 'nA',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_thresh': 'mV',
        'i': 'nA',
        'tau_thresh': 'ms',
        'mult_thresh': '',
        'v_thresh_adapt': 'mV',
    }

    receptor_types = (
        'excitatory', 'inhibitory',
    )

    genn_neuron_name = "IF_i"
    genn_postsyn_name = "ExpCurr"
    neuron_defs = genn_neuron_defs
    postsyn_defs = genn_postsyn_defs[genn_postsyn_name]
