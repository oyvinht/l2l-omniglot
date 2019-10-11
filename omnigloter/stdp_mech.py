from copy import deepcopy
from pyNN.standardmodels import synapses, StandardModelType, build_translations
from pynn_genn.simulator import state
import logging
from pygenn.genn_wrapper.WeightUpdateModels import StaticPulse
from pynn_genn.model import GeNNStandardSynapseType, GeNNDefinitions
from pynn_genn.standardmodels.synapses import WeightDependence, delayMsToSteps, delayStepsToMs, DDTemplate

class MySTDPMechanism(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    mutable_vars = set(["g"])

    base_translations = build_translations(
        ("weight", "g"),
        ("delay", "delaySteps", delayMsToSteps, delayStepsToMs),
        ("dendritic_delay_fraction", "_dendritic_delay_fraction"))

    base_defs = {
        "vars" : {"g": "scalar"},
        "pre_var_name_types": [],
        "post_var_name_types": [],

        "sim_code" : DDTemplate("""
            $(addToInSyn, $(g));
            scalar dt = $(t) - $(sT_post);
            $${TD_CODE}
        """),
        "learn_post_code" : DDTemplate("""
            scalar dt = $(t) - $(sT_pre);
            $${TD_CODE}
        """),

        "is_pre_spike_time_required" : True,
        "is_post_spike_time_required" : True,
    }

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

    def __init__(self, timing_dependence=None, weight_dependence=None,
            voltage_dependence=None, dendritic_delay_fraction=1.0,
            weight=0.0, delay=None):
        super(MySTDPMechanism, self).__init__(
            timing_dependence, weight_dependence, voltage_dependence,
            dendritic_delay_fraction, weight, delay)

        # Create a copy of the standard STDP defs
        self.wum_defs = deepcopy(self.base_defs)

        # Adds variables from timing and weight dependence to definitions
        self.wum_defs["vars"].update(self.timing_dependence.vars)
        self.wum_defs["vars"].update(self.weight_dependence.vars)

        # Add pre and postsynaptic variables from timing dependence to definition
        if hasattr(self.timing_dependence, "pre_var_name_types"):
            self.wum_defs["pre_var_name_types"].extend(
                self.timing_dependence.pre_var_name_types)

        if hasattr(self.timing_dependence, "post_var_name_types"):
            self.wum_defs["post_var_name_types"].extend(
                self.timing_dependence.post_var_name_types)

        # Apply substitutions to sim code
        td_sim_code = self.timing_dependence.sim_code.substitute(
            WD_CODE=self.weight_dependence.depression_update_code)
        self.wum_defs["sim_code"] =\
            self.wum_defs["sim_code"].substitute(TD_CODE=td_sim_code)

        # Apply substitutions to post learn code
        td_post_learn_code = self.timing_dependence.learn_post_code.substitute(
            WD_CODE=self.weight_dependence.potentiation_update_code)
        self.wum_defs["learn_post_code"] =\
            self.wum_defs["learn_post_code"].substitute(TD_CODE=td_post_learn_code)

        # Use pre and postsynaptic spike code from timing dependence
        if hasattr(self.timing_dependence, "pre_spike_code"):
            self.wum_defs["pre_spike_code"] = self.timing_dependence.pre_spike_code

        if hasattr(self.timing_dependence, "post_spike_code"):
            self.wum_defs["post_spike_code"] = self.timing_dependence.post_spike_code


class MyWeightDependence(synapses.MultiplicativeWeightDependence, WeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    depression_update_code = "$(g) += ($(g) - $(Wmin)) * update;\n"

    potentiation_update_code = "$(g) += ($(Wmax) - $(g)) * update;\n"

    translations = build_translations(*WeightDependence.wd_translations)

# class MyWeightDependence(synapses.AdditiveWeightDependence, WeightDependence):
#     __doc__ = synapses.AdditiveWeightDependence.__doc__
#
#     depression_update_code = "$(g) = fmin($(Wmax), fmax($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"
#
#     potentiation_update_code = "$(g) = fmin($(Wmax), fmax($(Wmin), $(g) + (($(Wmax) - $(Wmin)) * update)));\n"
#
#     translations = build_translations(*WeightDependence.wd_translations)

class MyTemporalDependence(synapses.STDPTimingDependence):
    # __doc__ = synapses.SpikePairRule.__doc__
    ### will try to do a displaced gaussian
    """mean: scalar: where should the Gaussian be centered (td == mean)
       std: scalar: how wide is the Gaussian
       displace: bias towards depression
       maxDt: scalar: temporal range, how far in future/past to see spikes
       Aminus: scalar: if total input is inhibitory, how much should we decrease weights
       Aplus: scalar: rate, how high the Gaussian is
    """
    vars = {
        # "tauPlus": "scalar",  # 0 - Potentiation time constant (ms)
        # "tauMinus": "scalar", # 1 - Depression time constant (ms)
        # "Aplus": "scalar",    # 2 - Rate of potentiation
        # "Aminus": "scalar",   # 3 - Rate of depression
        "Aplus": "scalar",
        "Aminus": "scalar",
        "mean": "scalar",
        "std": "scalar",
        "displace": "scalar",
        "maxDt": "scalar",
        # "": "scalar",
        # "": "scalar",
    }

    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # using {.brc} for left{ or right} so that .format() does not freak out
    sim_code = DDTemplate("""
        // std::cout << "pre(" << dt << ")" << std::endl;
        if (dt > 0 && dt <= $(maxDt)){
            scalar update = 0.0;
            if (dt <= $(std)){
                update = $(Aplus) - ( $(displace) * ($(I_post) < 0.0f) );
                
            }
            else{
                update = -$(Aminus);
            }
            
            $${WD_CODE}
        }
        """)

    learn_post_code = DDTemplate("""
        // std::cout << "post(" << dt << ")" << std::endl;
        if (dt > 0 && dt <= $(maxDt)){
            scalar update = 0.0;
            if (dt <= $(std)){
                update = $(Aplus) - ( $(displace) * ($(I_post) < 0.0f) );
                
            }
            else{
                update = -$(Aminus);
            }

            $${WD_CODE}
        }
        """)

    # sim_code = DDTemplate("""
    #     // std::cout << "pre(" << dt << ")" << std::endl;
    #     if (dt > 0 && dt <= $(maxDt)){
    #         const scalar e = (dt - $(mean))/$(std);
    #         const scalar div = 1.0f/(6.283185307179586f * $(std));
    #         const scalar update = $(Aplus) * (div * exp(-(e*e)) - $(displace)) - $(Aminus) * ($(I_post) < 0.0f);
    #         // std::cout << "pre(" << dt << ") e = " << e << std::endl;
    #         // std::cout << "pre(" << dt << ") update = " << update << std::endl;
    #         $${WD_CODE}
    #     }
    #     """)
    #
    # learn_post_code = DDTemplate("""
    #     // std::cout << "post(" << dt << ")" << std::endl;
    #     if (dt > 0 && dt <= $(maxDt)){
    #         const scalar e = (dt - $(mean))/$(std);
    #         const scalar div = 1.0f/(6.283185307179586f * $(std));
    #         const scalar update = $(Aplus) * (div * exp(-(e*e)) - $(displace));
    #         // std::cout << "post(" << dt << ") e = " << e << std::endl;
    #         // std::cout << "post(" << dt << ") update = " << update << std::endl;
    #         $${WD_CODE}
    #     }
    #     """)
    #
    pre_spike_code = """\
        const scalar dt = $(t) - $(sT_pre);
        $(preTrace) = $(preTrace) * exp(-dt) + 1.0;
        """

    post_spike_code = """\
        const scalar dt = $(t) - $(sT_post);
        $(postTrace) = $(postTrace) * exp(-dt) + 1.0;
        """
    # pre_spike_code = """\
    #     const scalar dt = $(t) - $(sT_pre);
    #     $(preTrace) = $(preTrace) * exp(-dt / $(tauPlus)) + 1.0;
    #     """
    #
    # post_spike_code = """\
    #     const scalar dt = $(t) - $(sT_post);
    #     $(postTrace) = $(postTrace) * exp(-dt / $(tauMinus)) + 1.0;
    #     """

    translations = build_translations(
        # ("tau_plus",   "tauPlus"),
        # ("tau_minus",  "tauMinus"),
        ("A_plus",     "Aplus"),
        ("A_minus",    "Aminus"),
        ("mean",       "mean"),
        ("std",        "std"),
        ("displace",   "displace"),
        ("maxDt",      "maxDt"),
    )

    default_parameters = {
        'mean':      0.0,
        'std':       1.0,
        'displace':  0.0,
        'maxDt':     20.0,
        'A_plus':    0.01,
        'A_minus':   0.0,
    }

    def __init__(self, A_plus=0.01, A_minus=0.01, mean=0.0, std=1.0, displace=0.0, maxDt=20.0):
        """
        Create a new specification for the timing-dependence of an STDP rule.
        """
        parameters = dict(locals())
        parameters.pop('self')
        synapses.STDPTimingDependence.__init__(self, **parameters)



class DVDTPlasticity(synapses.STDPMechanism, GeNNStandardSynapseType):
    __doc__ = synapses.STDPMechanism.__doc__

    mutable_vars = set(["g"])

    base_translations = build_translations(
        ("weight", "g"),
        ("delay", "delaySteps", delayMsToSteps, delayStepsToMs),
        ("dendritic_delay_fraction", "_dendritic_delay_fraction"),
    )

    base_defs = {
        "vars" : {"g": "scalar",},
        "pre_var_name_types": [],
        "post_var_name_types": [],

        "sim_code" : DDTemplate("""
            $(addToInSyn, $(g));
            $${TD_CODE}
        """),
        "learn_post_code" : DDTemplate("""
            $${TD_CODE}
        """),

        "is_pre_spike_time_required" : True,
        "is_post_spike_time_required" : False,
    }

    def _get_minimum_delay(self):
        if state._min_delay == "auto":
            return state.dt
        else:
            return state._min_delay

    def __init__(self, timing_dependence=None, weight_dependence=None,
            voltage_dependence=None, dendritic_delay_fraction=1.0,
            weight=0.0, delay=None):

        super(DVDTPlasticity, self).__init__(
            timing_dependence, weight_dependence, voltage_dependence,
            dendritic_delay_fraction, weight, delay)

        # Create a copy of the standard STDP defs
        self.wum_defs = deepcopy(self.base_defs)

        # Adds variables from timing and weight dependence to definitions
        self.wum_defs["vars"].update(self.timing_dependence.vars)
        self.wum_defs["vars"].update(self.weight_dependence.vars)

        # Add pre and postsynaptic variables from timing dependence to definition
        if hasattr(self.timing_dependence, "pre_var_name_types"):
            self.wum_defs["pre_var_name_types"].extend(
                self.timing_dependence.pre_var_name_types)

        if hasattr(self.timing_dependence, "post_var_name_types"):
            self.wum_defs["post_var_name_types"].extend(
                self.timing_dependence.post_var_name_types)

        # Apply substitutions to sim code
        td_sim_code = self.timing_dependence.sim_code.substitute(
            WD_CODE=self.weight_dependence.depression_update_code)
        self.wum_defs["sim_code"] =\
            self.wum_defs["sim_code"].substitute(TD_CODE=td_sim_code)

        # Apply substitutions to post learn code
        td_post_learn_code = self.timing_dependence.learn_post_code.substitute(
            WD_CODE=self.weight_dependence.potentiation_update_code)
        self.wum_defs["learn_post_code"] =\
            self.wum_defs["learn_post_code"].substitute(TD_CODE=td_post_learn_code)

        # Use pre and postsynaptic spike code from timing dependence
        if hasattr(self.timing_dependence, "pre_spike_code"):
            self.wum_defs["pre_spike_code"] = self.timing_dependence.pre_spike_code

        if hasattr(self.timing_dependence, "post_spike_code"):
            self.wum_defs["post_spike_code"] = self.timing_dependence.post_spike_code


class DVDTRuleTime(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    vars = {"tauPlus": "scalar",  # 0 - Potentiation time constant (ms)
            "tauMinus": "scalar", # 1 - Depression time constant (ms)
            "Aplus": "scalar",    # 2 - Rate of potentiation
            "Aminus": "scalar"}   # 3 - Rate of depression

    pre_var_name_types = [("preTrace", "scalar")]
    post_var_name_types = [("postTrace", "scalar")]

    # using {.brc} for left{ or right} so that .format() does not freak out
    # const scalar update = $(Aplus) * $(sT_pre) * $(dvdt);
    # this goes to depression update!
    # const scalar update = -($(Aplus) * $(dvdt_post) );
    # cout << "t = " << t << " , sT_pre = " << $(sT_pre) << endl;
    # cout <<  $(Aplus) << " * " << $(dvdt_post) << " = " << update << endl;
    #
    # const scalar old_g = $(g);
    # $${WD_CODE}
    # cout << "dg = " << $(g) << " - " << old_g << " = " << $(g) - old_g << endl;
    # cout << endl;

    sim_code = DDTemplate("""
        const scalar update = -( $(Aplus) * $(dvdt_post) );
        $${WD_CODE}
        """)

    learn_post_code = DDTemplate("""
        """)

    pre_spike_code = """\
        """

    post_spike_code = """\
        """

    translations = build_translations(
        ("tau_plus",   "tauPlus"),
        ("tau_minus",  "tauMinus"),
        ("A_plus",     "Aplus"),
        ("A_minus", "Aminus"))