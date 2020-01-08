import pyNN.spiNNaker as sim

from python_models8.neuron.plasticity.stdp.timing_dependence.timing_dependence_step import \
    TimingDependenceStep as MyTemporalDependence

MyWeightDependence = sim.MultiplicativeWeightDependence
MySTDPMechanism = sim.STDPMechanism

# from stdp_mech_genn import MySTDPMechanism, MyWeightDependence, MyTemporalDependence