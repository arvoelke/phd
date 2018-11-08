from phd import *

from collections import defaultdict

from nengo.neurons import LIFRate


# https://github.com/nengo/nengo/issues/1487
class PoissonLIF(LIFRate):
    """Poisson-spiking leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    """

    probeable = ('spikes',)

    def __init__(self, tau_rc=0.02, tau_ref=0.002, amplitude=1, seed=None):
        super(PoissonLIF, self).__init__(
            tau_rc=tau_rc, tau_ref=tau_ref, amplitude=amplitude)

        # TODO(arvoelke): the simulator should pass in the rng
        self.rng = np.random.RandomState(seed=seed)

    def _sample_exponential(self, rates):
        # generate an exponential random variable (time between
        # poisson events) using its inverse CDF. note that this
        # distribution is "memoryless", which is why we don't
        # need to save this state or worry about what happens
        # outside of this time-step.
        return -np.log1p(-self.rng.rand(len(rates))) / rates

    def _poisson_step_math(self, dt, rates, spiked):
        spiked[...] = 0
        next_spikes = np.zeros_like(spiked)
        to_process = np.ones_like(spiked, dtype=bool)

        while np.any(to_process):
            next_spikes[to_process] += self._sample_exponential(
                rates[to_process])
            to_process &= next_spikes < dt
            spiked[to_process] += self.amplitude / dt

    def step_math(self, dt, J, spiked):
        rates = np.zeros_like(J)
        LIFRate.step_math(self, dt=1, J=J, output=rates)
        self._poisson_step_math(dt, rates, spiked)


def go(freq,
       neuron_type,
       n_neurons_over_freq=50,  # scale-invariant
       n_steps=1000,
       tau_times_freq=0.01,  # dimensionless
       dt_times_freq=0.001,  # dimensionless
       max_rates=nengo.dists.Uniform(20, 40),
       seed=0,
      ):

    n_neurons = int(n_neurons_over_freq * freq)
    tau = tau_times_freq / freq
    dt = dt_times_freq / freq

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(lambda t: np.sin(freq*2*np.pi*t))
        x = nengo.Ensemble(n_neurons, 1, neuron_type=neuron_type,
                           max_rates=max_rates)
        nengo.Connection(u, x, synapse=None)    
        p_actual = nengo.Probe(x, synapse=tau)
        p_ideal = nengo.Probe(u, synapse=tau)

    with nengo.Simulator(model, dt=dt) as sim:
        if isinstance(neuron_type, nengo.LIF):
            # https://github.com/nengo/nengo/issues/1415
            # uniform is a decent approximation for the voltage
            # after the mixing time
            sim.signals[sim.model.sig[x.neurons]['voltage']] = (
                np.random.RandomState(seed=seed).rand(n_neurons))
        
            # For some extra assurances, run
            # until every neuron has spiked at least once
            sim.run(1. / np.max(sim.data[x].max_rates))
            
            # Then clear that probe data
            # https://github.com/nengo/nengo/issues/963
            for probe in sim.model.probes:
                sim._probe_outputs[probe] = []
            
        sim.run_steps(n_steps)

    return nengo.utils.numpy.rmse(
        sim.data[p_actual], sim.data[p_ideal])


n_trials = 10

########################

data = defaultdict(list)
for seed in range(n_trials):
    for neuron_type in (PoissonLIF(seed=seed), nengo.neurons.LIF()):
        for freq in np.linspace(1, 101, 11):
            data['Model'].append(type(neuron_type).__name__)
            data['Frequency (Hz)'].append(freq)
            data['Seed'].append(seed)
            data['RMSE'].append(go(freq, neuron_type, seed=seed))

plt.figure(figsize=(14, 6))
sns.lineplot(data=DataFrame(data), x="Frequency (Hz)", y="RMSE", hue="Model")
savefig("poisson-frequency-scaling.pdf")

########################

data = defaultdict(list)
for seed in range(n_trials):
    for neuron_type in (PoissonLIF(seed=seed), nengo.neurons.LIF()):
        for n_neurons_over_freq in np.linspace(10, 1001, 11):
            freq = 10
            data['Model'].append(type(neuron_type).__name__)
            data['# Neurons'].append(n_neurons_over_freq * freq)
            data['Seed'].append(seed)
            data['RMSE'].append(go(freq, neuron_type, n_neurons_over_freq=n_neurons_over_freq, seed=seed))

plt.figure(figsize=(14, 6))
sns.lineplot(data=DataFrame(data), x="# Neurons", y="RMSE", hue="Model")
plt.xscale('log')
savefig("poisson-neuron-scaling.pdf")
