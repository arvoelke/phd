from phd import *

from nengolib.compat import get_activities


def go(freq, n_neurons=100, n_steps=10000, dt=0.001, sample_every=0.001, seed=0):
    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.sin(freq*np.pi*t))

        x = nengo.Ensemble(n_neurons, 1)

        nengo.Connection(u, x, synapse=None)

        p_u = nengo.Probe(u, synapse=None, sample_every=sample_every)
        p_v = nengo.Probe(x.neurons, 'voltage', synapse=None, sample_every=sample_every)
        p_j = nengo.Probe(x.neurons, 'input', synapse=None, sample_every=sample_every)
        p_r = nengo.Probe(x.neurons, 'refractory_time', synapse=None, sample_every=sample_every)

    with nengo.Simulator(model, dt=dt) as sim:
        sim.run_steps(n_steps)

    A = get_activities(sim.model, x, sim.data[p_u])
    assert A.shape == sim.data[p_v].shape == sim.data[p_j].shape

    a_flatten = A.flatten()
    sl = a_flatten > 0
    v = sim.data[p_v].flatten()[sl]
    j = sim.data[p_j].flatten()[sl]
    a = a_flatten[sl]
    r = (sim.data[p_r].flatten()[sl] - dt).clip(0)

    s = (x.neuron_type.tau_rc * np.log1p((1 - v) / (j - 1)) + r) * a

    return s

freqs = np.linspace(0, 100, 11)
line = np.linspace(0, 1)
cmap = sns.color_palette("GnBu_d", len(freqs))

plt.figure(figsize=(8, 8))
for i, freq in enumerate(freqs):
    #sns.kdeplot(go(freq), clip=(0, 1), cumulative=True, bw='silverman',
    #            color=cmap[i], alpha=0.5, label="%s Hz" % freq)
    plt.hist(go(freq), cumulative=True, density=True, histtype='step',
                bins=100, color=cmap[i], alpha=0.7, label="%s Hz" % freq)
    #sns.distplot(go(freq), color=cmap[i], label="%s Hz" % freq,
    #             hist_kws={'cumulative': True},
    #             kde_kws={'cumulative': True})
plt.plot(line, line, label="Ideal Uniform", linestyle='--')
plt.xlabel("Spike Time Interval")
plt.ylabel("Cumulative Distribution Function (CDF)")
plt.xlim(0, 1)
plt.legend()
savefig("spike-time-intervals.pdf")