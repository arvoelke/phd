from phd import *

from nengolib.compat import get_activities
from nengolib.neurons import init_lif
from nengolib.signal import nrmse
#from nengo.utils.numpy import rmse

from scipy.stats import kstest


def go(freq, max_rates, n_neurons=100, n_steps=100000,
       dt=1e-4, sample_every=0.001, tau_times_freq=10., seed=0):

    tau = tau_times_freq / freq
    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.cos(freq*np.pi*t))

        x = nengo.Ensemble(n_neurons, 1, max_rates=max_rates)

        nengo.Connection(u, x, synapse=None)

        p_u = nengo.Probe(u, synapse=None, sample_every=sample_every)
        p_v = nengo.Probe(x.neurons, 'voltage', synapse=None, sample_every=sample_every)
        p_j = nengo.Probe(x.neurons, 'input', synapse=None, sample_every=sample_every)
        p_r = nengo.Probe(x.neurons, 'refractory_time', synapse=None, sample_every=sample_every)

        p_ideal = nengo.Probe(u, synapse=tau, sample_every=sample_every)
        p_actual = nengo.Probe(x, synapse=tau, sample_every=sample_every)

    with nengo.Simulator(model, dt=dt) as sim:
        init_lif(sim, x, x0=1)
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

    return kstest(s, 'uniform'), nrmse(sim.data[p_actual], target=sim.data[p_ideal])


data = defaultdict(list)

rate_scale = 50
for seed in range(25):
    for freq in np.linspace(1, 201, 20):
        low = rate_scale
        high = 2*low
        max_rates = nengo.dists.Uniform(low, high)
        (k, p), e = go(freq, max_rates, seed=seed)
        
        data['Low'].append(low)
        data['High'].append(high)
        data['Frequency (Hz)'].append(freq)
        data['KS-Statistic'].append(k)
        data['p-value'].append(p)
        data['NRMSE'].append(e)
        data['Seed'].append(seed)

df = DataFrame(data)

fig, ax1 = plt.subplots(1, 1)
ax2 = ax1.twinx()
c1, c2 = sns.color_palette("hls", 2)

sns.lineplot(data=df, x="Frequency (Hz)", y="KS-Statistic", ax=ax1, c=c1)
sns.lineplot(data=df, x="Frequency (Hz)", y="NRMSE", ax=ax2, c=c2)
#ax1.set_yscale('log')
#ax2.set_yscale('log')
fig.show()

print(data['p-value'])