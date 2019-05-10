from phd import *

from nengolib.compat import get_activities
from nengolib.neurons import init_lif
from nengo.utils.numpy import rmse

from scipy.stats import kstest, pearsonr


def go(freq, max_rates, n_neurons=2500, n_steps=10000,
       dt=1e-4, sample_every=1e-4, tau=0.02, seed=0):

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(output=lambda t: np.sin(freq*2*np.pi*t))

        x = nengo.Ensemble(n_neurons, 1, max_rates=max_rates, seed=seed,
                           neuron_type=nengo.LIF())
        x_rate = nengo.Ensemble(n_neurons, 1, max_rates=max_rates, seed=seed,
                                neuron_type=nengo.LIFRate())

        nengo.Connection(u, x, synapse=None)
        nengo.Connection(u, x_rate, synapse=None)

        p_u = nengo.Probe(u, synapse=None, sample_every=sample_every)
        p_v = nengo.Probe(x.neurons, 'voltage', synapse=None, sample_every=sample_every)
        p_j = nengo.Probe(x.neurons, 'input', synapse=None, sample_every=sample_every)
        p_r = nengo.Probe(x.neurons, 'refractory_time', synapse=None, sample_every=sample_every)
        
        p_ideal = nengo.Probe(x_rate, synapse=tau, sample_every=sample_every)
        p_actual = nengo.Probe(x, synapse=tau, sample_every=sample_every)

    with nengo.Simulator(model, dt=dt) as sim:
        init_lif(sim, x, x0=0)
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

    actual = sim.data[p_actual]
    ideal = sim.data[p_ideal]
    assert ideal.shape == actual.shape

    return (
        kstest(s, 'uniform'),
        rmse(actual, ideal),
    )


data = defaultdict(list)

max_freq = 101
tau = 1/(2*np.pi*max_freq)  # cut-off frequency of lowpass filter
rate_scale = 50
for seed in range(5):
    for freq in np.linspace(1, max_freq, 20):
        low = rate_scale
        high = 2*low
        max_rates = nengo.dists.Uniform(low, high)
        (k, p), e = go(freq, max_rates, tau=tau, seed=seed)
        
        data['Low'].append(low)
        data['High'].append(high)
        data['Frequency (Hz)'].append(freq)
        data['KS-Statistic'].append(k)
        data['p-value'].append(p)
        data['RMSE'].append(e)

df = DataFrame(data)
y = "RMSE"
c1, c2 = sns.color_palette("hls", 2)

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 5))

ax2 = ax1.twinx()
sns.regplot(data=df, x="KS-Statistic", y=y, marker='+', ax=ax3)

sns.lineplot(data=df, x="Frequency (Hz)", y="KS-Statistic", ax=ax1, c=c1, label="KS-Statistic")
sns.lineplot(data=df, x="Frequency (Hz)", y=y, ax=ax2, c=c2, label=y)

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax2.set_frame_on(False)
plt.subplots_adjust(wspace=0.27)
ax2.set_ylabel("")

offset = 10
sns.despine(offset=offset, ax=ax1, right=False)
ax2.spines['right'].set_position(('outward', offset))
sns.despine(offset=offset, ax=ax3)

fig.show()

print(pearsonr(df["KS-Statistic"], df[y]))
print(data['p-value'])

savefig("frequency-ks-test.pdf")
