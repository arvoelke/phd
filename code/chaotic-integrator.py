# Adapted from https://forum.nengo.ai/t/robustness-of-deneves-networks/320/3

from phd import *

def go(n_neurons=10, perturb=0, perturb_T=0.001, T=5.0, dt=1e-4,
       tau=0.005, seed=0, solver=nengo.solvers.LstsqL2(reg=1.0),
       sample_every=0.02):

    with nengolib.Network(seed=seed) as model:
        stim = nengo.Node(output=lambda t: perturb if t <= perturb_T else 0)
        
        x = nengo.Ensemble(n_neurons, 1, seed=seed)
        nengo.Connection(stim, x, synapse=None)
        nengo.Connection(x, x, synapse=tau, solver=solver)

        p_voltage = nengo.Probe(x.neurons, 'voltage', sample_every=sample_every)
        p_x = nengo.Probe(x, synapse=tau, sample_every=sample_every)
        
    with nengo.Simulator(model, dt=dt, seed=seed) as sim:
        sim.run(T)
        
    return sim.trange(dt=sample_every), sim.data[p_voltage], sim.data[p_x]


t, v1, x1 = go()

x0 = -15
_, v2, x2 = go(perturb=10**x0)

delta = np.log(np.linalg.norm(v1 - v2, axis=1))

t1 = 0
t2 = 147
d1 = delta[t1]
d2 = delta[t2]
tlim = t <= 5.0
lyap = (d2 - d1) / (t[t2] - t[t1])

colors = sns.color_palette(None, 6)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(t, x1, c=colors[0], alpha=0.5, label="$x_1$") #(0) = 0$")
axes[0].plot(t, x2, c=colors[1], alpha=0.5, label="$x_2$") #(0) = 10^{%d}$" % x0)
axes[0].set_ylabel(r"$x(t)$")
axes[0].set_ylim(-1, 1)
axes[0].set_xlabel("Time (s)")
axes[0].legend(loc='upper right')

l1 = axes[1].plot(t[tlim], delta[tlim],
                  alpha=0.5, c='black', label=r"$\delta$")
l2 = axes[1].plot((t[t1], t[t2]), (d1, d2), linestyle='--',
                  c=colors[2], lw=3,
                  label=r"$\lambda \approx %.1f$ s${}^{-1}$" % lyap)
axes[1].set_ylabel(r"$ln \|| \delta(t) \||$")
axes[1].set_xlabel("Time (s)")

axes2 = axes[1].twinx()
l3 = axes2.plot(t[tlim], x1[tlim] - x2[tlim], c=colors[3],
                label=r"$\Delta$")
axes2.set_frame_on(False)
axes2.set_ylim(-0.5, 0.5)
axes2.set_ylabel(r"$\Delta(t)$")

lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
axes[1].legend(lns, labs, loc='upper left')

offset = 10
axes2.spines['right'].set_position(('outward', offset))
sns.despine(offset=offset, ax=axes[0])
sns.despine(offset=offset, ax=axes[1], right=False)

savefig("chaotic-integrator.pdf")

print(1/lyap)
