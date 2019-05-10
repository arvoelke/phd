from phd import *

k = 8
tau_probe = 0.1
seed = 0
t_run = 2
n_neurons = 2500
neuron_type = nengo.LIF()


with nengo.Network(seed=seed) as model:
    process = nengo.processes.WhiteSignal(t_run, high=10, rms=0.2, y0=0)
    stim = nengo.Node(output=process)
    
    rw = []
    cc = []
    rw_probes = []
    cc_probes = []
    next_rw = stim
    next_cc = stim
    
    for i in range(k):
        rw.append(RollingWindow(
            theta=0.05, n_neurons=n_neurons, process=process,
            neuron_type=neuron_type))
        cc.append(nengo.Ensemble(
            n_neurons=n_neurons, dimensions=1, neuron_type=neuron_type))

        nengo.Connection(next_rw, rw[i].input, synapse=None)
        next_rw = rw[i].add_output(0)

        nengo.Connection(next_cc, cc[i], synapse=rw[i].input_synapse)
        next_cc = cc[i]
        
        rw_probes.append(nengo.Probe(next_rw, synapse=tau_probe))
        cc_probes.append(nengo.Probe(next_cc, synapse=tau_probe))
        
    p_stim = nengo.Probe(stim, synapse=tau_probe)

with nengo.Simulator(model, seed=seed) as sim:
    sim.run(t_run)

cmap = sns.color_palette("GnBu_d", k)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
ax[0].set_title("Standard (Feed-Forward)")
ax[1].set_title("Delay Network (Recurrent)")
for i in range(k):
    ax[0].plot(sim.trange(), sim.data[cc_probes[i]],
               c=cmap[i], alpha=0.7, label="Layer %d" % (i+1))
    ax[1].plot(sim.trange(), sim.data[rw_probes[i]],
               c=cmap[i], alpha=0.7, label="Layer %d" % (i+1))
ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
ax[0].set_ylabel("Decoded Output")
ax[1].set_ylabel("Decoded Output")
ax[1].set_xlabel("Time (s)")
sns.despine(offset=10)

savefig("comm-channels.pdf")
