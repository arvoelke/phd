from phd import *

def stim(t):
    return (np.sin(-t - 1.) + np.sin(-3 * t) + np.sin(-5 * (t + 1))) / 3.

with nengo.Network() as model:
    u = nengo.Node(stim)

    peak = nengo.Ensemble(1, dimensions=2, neuron_type=nengo.Direct())
        
    nengo.Connection(u, peak[1], synapse=None)
        
    function = lambda x: (x[1] - x[0]).clip(min=0) + x[0]
    nengo.Connection(peak, peak[0], synapse=~z, function=function)
        
    p_u = nengo.Probe(u, synapse=0.01)
    p_peak = nengo.Probe(peak[0], synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(6)

plt.figure()
plt.plot(sim.trange(), sim.data[p_u], label=r"$u[k]$")
plt.plot(sim.trange(), sim.data[p_peak], label="Peak")
plt.legend(loc='best')
plt.xlabel("Time (s)")
plt.ylim(-1.1, 1.1)
savefig("wta-peak-detector.pdf")
# plt.show()
