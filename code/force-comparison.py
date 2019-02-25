## References
# [1] DePasquale, B., Cueva, C. J., Rajan, K., & Abbott, L. F. (2018). full-FORCE: A target-based method for training recurrent networks. PloS one, 13(2), e0191527. http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191527
# [2] Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. Neuron, 63(4), 544-557. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/

from phd import *
from nengolib import RLS, Network
from nengolib.signal import nrmse

def go(seed, theta):
    # Task parameters
    T_train = 10
    T_total = 15
    dt = 0.005

    amplitude = 1
    process = nengo.processes.WhiteSignal(T_total, high=1, rms=0.5, y0=0)

    # Fixed model parameters
    n = 500
    rng = np.random.RandomState(seed)
    ens_kwargs = dict(  # neuron parameters
        n_neurons=n,
        dimensions=1,
        neuron_type=nengolib.neurons.Tanh(),
        intercepts=[-1]*n,  # intercepts are irelevant for Tanh
        seed=seed,
    )

    # Hyper-parameters
    tau = 0.01                  # lowpass time-constant (10ms in [1])
    tau_learn = None            # filter for error / learning (needed for spiking)
    tau_probe = None            # filter for readout (needed for spiking
    learning_rate = 0.001       # 1 in [1]
    g = 1.5 # / 400             # 1.5 in [1] (scale by firing rates for spiking)
    g_in = 1 # tau / amplitude  # scale the input encoders (usually 1)
    g_out = 1.0                 # scale the recurrent encoders (usually 1)
    q = 6                       # NEF solution (with linear units)
    
    with Network(seed=seed) as model:
        u = nengo.Node(output=process)

        z = nengo.Node(size_in=1)
        nengo.Connection(u, z, synapse=nengolib.synapses.DiscreteDelay(int(theta/dt)))

        ref = nengo.Node(size_in=1)
        nengo.Connection(u, ref, synapse=nengolib.synapses.PadeDelay(theta, order=q))

    # Initial weights
    e_in = g_in * rng.uniform(-1, +1, (n, 1))  # fixed encoders for f_in (u_in)
    e_out = g_out * rng.uniform(-1, +1, (n, 1))  # fixed encoders for f_out (u)
    JD = rng.randn(n, n) * g / np.sqrt(n)  # target-generating weights (variance g^2/n)
    
    with model:
        xC = nengo.Ensemble(**ens_kwargs)
        sC = nengo.Node(size_in=n)  # pre filter
        eC = nengo.Node(size_in=1, output=lambda t, e: e if t < T_train else 0)
        zC = nengo.Node(size_in=1)  # learned output

        nengo.Connection(u, sC, synapse=None, transform=e_in)
        nengo.Connection(sC, xC.neurons, synapse=tau)
        nengo.Connection(xC.neurons, sC, synapse=None, transform=JD)  # chaos
        connC = nengo.Connection(
            xC.neurons, zC, synapse=None, transform=np.zeros((1, n)),
            learning_rule_type=RLS(learning_rate=learning_rate, pre_synapse=tau_learn))
        nengo.Connection(zC, sC, synapse=None, transform=e_out)

        nengo.Connection(zC, eC, synapse=None)  # actual
        nengo.Connection(z, eC, synapse=None, transform=-1)  # ideal
        nengo.Connection(eC, connC.learning_rule, synapse=tau_learn)
        
    with model:    
        xR = nengo.Ensemble(**ens_kwargs)
        sR = nengo.Node(size_in=n)  # pre filter

        nengo.Connection(u, sR, synapse=None, transform=e_in)
        # nengo.Connection(z, sR, synapse=None, transform=e_out)  # <- don't reencode the input!
        nengo.Connection(sR, xR.neurons, synapse=tau)
        nengo.Connection(xR.neurons, sR, synapse=None, transform=JD)
        
    with model:
        xD = nengo.Ensemble(**ens_kwargs)
        sD = nengo.Node(size_in=n)  # pre filter

        nengo.Connection(u, sD, synapse=None, transform=e_in)
        nengo.Connection(z, sD, synapse=None, transform=e_out)
        nengo.Connection(sD, xD.neurons, synapse=tau)
        nengo.Connection(xD.neurons, sD, synapse=None, transform=JD)

    with model:
        xF = nengo.Ensemble(**ens_kwargs)
        sF = nengo.Node(size_in=n)  # pre filter
        eF = nengo.Node(size_in=n, output=lambda t, e: e if t < T_train else np.zeros_like(e))

        nengo.Connection(u, sF, synapse=None, transform=e_in)
        nengo.Connection(sF, xF.neurons, synapse=tau)
        connF = nengo.Connection(
            xF.neurons, sF, synapse=None, transform=np.zeros((n, n)),
            learning_rule_type=RLS(learning_rate=learning_rate, pre_synapse=tau_learn))

        nengo.Connection(sF, eF, synapse=None)  # actual
        nengo.Connection(sD, eF, synapse=None, transform=-1)  # ideal
        nengo.Connection(eF, connF.learning_rule, synapse=tau_learn)   
        
    with model:
        # Probes
        p_z = nengo.Probe(z, synapse=tau_probe)
        p_zC = nengo.Probe(zC, synapse=tau_probe)
        p_xF = nengo.Probe(xF.neurons, synapse=tau_probe)
        p_xR = nengo.Probe(xR.neurons, synapse=tau_probe)
        p_ref = nengo.Probe(ref, synapse=tau_probe)

    with nengo.Simulator(model, dt=dt, seed=seed) as sim:
        sim.run(T_total)
        
    # We do the readout training for full-FORCE offline, since this gives better
    # performance without affecting anything else
    t_train = sim.trange() < T_train
    t_test = sim.trange() >= T_train

    solver = nengo.solvers.LstsqL2(reg=1e-2)
    wF, _ = solver(sim.data[p_xF][t_train], sim.data[p_z][t_train])
    zF = sim.data[p_xF].dot(wF)

    wR, _ = solver(sim.data[p_xR][t_train], sim.data[p_z][t_train])
    zR = sim.data[p_xR].dot(wR)

    return (
        ('Classic-FORCE', nrmse(sim.data[p_zC][t_test], target=sim.data[p_z][t_test])),
        ('Full-FORCE', nrmse(zF[t_test], target=sim.data[p_z][t_test])),
        ('No-FORCE', nrmse(zR[t_test], target=sim.data[p_z][t_test])),
        ('NEF (n=%d)' % q, nrmse(sim.data[p_ref][t_test], target=sim.data[p_z][t_test])),
    )


thetas = np.geomspace(0.01, 1, 25)
trials = 10
data = defaultdict(list)
label_theta = r'$\theta$ (s)'

for seed in range(trials):
    for theta in thetas:
        print(seed, theta)
        for name, e in go(seed=seed, theta=theta):
            data['Seed'].append(seed)
            data[label_theta].append(theta)
            data['Method'].append(name)
            data['NRMSE'].append(e)
            
plt.figure(figsize=(8, 5))
plt.plot(thetas, np.ones_like(thetas), linestyle='--', c='black', label="100% Error")
sns.lineplot(data=DataFrame(data), x=label_theta, y="NRMSE", hue="Method")
plt.xlim(thetas[0], thetas[-1])
plt.xscale('log')
plt.yscale('log')
sns.despine(offset=15)

savefig("force-comparison.pdf")