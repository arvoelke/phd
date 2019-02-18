# Adapted from _braindrop_integrator.ipynb
# Must be on nengo_loihi branch=integrator-accuracy (PR #124)

from phd import *
from vens import VirtualEnsemble
from dn import set_style
set_style()

import nengo_loihi
nengo.Ensemble.max_rates.default = nengo.dists.Uniform(100, 120)
nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-1, 0.5)
# nengo_loihi.set_defaults()

import nengolib
from nengolib.signal import s, nrmse
from nengolib.synapses import ss2sim

import warnings
warnings.filterwarnings("ignore")  # otherwise 100+ MB of loihi warnings will crash your computer!


def init_generators(sim, name='host_pre', rng=np.random):
    """Jitters all of the spike-generators to reduce spike-synchrony."""
    sim_host = sim.sims[name]
    i = 0
    for a in sim_host.model.sig:
        if isinstance(a, nengo.Ensemble) and isinstance(a.neuron_type, nengo_loihi.neurons.NIF):
            sim_host.signals[sim_host.model.sig[a.neurons]['voltage']] = rng.rand(a.n_neurons)
            i += 1
    return i


def do_trial(n_neurons=300,
             partition=None,
             freq=1,
             sim_t=10,
             tau=0.2,
             tau_probe=0.2,
             dt=0.001,
             input_seed=0,
             ens_kwargs = {
                 'neuron_type': nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
             }):

    # Discretized Principle 3 mapping of the integrator
    A, B, C, D = ss2sim(~s, nengo.Lowpass(tau), dt=dt).ss
    assert np.allclose(C, 1)
    assert np.allclose(D, 0)

    def loihi_factory(model, dt, tau=tau):
        loihi_model = nengo_loihi.builder.Model(dt=dt)
        # https://github.com/nengo/nengo-loihi/issues/97
        assert loihi_model.decode_tau == 0.005
        loihi_model.decode_tau = tau  # used by spike generator
        return nengo_loihi.Simulator(
            model, model=loihi_model, precompute=True, dt=dt) 

    # The integral of test_u is test_x, where test_x is [-1, 1] and
    # starts at zero. 
    process = nengo.processes.WhiteSignal(
        period=sim_t, high=freq, y0=0, seed=input_seed)
    test_x = process.run(sim_t, dt=dt)
    test_x /= np.max(np.abs(test_x))
    test_u = (test_x - np.roll(test_x, 1)) / dt

    # Roll the signals so that it starts where both u and x are close to 0
    w_x = 2  # used for Braindrop experiment
    cost = test_u**2 + w_x*(test_x)**2
    test_u = np.roll(test_u, -np.argmin(cost))
    # don't need to roll test_x because we don't use it anywhere else
    
    # But due to:
    #   https://github.com/nengo/nengo-loihi/issues/115
    # we need to split the input B*u across multiple spike generators
    # each normalized to [1, 1], and furthermore move the synapse
    # to inter_tau. Both of these changes have additional benefits
    # for accuracy as well, related to altering the noise floor
    # introduced by spike generation, and scaling with dynamic range
    # (number of generators created).
    split = np.ceil(np.max(np.abs(B*test_u))).astype(int)

    with nengo.Network() as model:
        u = nengo.Node(output=nengo.processes.PresentInput(test_u, dt))

        if partition is None:
            x = nengo.Ensemble(n_neurons, 1, **ens_kwargs)

            make_input_connection = nengo.Connection

            nengo.Connection(x, x, transform=A, synapse=tau,
                             solver=nengo.solvers.LstsqL2(weights=True))

            p_x = nengo.Probe(x, synapse=tau_probe)

        else:
            if n_neurons % partition != 0:
                raise ValueError("n_neurons (%s) must be divisible by partition (%s)" % (
                    n_neurons, partition))

            x = VirtualEnsemble(n_ensembles=n_neurons // partition,
                                n_neurons_per_ensemble=partition,
                                dimensions=1,
                                **ens_kwargs)

            def make_input_connection(pre, post, **kwargs):
                return post.add_input(u, **kwargs)

            x.add_input(x.add_output(dt=dt)[0], transform=A, synapse=tau)

            # Copy the output so that the above is collapsed as a passthrough
            p_x = nengo.Probe(x.add_output(dt=dt)[0], synapse=tau_probe)

        for _ in range(split):
            # Use synapse=None here because the spike generator
            # will have a synapse of tau
            make_input_connection(u, x, transform=B / split, synapse=None)

        p_u = nengo.Probe(u, synapse=None)
        p_ideal = nengo.Probe(u, synapse=nengolib.Lowpass(tau_probe) / s)

    with loihi_factory(model, dt) as sim:
        init_generators(sim)
        sim.run(sim_t)
        
    return {
        't': sim.trange(),
        'u': sim.data[p_u],
        'actual': sim.data[p_x],
        'ideal': sim.data[p_ideal],
    }


if __name__ == '__main__':
    data = []
    for i in range(200):
        print("Trial #", i)
        data.append(do_trial(n_neurons=256, partition=None,
                             freq=1, sim_t=10, input_seed=1,
                             tau=0.2, tau_probe=0.2))

    np.savez(datapath("loihi-integrator.npz"), data=data)

    flattened = []
    for r in data:
        flattened.append(r['actual'])
        # all r['t'] and r['ideal'] are identical

    plt.figure(figsize=(3.5, 2))

    blue, _, green = sns.color_palette('bright', 3)
    plt.plot(r['t'], r['ideal'], c=green, linestyle='--', lw=1, label="Ideal")
    sns.tsplot(flattened, r['t'], ci=95, color=blue, condition="Output")

    plt.xlabel("Time (s)")
    plt.legend()

    savefig("loihi-integrator.pdf")
