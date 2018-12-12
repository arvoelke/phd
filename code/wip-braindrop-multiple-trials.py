from phd import *

#################################

plt.style.use(datapath('ieee_tran.mplstyle'))

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('text', usetex=True)

#################################

import time

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

import nengo_brainstorm as brd

from nengolib import Lowpass
from nengolib.networks import readout
from nengolib.signal import Balanced, nrmse
from nengolib.synapses import PadeDelay, pade_delay_error, ss2sim


print(nengo.__version__)
print(nengolib.__version__)
# nengo_brainstorm=aaron-mismatch-hacks (branch)
# pystorm=modify_flush


from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D


class HandlerDashedLines(HandlerLineCollection):
    """Adapted from http://matplotlib.org/examples/pylab_examples/legend_demo5.html"""  # noqa: E501

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(
            legend, xdescent, ydescent, width / numlines, height, fontsize)
        leglines = []
        for i in range(numlines):
            legline = Line2D(
                xdata + i * width / numlines,
                np.zeros_like(xdata.shape) - ydescent + height / 2)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[0] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines


def analyze(name, t, u, x_hat, x_ideal, C, dump_file=True, do_plot=True):
    #print("Radii:", np.max(np.abs(x_hat), axis=0))
    w = C.dot(x_hat.T)
    w_ideal = C.dot(x_ideal.T)
    assert C.shape == (t_samples, order)
    
    if do_plot:
        top_cmap = sns.color_palette('GnBu_d', t_samples)[::-1]
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3.5))
        for c, w_i in list(zip(top_cmap, w))[::-1]:
            ax[0].plot(t, w_i, c=c, alpha=0.7)
        target_line, = ax[0].plot(t, u, c='green', linestyle='--', lw=1)
        ax[0].set_ylim(np.min(w), np.max(w) + 1)

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        insert = inset_axes(ax[0], width="25%", height=0.3, loc='upper right')
        insert.patch.set_alpha(0.8)
        insert.xaxis.tick_top()
        insert.tick_params(axis='x', labelsize=4)
        insert.tick_params(axis='y', labelsize=4)
        insert.xaxis.set_label_position('top') 
        t_window = np.linspace(0, theta, t_samples)
        e_window = nrmse(w, target=w_ideal, axis=1)
        for i in range(1, t_samples):
            insert.plot([t_window[i-1], t_window[i]],
                        [e_window[i-1], e_window[i]],
                        c=top_cmap[i])
        insert.set_xlabel("Delay Length (s)", size=4)
        insert.set_ylabel("NRMSE", size=4)
        #insert.set_ylim(0, max(e_window))

        bot_cmap = sns.color_palette('bright', order)
        for i in range(order):
            ax[1].plot(t, x_hat[:, i], c=bot_cmap[i], alpha=0.9)
            ax[1].plot(t, x_ideal[:, i], c=bot_cmap[i], linestyle='--', lw=1)

        ax[0].set_title("Delay Network")
        ax[1].set_title("State Vector")
        ax[-1].set_xlabel("Time (s)")

        top_lc = LineCollection(
            len(C) * [[(0, 0)]], lw=1, colors=top_cmap)
        ax[0].legend([target_line, top_lc], ["Input", "Output"],
                     handlelength=3.2, loc='lower right',
                     handler_map={LineCollection: HandlerDashedLines()})

        bot_lc_ideal = LineCollection(
            order * [[(0, 0)]], lw=1, colors=bot_cmap, linestyle='--')
        bot_lc_actual = LineCollection(
            order * [[(0, 0)]], lw=1, colors=bot_cmap)
        ax[1].legend([bot_lc_ideal, bot_lc_actual], ["Ideal", "Actual"],
                     handlelength=3.2, loc='lower right',
                     handler_map={LineCollection: HandlerDashedLines()})

        fig.savefig('%s.pdf' % name, dpi=600, bbox_inches='tight')

    if dump_file:
        np.savez("%s-%s" % (name, time.time()),
                 t=sim.trange(), u=u, x_hat=x_hat, x_ideal=x_ideal, C=C)

    return nrmse(w.flatten(), target=w_ideal.flatten())


theta = 0.1
order = 3
freq = 3
power = 1.0  # chosen to keep radii normalized to [-1, 1]

print("PadeDelay(%s, %s) => %f%% error @ %sHz" % (
    theta, order, 100*abs(pade_delay_error(theta*freq, order=order)), freq))
pd = PadeDelay(theta=theta, order=order)

# Heuristic for normalizing state so that each dimension is ~[-1, +1]
rz = Balanced()(pd, radii=1./(np.arange(len(pd))+1))
sys = rz.realization

# Compute matrix to transform from state (x) -> sampled window (u)
t_samples = 100
C = np.asarray([readout(len(pd), r)
                for r in np.linspace(0, 1, t_samples)]).dot(rz.T)
assert C.shape == (t_samples, len(sys))


n_neurons = 128  # per dimension
tau = 0.018329807108324356  # guess from Terry's notebook
map_hw = ss2sim(sys, synapse=Lowpass(tau), dt=None)
assert np.allclose(map_hw.A, tau*sys.A + np.eye(len(sys)))
assert np.allclose(map_hw.B, tau*sys.B)

with nengo.Network() as model:
    brd.add_params(model)

    u = nengo.Node(output=0, label='u')
    p_u = nengo.Probe(u, synapse=None)
    
    # This is needed because a single node can't connect to multiple
    # different ensembles. We need a separate node for each ensemble.
    Bu = [nengo.Node(output=lambda _, u, b_i=map_hw.B[i].squeeze(): b_i*u,
                     size_in=1, label='Bu[%d]' % i)
          for i in range(len(sys))]
    
    X = []
    for i in range(len(sys)):
        ens = nengo.Ensemble(
            n_neurons=n_neurons, dimensions=1, label='X[%d]' % i)

        X.append(ens)
 
    P = []
    for i in range(len(sys)):
        nengo.Connection(u, Bu[i], synapse=None)
        nengo.Connection(Bu[i], X[i], synapse=tau)
        for j in range(len(sys)):
            nengo.Connection(X[j], X[i], synapse=tau,
                             function=lambda x_j, a_ij=map_hw.A[i, j]: a_ij*x_j)
        P.append(nengo.Probe(X[i], synapse=None))


with model:
    from nengo_brainstorm import solvers
    solver = solvers.FallbackSolver([nengo.solvers.LstsqL2(reg=0.01),
                                     solvers.CVXSolver(reg=0.01)])
    model.config[nengo.Ensemble].solver = solver
    
factory = lambda model, dt: brd.Simulator(
    model, dt,
    precompute_inputs=True,
    compute_stats=False,
    generate_offset=1.0,
    precompute_offset=1.0,
)


def do_trial(name, seed, factory, length=2000, dt=0.001, tau_probe=0.02,
             sanity=False, **kwargs):
    # Note: depends on the globals, (C, model, u, p_u, P, sys)

    process = nengo.processes.WhiteSignal(
        period=length*dt, rms=power, high=freq, y0=0, seed=seed)

    test_u = process.run_steps(length, dt=dt)
    x_ideal = sys.X.filt(test_u, dt=dt)

    if sanity:
        analyze("ideal-%s" % name, 
                t=process.ntrange(length, dt=dt),
                u=test_u,
                x_hat=x_ideal,
                x_ideal=x_ideal,
                C=C,
                dump_file=False,
                do_plot=False)
        
    u.output = process

    with factory(model=model, dt=dt) as sim:
        sim.run(length*dt)

    assert np.allclose(test_u, np.asarray(sim.data[p_u]))
    assert sim.hal.get_overflow_counts() == 0

    # Use discrete principle 3, offline, to get x_hat
    # from the unfiltered spikes representing x.
    # This is analagous to probing the PSC, pre-encoding.
    syn_probe = Lowpass(tau_probe)
    map_out = ss2sim(sys, synapse=syn_probe, dt=dt)
    x_raw = np.asarray([sim.data[p] for p in P]).squeeze()
    f = map_out.A.dot(x_raw) + map_out.B.dot(test_u.T)
    x_hat = syn_probe.filt(f, axis=1, dt=dt).T

    return analyze(
        name=name, t=sim.trange(), u=test_u,
        x_hat=x_hat, x_ideal=x_ideal, C=C,
        **kwargs)


data = defaultdict(list)

for trial in range(25):
    for seed in range(1, 11):
        data['Trial'].append(trial)
        data['Test Case (#)'].append(seed)
        data['NRMSE'].append(
            do_trial(name="scratch-braindrop-DN-%d-%d" % (trial, seed),
                     seed=seed, factory=factory, dump_file=False))

df = DataFrame(data)
df.to_pickle(datapath("braindrop-delay-network.pkl"))

print(bs.bootstrap(np.asarray(df['NRMSE']), stat_func=bs_stats.mean, alpha=1-0.95))
