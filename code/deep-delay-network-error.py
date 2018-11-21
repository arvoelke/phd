from phd import *

from nengolib.synapses import PadeDelay


def compute_error(phi_times_freq, depth, order, p=None):
    pf = np.asarray(phi_times_freq)
    # switch to a delay of 1 for simplicity
    # this works due to the substitution of variables: theta*s <-> 1*s'
    sys = PadeDelay(1., order, p=p)
    return sys.evaluate(pf/depth)**depth - np.exp(-2j*np.pi*pf)


def go(ax, orders, depths, upper, n_samples=10000):
    colors = sns.color_palette("hls", len(orders))
    eval_points = np.linspace(0, upper, n_samples)
    for color, depth, order in zip(colors, depths, orders):
        ax.plot(eval_points,
                 np.abs(compute_error(eval_points, depth=depth, order=order)),
                 c=color, lw=2, label="q=%d, k=%d" % (order, depth))

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14, 4))

depths = np.arange(1, 7, dtype=int)
cost = np.lcm.reduce(depths)
orders = cost // depths
assert np.allclose(np.std(orders * depths), 0)
go(axes[0], orders, depths, upper=50)
axes[0].set_title("Fixing Neuron Count ($kq$)")

depths **= 2
assert np.allclose(np.std(orders**2 * depths), 0)
go(axes[1], orders, depths, upper=100)
axes[1].set_title("Fixing Memory Usage ($kq^2$)")

axes[0].set_ylabel(r"$|E_{q,k}(2 \pi i f \cdot \theta)|$")
for ax in axes:
    ax.legend(loc='upper left')
    ax.set_xlabel(r"$f \cdot \theta$ (Hz $\times \, s$)")
    sns.despine(offset=10, ax=ax)

savefig("deep-delay-network-error.pdf")
