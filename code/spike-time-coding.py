from phd import *

frequency = 20  # here we're not properly accounting for a few things
tau = 0.1
dt = 0.001
T = 5.0

with nengo.Network() as model:
    kick = nengo.Node(output=lambda t: 1.0/dt if t <= dt else 0)
    psc = nengo.Node(size_in=2)
    x = nengo.Ensemble(
        5000, dimensions=2, seed=0,
        neuron_type=nengo.LIF(),
        intercepts=nengo.dists.Uniform(0.5, 1.0),
        max_rates=nengo.dists.Uniform(20, 40))
    tA = [[1, -frequency * tau],
          [frequency * tau, 1]]
    nengo.Connection(x, psc, synapse=tau, transform=tA)
    nengo.Connection(psc, x, synapse=None)
    nengo.Connection(kick, x, transform=[[1], [1]], synapse=0.05)
    
    p = nengo.Probe(psc, synapse=None)
    p_a = nengo.Probe(x.neurons, 'spikes', synapse=None)

with nengo.Simulator(model, dt=dt) as sim:
    sim.run(T)

sample_size = 50
rng = np.random.RandomState(seed=x.seed)
e = sim.data[x].encoders
theta = np.arctan2(e[:, 1], e[:, 0])
subset = np.sort(rng.choice(range(x.n_neurons), size=sample_size, replace=False))
order = np.argsort(theta)[subset]

from nengo.utils.matplotlib import rasterplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
#from matplotlib.collections import PatchCollection

t_range = (sim.trange() >= 1.2) & (sim.trange() < 1.8)

box_sn = 24
box_h = 16
box_s1 = 1.32
box_s2 = 1.605
box_w = 0.15
neuron_slice = order[range(box_sn, box_sn+box_h)]
t_box1 = (sim.trange() >= box_s1) & (sim.trange() < box_s1 + box_w)
t_box2 = (sim.trange() >= box_s2) & (sim.trange() < box_s2 + box_w)

# skip the first two colors to avoid confusion with the two dimensions of x(t)
c1, c2 = sns.color_palette(None, 4)[2:]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].set_title("Decoded Activity")
axes[1].set_title("Spike Raster of %d / %d Neurons" % (sample_size, x.n_neurons))
axes[0].plot(sim.trange()[t_range], sim.data[p][t_range, :])
rasterplot(sim.trange()[t_range], sim.data[p_a][t_range, :][:, order], ax=axes[1])
axes[0].set_ylabel(r"$\mathbf{x}(t)$")
axes[1].set_ylabel("Neuron #")
for axis in axes:
    axis.set_xlabel("Time (s)")
    sns.despine(offset=10, ax=axis)

axes[1].add_patch(Rectangle((box_s1, box_sn), box_w, box_h+1, fill=False, color=c1, lw=1))
axes[1].add_patch(Rectangle((box_s2, box_sn), box_w, box_h+1, fill=False, color=c2, lw=1))
    
# https://matplotlib.org/gallery/axes_grid1/inset_locator_demo.html
insert1 = inset_axes(axes[1], width=2, height=4, loc=3,
                     bbox_to_anchor=(1.1, 0.1), bbox_transform=axes[1].transAxes) 
insert2 = insert1.twiny()

# https://stackoverflow.com/questions/7761778/matplotlib-adding-second-axes-with-transparent-background
insert2.set_frame_on(True)
insert2.patch.set_visible(False)
insert2.xaxis.set_ticks_position('bottom')
insert2.xaxis.set_label_position('bottom')
insert2.spines['bottom'].set_position(('outward', 20))

rasterplot(sim.trange()[t_box1], sim.data[p_a][t_box1, :][:, neuron_slice], ax=insert1,
           colors=[c1]*box_h)
rasterplot(sim.trange()[t_box2], sim.data[p_a][t_box2, :][:, neuron_slice], ax=insert2,
           colors=[c2]*box_h)

insert1.set_yticks([])
insert2.set_xlabel("Time (s)")

# https://stackoverflow.com/questions/18571474/matplotlib-can-a-plot-a-line-from-one-set-of-axes-to-another
alpha = 0.5
fudge1 = 0.5  # to adjust y-pos of arrows connecting to the frame
fudge2 = 0.32  # to fake the second arrows going behind the frame
fudge3 = 0.95  # to fake the second arrows going behind the frame
insert1.annotate('', xy=(box_s1, box_sn),
                 xytext=(box_s1, fudge1),
                 xycoords=axes[1].transData, 
                 textcoords=insert1.transData,
                 arrowprops=dict(color=c1, arrowstyle='-', linestyle='--', alpha=alpha))
insert1.annotate('', xy=(box_s1, box_sn+box_h+1),
                 xytext=(box_s1, box_h+fudge1), 
                 xycoords=axes[1].transData, 
                 textcoords=insert1.transData,
                 arrowprops=dict(color=c1, arrowstyle='-', linestyle='--', alpha=alpha))
insert2.annotate('', xy=(box_s2+box_w, box_sn), 
                 xytext=(box_s2, box_h*fudge2 + fudge1),
                 xycoords=axes[1].transData, 
                 textcoords=insert2.transData,
                 arrowprops=dict(color=c2, arrowstyle='-', linestyle='--', alpha=alpha))
insert2.annotate('', xy=(box_s2+box_w, box_sn+box_h+1), 
                 xytext=(box_s2, box_h*fudge3 + fudge1), 
                 xycoords=axes[1].transData, 
                 textcoords=insert2.transData,
                 arrowprops=dict(color=c2, arrowstyle='-', linestyle='--', alpha=alpha))

savefig('spike-time-coding.pdf')

print(1. / (box_s2 - box_s1))

print(np.count_nonzero(sim.data[p_a][t_range]) /
      x.n_neurons /
      (sim.trange()[t_range][-1] - sim.trange()[t_range][0]))
