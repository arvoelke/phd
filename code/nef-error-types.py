from phd import *

from scipy.stats import kstest

from nengo.utils.matplotlib import rasterplot

n_neurons = 100
seed = 0
u0 = 1
tau = 0.05

with nengo.Network() as model:
    
    u = nengo.Node(u0)
    
    x = nengo.Ensemble(
        n_neurons, 1,
        max_rates=nengo.dists.Uniform(10, 20),
        intercepts=nengo.dists.Uniform(-1, -0.5),
        seed=seed)

    nengo.Connection(u, x, synapse=None)
    conn = nengo.Connection(x, nengo.Node(size_in=1))

    p_x = nengo.Probe(x, synapse=tau)
    p_v = nengo.Probe(x.neurons, 'voltage', synapse=None)
    p_r = nengo.Probe(x.neurons, 'refractory_time', synapse=None)
    p_spikes = nengo.Probe(x.neurons, 'spikes', synapse=None)

with nengo.Simulator(model, dt=1e-5) as sim:
    sim.run(0.3)

fudge1 = -0.01
fudge2 = -0.008
fudge3 = 0.096
fudge4 = -0.007
fudge5 = 0.2
fudge6 = -0.008
fudge7 = -0.015

t_mixed = 0.25
t_spike = 1. / np.max(sim.data[x].max_rates) - x.neuron_type.tau_ref
d = sim.data[conn].weights.T
a = nengo.builder.ensemble.get_activities(sim.data[x], x, [u0])
x_hat = a.dot(d)

ones = np.ones_like(sim.trange())
filt_x_hat = p_x.synapse.filt(x_hat*ones, y0=0, dt=sim.dt)
filt_u = p_x.synapse.filt(u0*ones, y0=0, dt=sim.dt)

fig, axes = plt.subplots(
    3, 1, figsize=(16, 9), sharex=False,
    gridspec_kw={'height_ratios': [1, 1, 3],
                 'hspace': 0})
v, r, o = axes

# Background colour indicators
_, _, green, red, _ = sns.color_palette(None, 5)
for axis in axes:
    axis.axvspan(0, t_spike, alpha=0.1, color=red, lw=0)
    axis.axvspan(t_mixed, t_mixed + t_spike, alpha=0.1, color=green, lw=0)
    
def pad_ylim(ax, amt):
    bot, top = ax.set_ylim()
    ax.set_ylim(bot - amt, top + amt)

v.annotate(text='Transient', xy=(t_spike / 2 - 0.005, 1.25))
v.annotate(text='Mixed', xy=(t_mixed + t_spike / 2 - 0.005, 1.25))

# Voltage plot
v.plot(sim.trange(), sim.data[p_v], alpha=0.5)
v.set_ylabel("Voltage")
pad_ylim(v, 0.2)

# Refractory plot
r.plot(sim.trange(), 1e3 * (sim.data[p_r] - sim.dt).clip(0), alpha=0.5)
r.set_ylabel("Refractory (ms)", labelpad=14)

arrowprops = dict(arrowstyle='<->', shrinkA=0, shrinkB=0)

# Annotate (d)
t1 = t_spike + fudge1
y1 = filt_u[int(t1 / sim.dt) - 1]
o.annotate(text='',
           xy=(t1, 0),
           xytext=(t1, y1),
           arrowprops=arrowprops)
o.annotate(text='(d)',
           xy=(t1 + fudge2, y1 / 2),
           fontweight='bold')

# Annotate (c)
t2 = int(fudge3 / sim.dt) - 1
o.annotate(text='',
           xy=(fudge3, sim.data[p_x][t2]),
           xytext=(fudge3, filt_u[t2]),
           arrowprops=arrowprops)
o.annotate(text='(c)',
           xy=(fudge3 + fudge4, (sim.data[p_x][t2] + filt_u[t2]) / 2),
           fontweight='bold')

# Annotate (b)
t3 = int(fudge5 / sim.dt) - 1
o.annotate(text='',
           xy=(fudge5, filt_x_hat[t3]),
           xytext=(fudge5, filt_u[t3]),
           arrowprops=arrowprops)
o.annotate(text='(b)',
           xy=(fudge5 + fudge6, (filt_x_hat[t3] + filt_u[t3]) / 2 + fudge7),
           fontweight='bold')

# Output plot
o.plot(sim.trange(), filt_x_hat,
           label=r"$\hat{x}$")
o.plot(sim.trange(), sim.data[p_x],
           label=r"$d^T a$")
o.plot(sim.trange(), filt_u,
           linestyle='--', label=r"x", lw=2)

o.legend(loc='upper left')
o.set_ylabel("Output", labelpad=8)
axes[-1].set_xlabel("Time (s)")

# Axes/spine polishing
sns.despine(offset=10)
for ax in (v, r):
    ax.set_xticks([])
    sns.despine(bottom=True, ax=ax)
for ax in axes:
    ax.set_xlim(0, sim.trange()[-1])

savefig("nef-error-types-a.pdf")

################################

J = sim.data[x].gain * sim.data[x].encoders.dot(u0).squeeze() + sim.data[x].bias
is_active = (J > 1)


def spike_times(J, voltage, refractory_time, tau_rc=x.neuron_type.tau_rc):
    # v(t) = v(0) + (J - v(0))*(1 - exp(-t/tau)) 
    # 1 - v = (J - v)*(1 - exp(-t/tau))
    # (1 - v) / (J - v) = (1 - exp(-t/tau))
    # exp(-t/tau) = 1 - (1 - v) / (J - v)
    # exp(-t/tau) = (J - 1) / (J - v)
    # 1/exp(-t/tau) = (J - v) / (J - 1)
    # 1/exp(-t/tau) = (J - v - (J - 1) + (J - 1)) / (J - 1)
    # 1/exp(-t/tau) = (1 - v) / (J - 1) + 1
    # -t/tau = -logp((1 - v) / (J - 1))
    # t = tau*logp((1 - v) / (J - 1))

    t_spike = tau_rc * np.log1p((1 - voltage) / (J - 1))
    return t_spike + (refractory_time - sim.dt).clip(0)


def sample_across(t_slice):
    samples = []
    for i in np.where(t_slice)[0]:
        t = spike_times(J[is_active],
                        sim.data[p_v][i, is_active],
                        sim.data[p_r][i, is_active])
        rel = t * a[is_active]
        samples.extend(rel)
    return samples


t_window = 0.025
t_indices = np.linspace(0, len(sim.trange())-1-int(t_window/sim.dt), 500,
                        dtype=int)
t_samples = sim.trange()[t_indices] + t_window / 2

K = []
P = []
all_samples = []
for t_i in t_samples:
    t_range = ((sim.trange() >= t_i - t_window / 2) &
               (sim.trange() < t_i + t_window / 2))
    samples = sample_across(t_range)
    k, p = kstest(samples, 'uniform')
    all_samples.append(samples)
    K.append(k)
    P.append(p)

print("(d) Maximum p-value:", np.max(P))

from mpl_toolkits.axes_grid1.inset_locator import inset_axes 

fig, ax = plt.subplots(1, 1, figsize=(6, 2.2))

insert1 = inset_axes(ax, width=2, height=0.7, loc='upper left',
                     bbox_to_anchor=(0, 1.8), bbox_transform=ax.transAxes)
sns.kdeplot(all_samples[0], clip=(0, 1), ax=insert1, c=red, shade=True)
insert1.set_title("Transient")
insert1.set_xlabel("ISI Position")
insert1.set_xlim(0, 1)
insert1.set_yticks([])
insert1.set_ylabel("Density")

i = -1
insert2 = inset_axes(ax, width=2, height=0.7, loc='upper right',
                     bbox_to_anchor=(1, 1.8), bbox_transform=ax.transAxes)
sns.kdeplot(all_samples[i], clip=(0, 1), ax=insert2, c=green, shade=True)
#insert2.hist(all_samples[i], range=(0, 1), color=green)
insert2.set_title("Mixed")
insert2.set_xlabel("ISI Position")
insert2.set_xlim(0, 1)
insert2.set_yticks([])

ax.plot(t_samples, K)
ax.scatter(t_samples[0], K[0], c=red, s=50, zorder=3)
ax.scatter(t_samples[i], K[i], c=green, s=50, zorder=3)

ax.set_xlabel("Time (s)")
ax.set_ylabel("KS-Statistic")

for axis in [ax]:
    axis.axvspan(0, t_spike, alpha=0.2, color=red, lw=0)
    axis.axvspan(t_mixed, t_mixed + t_spike, alpha=0.2, color=green, lw=0)

sns.despine(offset=10)
sns.despine(ax=insert1, left=True)
sns.despine(ax=insert2, left=True)
    
savefig("nef-error-types-d.pdf")

################################

t_test = sim.trange() > 0.1

noise = sim.data[p_x][t_test, 0] - filt_x_hat[t_test]
print("(c) Noise KS-Test:", kstest(noise / np.std(noise), 'norm'))

plt.figure(figsize=(6, 3.9))

sns.kdeplot(noise, shade=True)
lower, upper = plt.xlim()
radius = 0.4  # max(-lower, upper)
plt.xlim(-radius, radius)
plt.xlabel(r"$d^T a - \hat{x}$")
plt.ylabel("Density")

sns.despine(left=True)
plt.yticks([])

savefig("nef-error-types-c.pdf")

################################

eval_points = np.linspace(-1, 1)
A = nengo.builder.ensemble.get_activities(sim.data[x], x, eval_points[:, None])
eval_approx = A.dot(d).squeeze()

fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True,
                       gridspec_kw={'height_ratios': [1.5, 3]})

ax[0].plot(eval_points, eval_approx - eval_points, c=red, label="Error")
ax[0].legend()
ax[0].set_ylabel(r"$\hat{x} - x$")

ax[1].plot(eval_points, eval_approx, label=r"$\hat{x}$")
ax[1].plot(eval_points, eval_points, c=green, lw=2, linestyle='--', label="$x$")
ax[1].legend()
ax[1].set_ylabel("Output")
ax[1].set_xlabel(r"$x$")

sns.despine(offset=10)

savefig("nef-error-types-b.pdf")
