from phd import *

from scipy.linalg import expm, inv

from nengolib.networks.rolling_window import readout


def canonical_basis(dimensions, t):
    """Temporal basis functions for PadeDelay in canonical form."""
    t = np.atleast_1d(t)
    B = np.asarray([readout(dimensions, r) for r in t])
    return B


def basis(T, dimensions, t):
    """Temporal basis functions for realized PadeDelay."""
    # Undo change of basis from realizer, and then transform into window
    B = canonical_basis(dimensions, t)
    return B.dot(T)


def inverse_basis(T, dimensions, t):
    """Moore-Penrose pseudoinverse of the basis functions."""
    B = basis(T, dimensions, t)
    return inv(B.T.dot(B)).dot(B.T)


def normalize(x):
    return x / np.std(x)


def compute_periodicity(x, cycles):
    parts = np.asarray(np.split(x, cycles))
    assert parts.shape == (cycles, len(x) // cycles)
    combined = np.mean(parts, axis=0)
    assert combined.shape == (len(x) // cycles,)
    return nengo.utils.numpy.rms(combined)


def go(
    ax,
    cycles,
    n_eval_points=1000,  # for basis solver
    dt=0.001,
    freq=22,
    theta=0.240,  # gives highly-divisble n_steps
    order=20,
    n_trials=500,
    seed=0,
):
    n_steps = int(theta / dt)
    assert np.allclose(n_steps, theta / dt)
    n_steps_per_cycle = int(n_steps / cycles)
    assert np.allclose(n_steps_per_cycle, n_steps / cycles)
    t = np.arange(0, theta, dt)

    pade_error = np.abs(nengolib.synapses.pade_delay_error(theta*freq, order=order))
    print(pade_error)

    rng = np.random.RandomState(seed=0)

    def generate_cyclic():
        process = nengo.processes.WhiteSignal(high=freq, period=theta/cycles, y0=0)
        signal = process.run(theta/cycles, dt=dt, rng=rng).squeeze()
        return normalize(np.tile(signal, cycles))

    def generate_acyclic():
        process = nengo.processes.WhiteSignal(high=freq, period=theta, y0=0)
        return normalize(process.run(theta, dt=dt, rng=rng).squeeze())

    realizer = nengolib.signal.Balanced()
    realization_result = realizer(nengolib.synapses.PadeDelay(theta, order=order))
    sys = realization_result.realization

    H = sum(expm(sys.A * i * theta / cycles) / cycles for i in range(cycles))

    tfull = np.linspace(0, 1, n_eval_points)
    tpart = np.linspace(1-1/cycles, 1, n_eval_points)
    Bpart = basis(realization_result.T, order, tpart)
    Binc = inverse_basis(realization_result.T, order, tfull)
    assert Bpart.shape == (n_eval_points, order)
    assert Binc.shape == (order, n_eval_points)
    R = Binc.dot(Bpart)   
    P = R.dot(H)

    ideal_label = r"$p_k(t)$"
    output_label = r"$\|\| P_%d\mathbf{x}(t) \|\|$" % cycles
    class_label = "Class"
    
    data = defaultdict(list)
    for i in range(n_trials):
        for condition, u in (('Aperiodic', generate_acyclic()),
                             ('Periodic', generate_cyclic()),):
            x = sys.X.filt(u, y0=0, dt=dt)[-1, :]
            x_hat = P.dot(x)

            data['Seed'].append(i)
            data[class_label].append(condition)
            data[output_label].append(np.linalg.norm(x_hat))
            data[ideal_label].append(compute_periodicity(u, cycles))
            
    df = DataFrame(data)
    
    ax.set_title("k=%d" % cycles)
    sns.scatterplot(data=df, x=output_label, y=ideal_label,
                    hue=class_label, ax=ax)

###################################

order = 20
cycles = 5

realizer = nengolib.signal.Hankel()
realization_result = realizer(nengolib.synapses.PadeDelay(1, order=order))
sys = realization_result.realization

tfull = np.linspace(0, 1, 1000)
Bfull = basis(realization_result.T, order, tfull)
tpart = np.linspace(1-1/cycles, 1, len(tfull))
Bpart = basis(realization_result.T, order, tpart)

assert Bfull.shape == (len(tfull), order)
assert Bpart.shape == (len(tfull), order)

R, _ = nengo.solvers.Lstsq(rcond=0.1)(Bfull, Bpart)
assert R.shape == (order, order)

plt.figure(figsize=(6.5, 5))
plt.imshow(R)
plt.gca().set_axis_off()
plt.colorbar()
savefig("zoom-matrix.pdf")

###################################

k_values = (2, 3, 4, 5)
fig, axes = plt.subplots(1, len(k_values), sharey=True,
                         figsize=(14, 3))
for ax, k in zip(axes, k_values):
    go(ax=ax, cycles=k)
    ax.axis('equal')
    if k == k_values[-1]:
        ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    else:
        ax.get_legend().remove()

savefig("periodicity.pdf")
