from phd import *

from nengolib.synapses import pade_delay_error

freq_times_theta = np.linspace(0, 15, 1000)
orders = range(3, 28, 2)

colors = sns.color_palette("hls", len(orders))

plt.figure(figsize=(12, 4))
for color, order in zip(colors, orders):
    plt.plot(freq_times_theta,
             np.abs(pade_delay_error(freq_times_theta, order=order)),
             c=color, lw=2, label="q=%s" % order)
plt.xlabel(r"$f \cdot \theta$ (Hz $\times \, s$)")
plt.ylabel(r"$|E_q(2 \pi i f \cdot \theta)|$")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
sns.despine(offset=10)

savefig("pade-delay-error.pdf")

print(np.abs(pade_delay_error(5, 21)))
