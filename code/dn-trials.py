from phd import *

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

hardware = ('Braindrop', 'Loihi', 'Desktop')
fnames = ('braindrop', 'loihi', 'nengo')

frames = []
for hw, fname in zip(hardware, fnames):
    frame = pd.read_pickle(datapath('%s-delay-network.pkl' % fname))
    frame['Hardware'] = hw
    frames.append(frame)
data = pd.concat(frames)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 5),
                       gridspec_kw={'width_ratios': [3, 1]})

sns.violinplot(data=data, x='Test Case (#)', y='NRMSE', hue='Hardware', ax=ax[0])
ax[0].legend(loc="upper left", bbox_to_anchor=(0.95, 1))

for hw in hardware:
    e = data[data['Hardware'] == hw]['NRMSE']
    print(hw, bs.bootstrap(np.asarray(e), stat_func=bs_stats.mean, alpha=1-0.95))  # 95% CI
    sns.kdeplot(e, vertical=True, label=hw, ax=ax[1])
ax[1].get_legend().remove()
ax[1].set_xlabel("Density")

sns.despine(offset=10)

savefig("dn-trials.pdf")
