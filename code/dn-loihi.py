# Delay Network on Loihi
# Assumes nengo-loihi==0.5.0

from dn import go

import nengo_loihi

import nengo
from nengo.solvers import LstsqL2


def pre_fixture(dummy_model):
    nengo.Ensemble.max_rates.default = nengo.dists.Uniform(100, 120)
    nengo.Ensemble.intercepts.default = nengo.dists.Uniform(-1, 0.5)


def factory(network, dt):
    return nengo_loihi.Simulator(
        network, dt=dt,
        precompute=True,
        remove_passthrough=True
    )


if __name__ == '__main__':
    print(go("loihi",
             tau=0.01,  # any higher causes issues
             factory=factory,
             recurrent_solver=LstsqL2(reg=0.1, weights=True),
             pre_fixture=pre_fixture))
