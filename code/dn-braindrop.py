# Delay Network on Braindrop (Neckar et al., 2019) 
# Branches:
#   nengo_brainstorm=aaron-mismatch-hacks
#   pystorm=modify_flush

from dn import go

import nengo_brainstorm as brd
from nengo_brainstorm import solvers

import nengo
from nengo.solvers import LstsqL2


def pre_fixture(model):
    brd.add_params(model)

    solver = solvers.FallbackSolver([LstsqL2(reg=0.01),
                                     solvers.CVXSolver(reg=0.01)])
    model.config[nengo.Ensemble].solver = solver


def post_fixture(sim):
    assert sim.hal.get_overflow_counts() == 0


def factory(network, dt):
    return brd.Simulator(
        network, dt=dt,
        precompute_inputs=True,
        compute_stats=False,
        generate_offset=1.0,
        precompute_offset=1.0,
    )


if __name__ == '__main__':
    print(go("braindrop",
             tau=0.018329807108324356,  # guess from Terry's notebook
             factory=factory,
             pre_fixture=pre_fixture,
             post_fixture=post_fixture))
