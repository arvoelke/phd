# Delay Network on Nengo reference simulator (2.8.0)

from dn import go

import nengo

if __name__ == '__main__':
    print(go("nengo", tau=0.01, factory=nengo.Simulator))
