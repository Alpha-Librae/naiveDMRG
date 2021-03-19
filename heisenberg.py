#! /usr/bin/env python
# filename: heisenberg.py

# import Pauli matrices
from basic_operators import identity, zero, Sp, Sm, Sz
import numpy

def local_operator(J=1.0, Jz=1.0, h=1.0):
    """
    construct local operator for Heisenberg model
    - J:  coupling constant
    - Jz: coupling constant
    - h:  strength of external field
    return: local operator with shape (5, 5, 2, 2)
    """
    # 5x5 matrix, each element is a 2x2 matrix
    # the local operator is 4-dimensional and with shape as (5,5,2,2)
    return numpy.float64([
        [identity, zero,   zero,   zero,  zero],
        [Sp,       zero,   zero,   zero,  zero],
        [Sm,       zero,   zero,   zero,  zero],
        [Sz,       zero,   zero,   zero,  zero],
        [-h*Sz,    J/2*Sm, J/2*Sp, Jz*Sz, identity]
    ])

if __name__ == '__main__':
    from mpo import MPO
    from dmrg import DMRG
    nSite = 10
    mpo = MPO(local_operator(), nSite)
    naiveDMRG = DMRG(mpo, maxM=50)
    naiveDMRG.run()
    print("--> DMRG energy: ", naiveDMRG.energy)
