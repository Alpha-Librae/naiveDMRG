#! /usr/bin/env python
# filename: ising.py

# MPO Hamiltonian for 1D Ising model
# reference: https://itensor.org/docs.cgi?vers=cppv2&page=tutorials/MPO

# import Pauli matrices
from basic_operators import identity, zero, Sx, Sz
import numpy

def local_operator(J=1.0, h=1.0):
    """
    construct local operator for transverse field Ising model
    - J: coupling constant
    - h: Strength of external field
    return: local operator with shape (3, 3, 2, 2)
    """
    # 3x3 matrix, each element is a 2x2 matrix
    # the local operator is 4-dimensional and with shape as (3,3,2,2)
    return numpy.float64([
        [identity,  zero,    zero],
        [Sz,        zero,    zero],
        [-h*Sx,    -J*Sz,    identity]
    ])

if __name__ == '__main__':
    from mpo import MPO
    from dmrg import DMRG
    nSite = 10
    mpo = MPO(local_operator(), nSite)
    naiveDMRG = DMRG(mpo, maxM=50)
    naiveDMRG.run()
    print("--> DMRG energy: ", naiveDMRG.energy)
