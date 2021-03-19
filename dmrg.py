#! /usr/bin/env python
# filename: dmrg.py

import numpy
from mps_tensor import local_state
from mpo import MPO

# Attention to the identifiers "pos" and "idx":
# pos: the position index of REAL sites
# idx: the index of all the sites, including two dummies
# pos == 0 is the first REAL site, whose idx is 1
# idx = pos + 1

class DMRG:
    '''
    ***** Density Matrix Renormalization Group Method *****
    '''
    def __init__(self, mpo, maxM):
        # the length of the lattice
        self._size = mpo.size
        # the maximum bond dimension
        self._maxM = maxM
        # dummy local tensor and local operator
        dummySite = local_state(left=0, phys=0, right=0)
        dummyOperator = numpy.zeros((0,0,0))
        # construct MPS and MPO
        firstSite = local_state(left=1, phys=mpo.physAt(0), right=self._maxM)
        lastSite = local_state(left=self._maxM, phys=mpo.physAt(self._size-1), right=1)
        self._mps = [dummySite, firstSite] + [local_state(self._maxM, mpo.physAt(i), self._maxM) for i in range(1, self._size-1)] + [lastSite, dummySite]
        self._mpo = [dummyOperator] + mpo.data + [dummyOperator]
        # consruct tensors F, L and R
        self._tensorsF = [numpy.ones((1,) * 6)] + [None for _ in range(self._size)] + [numpy.ones((1,) * 6)]
        self._tensorsL = self._tensorsF.copy()
        self._tensorsR = self._tensorsF.copy()
        # ground state energy
        self._gsEnergy = 0.0
        # energies of each sweep iteration
        self._iterationEnergies = []
        # do right canonicalization
        self.right_canonicalize_from(idx=self._size)

    def update_local_site(self, idx, newState):
        self._mps[idx].update(newState)
        # since the current site has changed,
        # tensor F at the current site need to be updated.
        self._tensorsF[idx] = None
        # the current site has influence on the tensor L of all the sites on its right
        for i in range(idx + 1, self._size + 1):
            if self._tensorsL[i] is None:
                break
            self._tensorsL[i] = None
        # on the tensor R of all the sites on its left
        for i in range(idx - 1, 0, -1):
            if self._tensorsR[i] is None:
                break
            self._tensorsR[i] = None

    # head dummy site: idx == 0
    # real sites : idx == 1~size
    # last dummy site: idx == size+1
    def left_canonicalize_at(self, idx):
        if idx >= self._size:
            return
        u, s, vh = self._mps[idx].truncate(direction="left", maxM=self._maxM)
        # update local sites pos and pos+1
        self.update_local_site(idx, newState=u.reshape((self._mps[idx].left, self._mps[idx].phys, -1)))
        # the next site is on the right of the current site
        self.update_local_site(idx+1, newState=numpy.tensordot(numpy.dot(numpy.diag(s), vh), self._mps[idx+1].data, axes=[1, 0]))

    def left_canonicalize_from(self, idx):
        for i in range(idx, self._size):
            self.left_canonicalize_at(i)

    # head dummy site: idx == 0
    # real sites : idx == 1~size
    # last dummy site: idx == size+1
    def right_canonicalize_at(self, idx):
        if idx <= 1:
            return
        site = self._mps[idx]
        u, s, vh = site.truncate(direction="right", maxM=self._maxM)
        # update the current site
        self.update_local_site(idx, newState=vh.reshape((-1, site.phys, site.right)))
        # update next site on the left
        self.update_local_site(idx-1, numpy.tensordot(self._mps[idx-1].data, numpy.dot(u, numpy.diag(s)), axes=[2, 0]))

    def right_canonicalize_from(self, idx):
        for i in range(idx, 1, -1):
            self.right_canonicalize_at(i)

    def tensorF_at(self, idx):
        '''
        return (after computing, if it does not exists) the tensor L at site idx
        '''
        if self._tensorsF[idx] is None: # tensor F for idx does not exist
            # compute tensor F for idx
            site = self._mps[idx] # local state
            operator = self._mpo[idx] # local operator
            # compute <site|operator
            F = numpy.tensordot(site.data.conj(), operator, axes=[1, 2])
            # compute <site|operator|site>
            F = numpy.tensordot(F, site.data, axes=[4, 1])
            self._tensorsF[idx] = F
        return self._tensorsF[idx]

    def tensorL_at(self, idx):
        if self._tensorsL[idx] is None: # tensor L for idx does not exist
            # compute tensor L for idx
            if idx <= 1: # for the first REAL site, L == F
                self._tensorsL[idx] = self.tensorF_at(idx)
            else:
                leftL = self.tensorL_at(idx - 1)
                currentF = self.tensorF_at(idx)
                self._tensorsL[idx] = numpy.tensordot(leftL, currentF, axes=[[1, 3, 5], [0, 2, 4]]).transpose((0, 3, 1, 4, 2, 5))
        return self._tensorsL[idx]

    def tensorR_at(self, idx):
        '''
        return (after computing, if it does not exists) the tensor R at site idx
        '''
        if self._tensorsR[idx] is None: # tensor R for idx does not exist
            # compute tensor R for idx
            if idx >= self._size: # for the last REAL site, R == F
                self._tensorsR[idx] = self.tensorF_at(idx)
            else:
                rightR = self.tensorR_at(idx + 1)
                currentF = self.tensorF_at(idx)
                self._tensorsR[idx] = numpy.tensordot(currentF, rightR, axes=[[1, 3, 5], [0, 2, 4]]).transpose((0, 3, 1, 4, 2, 5))
        return self._tensorsR[idx]

    def variational_tensor_at(self, idx):
        '''
        compute the variational tensor
        '''
        site = self._mps[idx]
        operator = self._mpo[idx]
        # step 1
        result = numpy.tensordot(self.tensorL_at(idx-1), operator, axes=[3, 0])
        # step 2
        result = numpy.tensordot(result, self.tensorR_at(idx+1), axes=[5, 2])
        # reshape
        result = result.reshape((site.left, site.left, site.phys, site.phys, site.right, site.right))
        # reorder the indices
        return result.transpose((0, 2, 4, 1, 3, 5))

    def sweep_at(self, idx, direction):
        '''
        DMRG sweep
        '''
        assert direction == "left" or direction == "right"
        site = self._mps[idx]
        localDimension = site.left * site.phys * site.right
        # reshape the variational tensor to a square matrix
        variationalTensor = self.variational_tensor_at(idx).reshape((localDimension, localDimension))
        # solve for eigen values and vectors
        eigVal, eigVec = numpy.linalg.eigh(variationalTensor)
        # update current site
        self.update_local_site(idx, newState=eigVec[:,0].reshape(site.shape))
        # normalization
        if direction == "left":
            self.left_canonicalize_at(idx)
        else: # direction == "right"
            self.right_canonicalize_at(idx)
        # return the ground state energy
        return eigVal[0]

    # the head dummy site has its idx as 0
    # the real sites have indices in the range 1 ~ size
    # the last dummy site has its idx as size+1
    def run(self):
        '''
        run DMRG calculation and search for the ground state.
        '''
        sweepCount = 0
        sweepEnergy = 0.0
        print("********* naive DMRG for spin model *********")
        while sweepCount < 2 or not numpy.isclose(self._iterationEnergies[-1], self._iterationEnergies[-2]):
            sweepCount += 1
            print('\n*************** sweep: %d ***************' % sweepCount)
            print(">>>>>>>>>> sweep from left to right >>>>>>>>>>")
            # left -> right sweep
            for idx in range(1, self._size + 1):
                sweepEnergy = self.sweep_at(idx, direction="left")
                print("site: %2d, energy: %.12f" %(idx-1, sweepEnergy))
            # right -> left sweep
            print("<<<<<<<<<< sweep from right to left <<<<<<<<<<")
            for idx in range(self._size, 0, -1):
                sweepEnergy = self.sweep_at(idx, direction="right")
                print("site: %2d, energy: %.12f" %(idx-1, sweepEnergy))
            self._iterationEnergies.append(sweepEnergy)
        print("\n********** task finished in %d sweep iterations **********" % sweepCount)
        self._gsEnergy = self._iterationEnergies[-1]

    @property
    def energy(self):
        '''
        return the ground state energy
        '''
        return self._gsEnergy
