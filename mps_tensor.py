#! /usr/bin/env python
# filename: mps_tensor.py

'''
local matrix product state tensor
bond order: [left, physical, right]
                1
                |
            0 --*-- 2
'''

import numpy

class local_state:
    '''
    ***** local tensor class *****
    '''
    def __init__(self, left, phys, right):
        # rank-3 local tensor
        if left and right: # if this tensor is not the left-/right-most one
            # initialize with random numbers
            self._data = numpy.random.random((left, phys, right))
        else:
            self._data = numpy.zeros((0,0,0))

    # return all the three bond dimensions
    @property
    def dims(self):
        '''
        return bond dimensions of [left, physical, right] bonds
        '''
        return self._data.shape

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        '''
        return the shape of local tensor
        '''
        return self._data.shape

    # return the left bond dimension
    @property
    def left(self):
        return self._data.shape[0]

    # return the physical bond dimension
    @property
    def phys(self):
        return self._data.shape[1]

    # return the right bond dimension
    @property
    def right(self):
        return self._data.shape[2]

    # mps truncation via SVD procedure
    def truncate(self, direction, maxM):
        '''
        MPS truncation via Singular Value Decomposition
        '''
        u = s = vh = None
        if direction == "left": # left -> right SVD
            u, s, vh = numpy.linalg.svd(self._data.reshape(self.left * self.phys, self.right), full_matrices=False)
        elif direction == "right": # right -> left SVD
            u, s, vh = numpy.linalg.svd(self._data.reshape(self.left, self.phys * self.right), full_matrices=False)
        if len(s) > maxM: # do truncation
            return u[:, :maxM], s[:maxM], vh[:maxM, :]
        return u, s, vh

    # update local tensor
    def update(self, tensor):
        self._data = tensor
