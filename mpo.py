#! /usr/bin/env python
# filename: mpo.py

import numpy

class MPO:
    '''
    ***** matrix product operator class for spin model ***
    This MPO class is designed for 1D-Ising model and Heisenberg model.
    For such simple spin models, all the sites except for the first and last ones
    share the same local MPO tensor.
    $$
        O_k = operatorMatrix, k != 1 and k != L
        O_1 = operatorMatrix.row(-1), the last row of the operatorMatrix
        O_L = operatorMatrix.col(1), the first column of the operatorMatrix
    $$
    The bond order of the local operator is [left, right, up, down], as
                                    | 2
                                0 --#-- 1
                                    | 3
    '''
    def __init__(self, localOperator, size, averaged=False):
        '''
        localOperator must be a numpy.ndarray object with 4 dimensions,
        in which shape[2] and shape[3] are physical dimensions.
        '''
        # check the input localOperator
        assert isinstance(localOperator, numpy.ndarray)
        assert localOperator.ndim == 4
        assert localOperator.shape[0] == localOperator.shape[1]
        assert localOperator.shape[2] == localOperator.shape[3]
        # the number of sites
        self._size = size
        # collection of mpo for each site
        self._data = []
        # mpo for the first site
        leftmost = localOperator[-1].copy()
        leftmost = leftmost.reshape((1,) + leftmost.shape)
        self._data.append(leftmost)
        # mpo for middle sites
        for _ in range(self._size - 2):
            self._data.append(localOperator.copy())
        # mpo for the last site
        rightmost = localOperator[:,0].copy()
        if averaged:
            rightmost /= size
        rightmost = rightmost.reshape((rightmost.shape[0],) + (1,) + rightmost.shape[1:])
        self._data.append(rightmost)

    # return the number of sites
    @property
    def size(self):
        '''
        return the number of sites
        '''
        return self._size

    # return the physical dimension at the given site
    def physAt(self, pos):
        return self._data[pos].shape[2]

    @property
    def data(self):
        return self._data
