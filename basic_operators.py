#! /usr/bin/env python
# filename: basic_operators.py

# Basic operators for spin model
# Pauli matrices

# basis: { up, down }

import numpy

# S^+, the aising operator
Sp = numpy.float64([[0, 1],
                    [0, 0]])

# S^-, the lowering operator
Sm = numpy.float64([[0, 0],
                    [1, 0]])

# Sx measurement
Sx = numpy.float64([[0,   0.5],
                    [0.5, 0  ]])

# Sz measurement
Sz = numpy.float64([[0.5,  0   ],
                    [0,   -0.5]])

# zero matrix block
zero = numpy.zeros((2, 2))

# identity matrix block
identity = numpy.eye((2))
