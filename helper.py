import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

X = T.fmatrix('X')

srng = RandomStreams(seed=234)
mat = srng.normal((10,10))
vec = srng.normal((10,1))

a_chng = function([], mat)
a_fix = function([], mat, no_default_updates=True)

b_chng = function([], vec)
b_fix = function([], vec, no_default_updates=True)

A = a_fix()
b = b_fix()

Z = (X + A) * b

f = function([X], Z)

x = np.ndarray(shape=(10,10), dtype=np.float32)

f(x)
