# source activate theano
# ipython
# from notebook.auth import passwd
# passwd()

# paste the password in config file in the password field

# cd ~/
# openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mykey.key -out mycert.pem

# jupyter notebook

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano import shared

X = T.fmatrix('X')

srng = RandomStreams()
A = srng.normal((10,10))
b = srng.normal((10,1))

sharedA = shared(np.zeros(shape=(10,10), dtype=np.float32))
sharedb = shared(np.zeros(shape=(10,1), dtype=np.float32))

Z = T.dot((X + A), b)
f = function([X], Z, updates=[(sharedA, sharedA + A), (sharedb, sharedb + b)])

x = np.ndarray(shape=(10,10), dtype=np.float32)

p = f(x) 
q = np.dot((x + sharedA.get_value()), sharedb.get_value()) 
p - q
