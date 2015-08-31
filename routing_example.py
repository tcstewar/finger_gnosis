import nengo
import numpy as np

N = 3

model = nengo.Network()
with model:
    
    pointers = nengo.Network()
    with pointers:
        for i in range(N):
            p = nengo.Ensemble(n_neurons=100, dimensions=1, label='%d' % (i+1))


    stim = nengo.Node([0])
    
    targets = nengo.Node([0]*N)
    
    route = nengo.Ensemble(n_neurons=1000, dimensions=N + 1, radius=np.sqrt(N))
    
    nengo.Connection(stim, route[0])
    nengo.Connection(targets, route[1:])
    
    
    for i in range(N):
        def func(x, i=i):
            if x[i+1] > 0.5:
                return x[0]
            return 0
        nengo.Connection(route, pointers.ensembles[i], function=func)
    