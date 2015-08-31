import nengo

N = 3

model = nengo.Network()
with model:
    
    pointers = nengo.Network()
    with pointers:
        for i in range(N):
            p = nengo.Ensemble(n_neurons=100, dimensions=1, label='%d' % (i+1))
    
    stim = nengo.Node([0] * N)
    for i in range(N):
        nengo.Connection(stim[i], pointers.ensembles[i])
