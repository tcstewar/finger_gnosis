import nengo
import nengo.spa as spa
import numpy as np
import nengo_plot
import magnitude

np.random.seed(seed=1)

vocab = spa.Vocabulary(8)

t_step = 0.2

model = nengo.Network(seed=1)
with model:
    array = magnitude.StateArray(dimensions=8, n_states=2, vocab=vocab,
                                 inh_synapse=0.005,
                                 channel_clarity=0.8)


    p_c0 = nengo.Probe(array.channels[0].all_ensembles[0].neurons)
    p_c1 = nengo.Probe(array.channels[1].all_ensembles[0].neurons)
    p_p0 = nengo.Probe(array.states[0].all_ensembles[0].neurons)
    p_p1 = nengo.Probe(array.states[1].all_ensembles[0].neurons)

    def mask(t):
        index = int(t/t_step)
        if index == 1:
            return [0,1]
        elif index == 2:
            return [1,0]
        else:
            return [1,1]
    stim_mask = nengo.Node(mask)
    nengo.Connection(stim_mask, array.mask, synapse=None)

    def input(t):
        index = int(t/t_step)
        if index == 1:
            return vocab.parse('FIVE').v
        elif index == 2:
            return vocab.parse('SEVEN').v
        else:
            return vocab.parse('0').v
    stim_input = nengo.Node(input)
    nengo.Connection(stim_input, array.input, synapse=None)

    p_mask = nengo.Probe(stim_mask, synapse=None)


sim = nengo.Simulator(model)
sim.run(t_step*5)

plot = nengo_plot.Time(sim.trange(), border_left=1.0, dpi=100)
ax = plot.add('input $x$\n\nmask',
              1-sim.data[p_mask],
              range=(0,5),
              overlays=[(0.3,'FIVE'), (0.5,'SEVEN')])
ax.set_yticks([])
ax.set_ylabel('input $x$\n\nmask', rotation='horizontal', ha='right', va='center')

ax = plot.add_spikes('channel 1', sim.data[p_c0], sample_by_variance=200, cluster=True, merge=25, overlays=[(0.3,'FIVE')])
ax.set_ylabel('channel 1', rotation='horizontal', ha='right', va='center')
ax = plot.add_spikes('channel 2', sim.data[p_c1], sample_by_variance=200, cluster=True, merge=25, overlays=[(0.5, 'SEVEN')])
ax.set_ylabel('channel 2', rotation='horizontal', ha='right', va='center')

ax = plot.add_spikes('pointer 1', sim.data[p_p0], sample_by_variance=200, cluster=True, merge=25, overlays=[(0.3, 'FIVE')])
ax.set_ylabel('pointer 1', rotation='horizontal', ha='right', va='center')
ax = plot.add_spikes('pointer 2', sim.data[p_p1], sample_by_variance=200, cluster=True, merge=25, overlays=[(0.5, 'SEVEN')])
ax.set_ylabel('pointer 2', rotation='horizontal', ha='right', va='center')

plot.save('spikes.png', dpi=600)
plot.show()
