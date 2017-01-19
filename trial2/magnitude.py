import nengo
import nengo.spa as spa
import numpy as np
import random
import pytry


class StateArray(nengo.Network):
    def __init__(self, dimensions, n_states,
                 vocab=None,
                 channel_clarity=2,
                 fwd_synapse=0.01, feedback_synapse=0.1):
        super(StateArray, self).__init__()
        with self:
            self.input = nengo.Node(None, size_in=dimensions)
            self.mask = nengo.Node(None, size_in=n_states)
            self.reset = nengo.Node(None, size_in=1)
            self.channels = []
            self.states = []
            self.outputs = []
            for i in range(n_states):
                c = spa.State(dimensions, vocab=vocab)
                s = spa.State(dimensions, feedback=1, vocab=vocab,
                              feedback_synapse=feedback_synapse)
                nengo.Connection(self.input, c.input, synapse=None)
                nengo.Connection(c.output, s.input, synapse=fwd_synapse)
                self.channels.append(c)
                self.states.append(s)
                self.outputs.append(s.output)
                for ens in c.all_ensembles:
                    nengo.Connection(self.mask[i], ens.neurons,
                        transform=-channel_clarity*np.ones((ens.n_neurons, 1)),
                        synapse=0.01)
                for ens in s.all_ensembles:
                    nengo.Connection(self.reset, ens.neurons,
                            transform=-2*np.ones((ens.n_neurons, 1)),
                                     synapse=0.01)





class MagStim(object):
    def __init__(self, vocab, T_reset, T_stim, T_answer, n_states):
        self.vocab = vocab
        self.T_reset = T_reset
        self.T_stim = T_stim
        self.T_answer = T_answer
        self.T_total = T_reset + T_stim + T_answer
        self.n_states = n_states
        self.stims = []
        for i in range(1, 10):
            for j in range(1, 10):
                if i != j:
                    answer = 1 if i<j else -1
                    self.stims.append((i, j, answer))
        random.shuffle(self.stims)

    def reset(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total
        return 1 if t < self.T_reset else 0

    def value(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        if self.T_reset < t < self.T_reset + self.T_stim:
            i, j, answer = self.stims[index]
            t = t - self.T_reset
            if t < self.T_stim / 2:
                return self.vocab.parse('N%d' % i).v
            else:
                return self.vocab.parse('N%d' % j).v
        else:
            return self.vocab.parse('0').v

    def mask(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        v = np.ones(self.n_states)
        if self.T_reset < t < self.T_reset + self.T_stim:
            i, j, answer = self.stims[index]
            t = t - self.T_reset
            if t < self.T_stim / 2:
                v[0] = 0
            else:
                v[1] = 0
        return v

    def answer(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        if self.T_reset + self.T_stim < t:
            i, j, answer = self.stims[index]
            return answer
        else:
            return 0



class MagTrial(pytry.NengoTrial):
    def params(self):
        self.param('number of dimensions', D=8)
        self.param('number of states', n_states=2)
        self.param('neurons for compare', n_compare=200)
        self.param('time to reset', T_reset=0.3)
        self.param('time to present stimuli', T_stim=0.3)
        self.param('time to read answer', T_answer=0.3)
        self.param('channel clarity', channel_clarity=1.0)

    def model(self, p):
        vocab = spa.Vocabulary(p.D)
        blend = np.linspace(0, 1, 9)
        N1 = vocab.parse('N1')
        N9 = vocab.parse('N9')
        for i in range(1, 8):
            n = N1*(1-blend[i]) + N9*(blend[i])
            n.normalize()
            vocab.add('N%d' % (i+1), n.v)

        self.stim = MagStim(vocab, p.T_reset, p.T_stim, p.T_answer, p.n_states)

        model = nengo.Network()
        with model:
            compare = nengo.Ensemble(n_neurons=p.n_compare, dimensions=p.D*2)
            pts = []
            answers = []
            for i in range(1, 10):
                for j in range(1, 10):
                    if i != j:
                        pt = np.hstack([vocab.parse('N%d'%i).v,vocab.parse('N%d'%j).v])
                        answers.append([1] if i < j else [-1])
                        pts.append(pt)

            answer = nengo.Ensemble(n_neurons=100, dimensions=1)
            nengo.Connection(compare, answer, eval_points=pts, function=answers)

            array = StateArray(p.D, p.n_states, vocab=vocab,
                               channel_clarity=p.channel_clarity)

            nengo.Connection(array.outputs[0], compare[:p.D])
            nengo.Connection(array.outputs[1], compare[p.D:])

            reset = nengo.Node(self.stim.reset)
            nengo.Connection(reset, array.reset, synapse=None)

            mask = nengo.Node(self.stim.mask)
            nengo.Connection(mask, array.mask, synapse=None)

            value = nengo.Node(self.stim.value)
            nengo.Connection(value, array.input, synapse=None)

            correct = nengo.Node(self.stim.answer)

            self.p_answer = nengo.Probe(answer, synapse=0.01)
        self.locals = locals()
        return model

    def evaluate(self, p, sim, plt):
        T = self.stim.T_total * len(self.stim.stims)
        sim.run(T)

        answer = sim.data[self.p_answer]

        results = []
        correct_delta = [[] for i in range(8)]

        for index in range(len(self.stim.stims)):
            i, j, correct = self.stim.stims[index]
            t_index = int(self.stim.T_total*(index+1)/p.dt)-1
            a = answer[t_index][0]
            results.append((i, j, a))
            delta = abs(j-i) - 1
            if (i<j and a>0) or (i>j and a<0):
                correct_delta[delta].append(1)
            else:
                correct_delta[delta].append(0)

        prob_correct = [np.mean(np.array(d).astype(float)) for d in correct_delta]
        return dict(
            correct=prob_correct,
            results=results
            )









if __name__ == '__builtin__':
    trial = MagTrial()
    model = trial.make_model()
    for k, v in trial.locals.items():
        globals()[k] = v
    del v
