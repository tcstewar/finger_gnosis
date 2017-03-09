import nengo
import nengo.spa as spa
import numpy as np
import random
import pytry


class StateArray(nengo.Network):
    def __init__(self, dimensions, n_states,
                 vocab=None,
                 channel_clarity=2,
                 fwd_synapse=0.01, feedback_synapse=0.1,
                 inh_synapse=0.01):
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
                        synapse=inh_synapse)
                for ens in s.all_ensembles:
                    nengo.Connection(self.reset, ens.neurons,
                            transform=-2*np.ones((ens.n_neurons, 1)),
                                     synapse=inh_synapse)





class FingStim(object):
    def __init__(self, vocab, T_reset, T_stim, T_answer, n_states=5,
            max_diff=4, mask_base=1.125):
        self.vocab = vocab
        self.T_reset = T_reset
        self.T_stim = T_stim
        self.T_answer = T_answer
        self.T_total = T_reset + T_stim + T_answer
        self.n_states = n_states
        self.mask_base=mask_base
        self.stims = []
        for i in range(1, 6):
            for j in range(1, 6):
                if i != j and abs(i-j) <= max_diff:
                    self.stims.append((i, j))
        random.shuffle(self.stims)

    def reset(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total
        return 1 if t < self.T_reset else 0

    def value(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        if self.T_reset < t < self.T_reset + self.T_stim:
            return self.vocab.parse('TOUCHED').v
        else:
            return self.vocab.parse('0').v

    def mask(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        v = np.ones(self.n_states)*self.mask_base
        if self.T_reset < t < self.T_reset + self.T_stim:
            i, j = self.stims[index]
            t = t - self.T_reset
            if t < self.T_stim / 2:
                v[i-1] = 0
                if i-2>=0: v[i-2] = 1
                if i<self.n_states: v[i] = 1
            else:
                v[j-1] = 0
                if j-2>=0: v[j-2] = 1
                if j<self.n_states: v[j] = 1
        return v

    def answer(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        v = np.zeros(self.n_states)
        if self.T_reset + self.T_stim < t:
            i, j = self.stims[index]
            v[i-1] = 1-i*0.1
            v[j-1] = 1-j*0.1
        return v

    def no_report(self, t):
        index = int(t / self.T_total) % len(self.stims)
        t = t % self.T_total

        v = np.zeros(self.n_states)
        if self.T_reset + self.T_stim < t:
            return 0
        else:
            return 1


class FingTrial(pytry.NengoTrial):
    def params(self):
        self.param('number of dimensions', D=8)
        self.param('number of states', n_states=5)
        self.param('time to reset', T_reset=0.3)
        self.param('time to present stimuli', T_stim=0.3)
        self.param('time to read answer', T_answer=0.3)
        self.param('channel clarity', channel_clarity=0.875)
        self.param('maximum difference to evaluate', max_diff=4)
        self.param('number of neurons collecting from state', n_collect=1000)
        self.param('number of samples', n_samples=2000)
        self.param('sample noise', sample_noise=0.15)
        self.param('mask max value', mask_base=1.125)

    def model(self, p):
        vocab = spa.Vocabulary(p.D)

        self.stim = FingStim(vocab, p.T_reset, p.T_stim, p.T_answer, p.n_states,
                            max_diff=p.max_diff, mask_base=p.mask_base)

        model = nengo.Network()
        with model:
            array = StateArray(p.D, p.n_states, vocab=vocab,
                               channel_clarity=p.channel_clarity)

            reset = nengo.Node(self.stim.reset)
            nengo.Connection(reset, array.reset, synapse=None)

            mask = nengo.Node(self.stim.mask)
            nengo.Connection(mask, array.mask, synapse=None)

            value = nengo.Node(self.stim.value)
            nengo.Connection(value, array.input, synapse=None)

            correct = nengo.Node(self.stim.answer)


            collect = nengo.Ensemble(n_neurons=p.n_collect,
                                     dimensions=p.n_states)
            readout = nengo.Node(None, size_in=p.n_states)

            pts = []
            target = []
            while len(pts) < p.n_samples:
                i = np.random.choice([1,2,3,4,5])
                j = np.random.choice([1,2,3,4,5])
                if i==j:
                    continue
                v = np.zeros(5, dtype=float)
                v[i-1] = 1
                v[j-1] = 1
                pts.append(v)
                target.append(v)
            target = np.array(target, dtype=float)
            target += np.random.randn(*target.shape)*p.sample_noise
            pts = np.array(pts, dtype=float)
            pts += np.random.randn(*train_pts.shape)*p.sample_noise

            nengo.Connection(collect, readout,
                             eval_points=pts,
                             function=target, scale_eval_points=False)

            for i in range(p.n_states):
                nengo.Connection(array.states[i].output, collect[i],
                                 transform=np.array([vocab.parse('TOUCHED').v/1.5]))

            self.p_answer = nengo.Probe(collect, synapse=0.01)

            '''
            no_report = nengo.Node(self.stim.no_report)

            report = nengo.networks.EnsembleArray(100, p.n_states,
                        encoders=nengo.dists.Choice([[1]]),
                        intercepts=nengo.dists.Uniform(0.6,0.9),
                        radius=1)
            for i in range(p.n_states):
                nengo.Connection(array.states[i].output, report.input[i],
                                    transform=np.array([vocab.parse('TOUCHED').v/1.5]))
            for ens in report.all_ensembles:
                nengo.Connection(no_report, ens.neurons,
                                    transform=-2*np.ones((ens.n_neurons,1)))

            reported = nengo.networks.EnsembleArray(100, p.n_states,
                radius=1, encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.05,0.9))


            m=[[-10]*p.n_states for i in range(p.n_states)]
            for i in range(p.n_states):
                m[i][i]=0
            nengo.Connection(report.output, report.input,
                             transform=m, synapse=0.01)


            nengo.Connection(report.output, reported.input,
                             transform=1, synapse=0.2)
            nengo.Connection(reported.output, report.input, transform=-1)
            nengo.Connection(reported.output, reported.input, transform=1.2)


            for ens in report.all_ensembles + reported.all_ensembles:
                nengo.Connection(reset, ens.neurons, transform=-2*np.ones((ens.n_neurons,1)))

            #self.p_answer = nengo.Probe(answer, synapse=0.01)
            '''
        self.locals = locals()
        return model

    def evaluate(self, p, sim, plt):
        T = self.stim.T_total * len(self.stim.stims)
        sim.run(T)

        answer = sim.data[self.p_answer]

        results = []
        correct_delta = [[] for i in range(4)]

        raw = []

        for index in range(len(self.stim.stims)):
            i, j = self.stim.stims[index]
            d = abs(i - j)
            t_index = int(self.stim.T_total*(index+1)/p.dt)-1
            a = answer[t_index]
            order = np.argsort(a)
            correct = i-1, j-1

            raw.append((i, j, a, order))

            if ((order[-1]==correct[0] and order[-2]==correct[1]) or
                (order[-2]==correct[0] and order[-1]==correct[1])):
                   correct_delta[d-1].append(1)
            else:
                   correct_delta[d-1].append(0)

        prob_correct = [np.mean(np.array(d).astype(float)) for d in correct_delta]
        return dict(
            correct=prob_correct,
            raw=raw,
            results=results
            )









if __name__ == '__builtin__':
    trial = FingTrial()
    model = trial.make_model()
    for k, v in trial.locals.items():
        globals()[k] = v
    del v
