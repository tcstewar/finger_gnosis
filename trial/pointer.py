import numpy as np
import nengo
import pytry

# define the inputs when doing number comparison task
class NumberExperiment:
    def __init__(self, p):
        self.p = p
        self.pairs = []
        self.order = []
        rng = np.random.RandomState(seed=p.seed)
        for i in range(1, 10):
            for j in range(i + 1, 10):
                order = rng.choice([-1, 1])
                self.order.append(order)
                if order < 0:
                    self.pairs.append((i, j))
                else:
                    self.pairs.append((j, i))
        rng.shuffle(self.pairs)

        #self.pairs = self.pairs[:3]

        self.trial_time = 1.0
        self.T = len(self.pairs) * self.trial_time
    def input0(self, t):
        return [0]*self.p.pointer_count
    def input1(self, t):
        index = int(t / self.trial_time)
        t = t % self.trial_time
        a, b = self.pairs[index % len(self.pairs)]
        if 0.1<t<0.2: return [a * self.p.num_slope + self.p.num_bias]
        if 0.3<t<0.4: return [b * self.p.num_slope + self.p.num_bias]
        return [0]
    def pointer_source(self, t):
        return [0, 1]
    def pointer_target(self, t):
        t = t % self.trial_time
        v = [0]*self.p.pointer_count
        if 0.1<t<0.2: v[0]=1
        if 0.3<t<0.4: v[1]=1
        return v
    def report_finger(self, t):
        return [0]
    def report_compare(self, t):
        t = t % self.trial_time
        if 0.5<t<self.trial_time:
            return [1]
        else:
            return [0]
    def memory_clear(self, t):
        t = t % self.trial_time
        if 1.0 - self.p.time_clear_mem < t < 1.0:
            return [1]
        else:
            return [0]

# define the inputs when doing the finger touching task
class FingerTouchExperiment:
    def __init__(self, p):
        self.p = p
        self.pairs = []
        rng = np.random.RandomState(seed=p.seed)
        for i in range(self.p.pointer_count):
            for j in range(i + 1, self.p.pointer_count):
                self.pairs.append((i, j))
        rng.shuffle(self.pairs)

        self.trial_time = 1.0
        self.T = len(self.pairs) * self.trial_time

    def input0(self, t):
        r=[0]*self.p.pointer_count
        index = int(t / self.trial_time)
        t = t % self.trial_time
        if 0.1<t<0.2:
            for i in self.pairs[index]:
                r[i]=self.p.touch_strength
        return r
    def input1(self, t):
        return [0]
    def pointer_source(self, t):
        return [1,0]
    def pointer_target(self, t):
        return [1]*self.p.pointer_count
    def report_finger(self, t):
        t = t % self.trial_time
        if 0.3<t<1.0:
            return [1]
        else:
            return [0]
    def report_compare(self, t):
        return [0]
    def memory_clear(self, t):
        t = t % self.trial_time
        if 1.0 - self.p.time_clear_mem < t < 1.0:
            return [1]
        else:
            return [0]



class FingerGnosis(pytry.NengoTrial):
    def params(self):
        self.param('number of input areas', input_count=2)
        self.param('neurons for input', N_input=200)
        self.param('neurons per pointer', N_pointer=400)
        self.param('neurons per decoded reference', N_reference=1000)
        self.param('neurons for memory', N_memory=200)
        self.param('neurons for comparison', N_compare=400)
        self.param('neurons for reporting', N_report=100)
        self.param('number of pointers', pointer_count=3)
        self.param('memory synapse time', memory_synapse=0.1)
        self.param('clear memory time', time_clear_mem=0.2)
        self.param('crosstalk', crosstalk=0.2)
        self.param('task', task='fingers')
        self.param('magnitude of touch signal', touch_strength=1.0)
        self.param('crosstalk decay', crosstalk_decay=0.0)
        self.param('scaling on numerical values', num_slope=0.1)
        self.param('scaling on numerical values', num_bias=0.0)
        self.param('memory input scale', memory_input_scale=1.0)
        self.param('direct mode', direct=False)
        self.param('compare scale', compare_scale=50.0)
        self.param('probe synapse', probe_synapse=0.01)
        self.param('reference radius', ref_radius=1.0)

    def model(self, p):
        model = nengo.Network()
        if p.direct:
            model.config[nengo.Ensemble].neuron_type=nengo.Direct()

        if p.task == 'compare':
            self.exp = NumberExperiment(p=p)
        elif p.task == 'fingers':
            self.exp = FingerTouchExperiment(p=p)

        with model:
            input0 = nengo.Node(self.exp.input0)
            input1 = nengo.Node(self.exp.input1)
            pointer_source = nengo.Node(self.exp.pointer_source)
            pointer_target = nengo.Node(self.exp.pointer_target)
            report_finger = nengo.Node(self.exp.report_finger)
            report_compare = nengo.Node(self.exp.report_compare)
            memory_clear = nengo.Node(self.exp.memory_clear)


            # create neural models for the two input areas
            #  (fingers and magnitude)
            #area0 = nengo.Ensemble(p.N_input*p.pointer_count, p.pointer_count,
            #                       radius=np.sqrt(p.pointer_count))
            area0 = nengo.networks.EnsembleArray(p.N_input, n_ensembles=p.pointer_count)
            area1 = nengo.Ensemble(p.N_input, 1)

            nengo.Connection(input0, area0.input)
            nengo.Connection(input1, area1)


            # define the connections to create the pointers
            def matrix(n,m,pre=None,post=None,value=1):
                m=[[0]*n for i in range(m)]
                if pre is None: pre=range(n)
                if post is None: post=range(m)
                for i in range(max(len(pre),len(post))):
                    m[post[i%len(post)]][pre[i%len(pre)]]=value
                return m

            pointers = nengo.Network()
            with pointers:
                for i in range(p.pointer_count):
                    nengo.Ensemble(p.N_pointer,
                                   dimensions = p.input_count*2+1,
                                   neuron_type=nengo.LIF(),
                                   radius = np.sqrt(p.input_count*2+1),
                                   label='%d' % i)
            for i in range(p.pointer_count):
                pointer = pointers.ensembles[i]
                nengo.Connection(pointer_source, pointer,
                        transform=matrix(
                            p.input_count, p.input_count*2+1,
                            post=[k*2 for k in range(p.input_count)]))

                nengo.Connection(pointer_target,pointer,
                                transform=matrix(p.pointer_count,
                                                 p.input_count*2+1,
                                                 pre=[i],
                                                 post=[p.input_count*2]))
                nengo.Connection(area0.output, pointer,
                                 transform=matrix(p.pointer_count,
                                                  p.input_count*2+1,
                                                  pre=[i],post=[1]))
                nengo.Connection(area1, pointer,
                                 transform=matrix(1,p.input_count*2+1,
                                                  pre=[0],post=[3]))



            # define the connections to extract the current value
            #  from the pointers
            def ref_func(x):
                if x[-1]<0.5: return 0
                sum=0
                for i in range(p.input_count):
                    if x[2*i]>0.5: sum+=x[2*i+1]
                return sum
            basis=[]
            for i in range(p.pointer_count):
                b=[0]*p.pointer_count
                b[i]=1
                basis.append(b)
                b=[0]*p.pointer_count
                b[i]=-1
                basis.append(b)
            #reference=nengo.Ensemble(p.N_reference,p.pointer_count,
            #                         radius=np.sqrt(p.pointer_count)*p.ref_radius,
            #                         neuron_type=nengo.LIF(),
            #                         encoders=nengo.dists.Choice(basis),
            #                         intercepts=nengo.dists.Uniform(0.1,0.9))
            reference = nengo.networks.EnsembleArray(p.N_reference,
                            n_ensembles = p.pointer_count,
                            radius=p.ref_radius,
                            neuron_type=nengo.LIF(),
                            encoders=nengo.dists.Choice([[1]]),
                            intercepts=nengo.dists.Uniform(0.1, 0.9))

            for i in range(p.pointer_count):
                matrix=[p.crosstalk]*p.pointer_count
                matrix[i]=1.0-p.crosstalk
                for j in range(p.pointer_count):
                    delta = np.abs(i-j)
                    if delta > 1:
                        matrix[j] = max(0, p.crosstalk - p.crosstalk_decay * (delta-1))
                pointer = pointers.ensembles[i]
                nengo.Connection(pointer, reference.input,
                                 function=ref_func,
                                 transform=[[x] for x in matrix])

            # add a memory to integrate the value referenced by the pointers
            memory = nengo.networks.EnsembleArray(p.N_memory, p.pointer_count,
                                                  neuron_type=nengo.LIF(),
                                                  radius=1)
            nengo.Connection(reference.output,memory.input,transform=p.memory_input_scale, synapse=p.memory_synapse)
            nengo.Connection(memory.output, memory.input,
                             transform=1, synapse=p.memory_synapse)



            # create a system to report which fingers were pressed
            report = nengo.networks.EnsembleArray(p.N_report,p.pointer_count,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.3,0.9),
                neuron_type=nengo.LIF(),
                radius=0.3)
            nengo.Connection(memory.output,report.input,transform=1)
            m=[[-10]*p.pointer_count for i in range(p.pointer_count)]
            for i in range(p.pointer_count):
                m[i][i]=0
            nengo.Connection(report.output, report.input,
                             transform=m, synapse=0.01)
            reported = nengo.networks.EnsembleArray(p.N_report, p.pointer_count,
                radius=1, encoders=nengo.dists.Choice([[1]]),
                neuron_type=nengo.LIF(),
                intercepts=nengo.dists.Uniform(0.05,0.9))
            nengo.Connection(report.output, reported.input,
                             transform=1, synapse=0.2)
            nengo.Connection(reported.output, report.input, transform=-1)
            nengo.Connection(reported.output, reported.input, transform=1.2)


            # create a system to report whether the first
            # or second number is bigger
            compare = nengo.Ensemble(p.N_compare,1, neuron_type=nengo.LIF())
            nengo.Connection(memory.ensembles[0],compare,transform=p.compare_scale)
            nengo.Connection(memory.ensembles[1],compare,transform=-p.compare_scale)

            # create inhibitory gates to control the two reporting systems
            report_gate_f = nengo.Ensemble(50,1,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.1,0.9))
            report_gate_c=nengo.Ensemble(50,1,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.1,0.9))
            nengo.Connection(report_finger,report_gate_f,transform=-1)
            nengo.Connection(report_compare,report_gate_c,transform=-1)
            report_bias=nengo.Node([1])
            nengo.Connection(report_bias,report_gate_f)
            nengo.Connection(report_bias,report_gate_c)

            nengo.Connection(report_gate_c, compare.neurons,
                             transform=[[-100.0]]*p.N_compare, synapse=0.01)
            for i in range(p.pointer_count):
                nengo.Connection(report_gate_f, report.ensembles[i].neurons,
                                 transform=[[-100.0]]*p.N_report, synapse=0.01)
                nengo.Connection(report_gate_f, reported.ensembles[i].neurons,
                                 transform=[[-100.0]]*p.N_report, synapse=0.01)

            for ens in memory.all_ensembles:
                nengo.Connection(memory_clear, ens.neurons,
                                 transform=[[-10]] * ens.n_neurons,
                                 synapse=0.01)

            self.p_report = nengo.Probe(report.output, synapse=p.probe_synapse)
            self.p_compare = nengo.Probe(compare, synapse=p.probe_synapse)
            self.p_memory = nengo.Probe(memory.output, synapse=p.probe_synapse)
        self.locals = locals()
        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.exp.T)

        t = sim.trange()

        if p.task == 'fingers':
            scores = np.zeros(p.pointer_count-1, dtype=float)
            count = np.zeros(p.pointer_count-1)
            mags = np.zeros(p.pointer_count-1)
            magnitudes = []
            error_magnitudes = []

            for i in range(len(self.exp.pairs)):
                t_start = i * self.exp.trial_time
                t_end = (i+1) * self.exp.trial_time
                index_start = np.argmax(t > t_start)
                index_end = np.argmax(t > t_end)
                if t_end >= t[-1]:
                    index_end = len(t)
                data = sim.data[self.p_report][index_start:index_end]
                answers = np.max(data, axis=0)
                values = [(v, ii) for ii, v in enumerate(answers)]
                values.sort()
                r = values[-1][1], values[-2][1]
                c = self.exp.pairs[i]

                delta = abs(c[0] - c[1])
                count[delta - 1] += 1
                if (r[0], r[1]) == (c[0], c[1]) or (r[0], r[1]) == (c[1], c[0]):
                    v = (values[-1][0] + values[-2][0])/2
                    magnitudes.append((c[0], c[1], v))
                    mags[delta - 1] += v
                    scores[delta - 1] += 1
                else:
                    error_magnitudes.append((c[0], c[1], v))


            mags = mags / scores
            scores = scores / count
        else:
            scores = np.zeros(8, dtype=float)
            count = np.zeros(8)
            mags = np.zeros(8, dtype=float)
            magnitudes = []
            error_magnitudes = []

            for i in range(len(self.exp.pairs)):
                t_start = i * self.exp.trial_time
                t_end = (i+1) * self.exp.trial_time
                index_start = np.argmax(t > t_start + 0.5)
                index_end = np.argmax(t > t_end)
                if t_end >= t[-1]:
                    index_end = len(t)
                data = sim.data[self.p_compare][index_start:index_end]

                max_a = np.max(data)
                min_a = np.min(data)
                if max_a > -min_a:
                    answer_value = max_a
                else:
                    answer_value = min_a
                answer_value = np.mean(data)

                c = self.exp.pairs[i]


                delta = abs(c[0] - c[1])
                count[delta - 1] += 1
                if (answer_value < 0 and c[0] < c[1]) or (answer_value > 0 and c[1] < c[0]):
                    scores[delta - 1] += 1
                    mags[delta - 1] += np.abs(answer_value)
                    magnitudes.append((c[0], c[1], answer_value))
                else:
                    error_magnitudes.append((c[0], c[1], answer_value))

            mags = mags / scores
            scores = scores / count



        if plt is not None:
            plt.subplot(2,1,1)
            if p.task == 'fingers':
                plt.plot(sim.trange(), sim.data[self.p_report])
            elif p.task == 'compare':
                plt.plot(sim.trange(), sim.data[self.p_compare])
                for i, (a, b) in enumerate(self.exp.pairs):
                    t = self.exp.trial_time * (i + 0.5)
                    colors = ['#000000', '#666666']
                    if a < b:
                        colors = colors[::-1]
                    plt.text(t, 1.7, '%d' % a, color=colors[0])
                    plt.text(t, -1.7, '%d' % b, color=colors[1])

            plt.subplot(2,1,2)
            plt.plot(sim.trange(), sim.data[self.p_memory])

        result = {}
        for i, s in enumerate(scores):
            result['score%d'%(i+1)] = s
        for i, m in enumerate(mags):
            result['mag%d'%(i+1)] = m
        result['magnitudes'] = magnitudes
        result['error_magnitudes'] = error_magnitudes
        return result
