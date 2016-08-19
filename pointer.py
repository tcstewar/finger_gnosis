import numpy as np
import nengo
import ctn_benchmark

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
    def display(self, t):
        index = int(t / self.trial_time)
        t = t % self.trial_time
        a, b = self.pairs[index % len(self.pairs)]
        if 0.1<t<0.2:
            self.display.im_func._nengo_html_ = '<h1>%d</h1>' % a
            return
        if 0.3<t<0.4: 
            self.display.im_func._nengo_html_ = '<h1>%d</h1>' % b
            return
        self.display.im_func._nengo_html_ = ''
    def input1(self, t):
        index = int(t / self.trial_time)
        t = t % self.trial_time
        a, b = self.pairs[index % len(self.pairs)]
        if 0.1<t<0.2:
            return [a * 0.1 - 1]
        if 0.3<t<0.4: 
            return [b * 0.1 - 1]
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
                r[i]=1
        return r
    def display(self, t):
        self.display.im_func._nengo_html_ = ''
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



class FingerGnosis(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of input areas', input_count=2)
        self.default('neurons for input', N_input=200)
        self.default('neurons per pointer', N_pointer=400)
        self.default('neurons per decoded reference', N_reference=1000)
        self.default('neurons for memory', N_memory=2000)
        self.default('neurons for comparison', N_compare=400)
        self.default('neurons for reporting', N_report=100)
        self.default('number of pointers', pointer_count=3)
        self.default('memory synapse time', memory_synapse=0.1)
        self.default('clear memory time', time_clear_mem=0.1)
        self.default('crosstalk', crosstalk=0.2)
        self.default('task', task='compare')
        self.default('evidence scale', evidence_scale=1.0)

    def model(self, p):
        model = nengo.Network()

        if p.task == 'compare':
            self.exp = NumberExperiment(p=p)
        elif p.task == 'fingers':
            self.exp = FingerTouchExperiment(p=p)

        with model:
            input0 = nengo.Node(self.exp.input0)
            input1 = nengo.Node(self.exp.input1)
            if hasattr(self.exp, 'display'):
                display = nengo.Node(self.exp.display)
            pointer_source = nengo.Node(self.exp.pointer_source)
            pointer_target = nengo.Node(self.exp.pointer_target)
            report_finger = nengo.Node(self.exp.report_finger)
            report_compare = nengo.Node(self.exp.report_compare)
            memory_clear = nengo.Node(self.exp.memory_clear)


            # create neural models for the two input areas
            #  (fingers and magnitude)
            area0 = nengo.Ensemble(p.N_input*p.pointer_count, p.pointer_count,
                                   radius=np.sqrt(p.pointer_count),
                                   label='area0')
            area1 = nengo.Ensemble(p.N_input, 1, label='area1')

            nengo.Connection(input0, area0)
            nengo.Connection(input1, area1)


            # define the connections to create the pointers
            def matrix(n,m,pre=None,post=None,value=1):
                m=[[0]*n for i in range(m)]
                if pre is None: pre=range(n)
                if post is None: post=range(m)
                for i in range(max(len(pre),len(post))):
                    m[post[i%len(post)]][pre[i%len(pre)]]=value
                return m

            pointers = nengo.Network(label='pointers')
            with pointers:
                for i in range(p.pointer_count):
                    nengo.Ensemble(p.N_pointer,
                                   dimensions = p.input_count*2+1,
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
                nengo.Connection(area0, pointer,
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
            reference=nengo.Ensemble(p.N_reference,p.pointer_count,
                                     radius=np.sqrt(p.pointer_count),
                                     encoders=nengo.dists.Choice(basis),
                                     intercepts=nengo.dists.Uniform(0.1,0.9),
                                     label='reference')
            for i in range(p.pointer_count):
                matrix=[p.crosstalk]*p.pointer_count
                matrix[i]=1.0-p.crosstalk
                pointer = pointers.ensembles[i]
                nengo.Connection(pointer, reference,
                                 function=ref_func,
                                 transform=[[x] for x in matrix])


            # add a memory to integrate the value referenced by the pointers
            memory = nengo.networks.EnsembleArray(p.N_memory, p.pointer_count,
                                                  radius=1, label='memory')
            nengo.Connection(reference,memory.input,transform=1)
            nengo.Connection(memory.output, memory.input,
                             transform=1, synapse=p.memory_synapse)



            # create a system to report which fingers were pressed
            report = nengo.networks.EnsembleArray(p.N_report,p.pointer_count,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.3,0.9),
                radius=0.3, label='report')
            nengo.Connection(memory.output,report.input,transform=1)
            m=[[-10]*p.pointer_count for i in range(p.pointer_count)]
            for i in range(p.pointer_count):
                m[i][i]=0
            nengo.Connection(report.output, report.input,
                             transform=m, synapse=0.01)
            reported = nengo.networks.EnsembleArray(p.N_report, p.pointer_count,
                radius=1, encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.05,0.9),
                label='reported')
            nengo.Connection(report.output, reported.input,
                             transform=1, synapse=0.2)
            nengo.Connection(reported.output, report.input, transform=-1)
            nengo.Connection(reported.output, reported.input, transform=1.2)


            # create a system to report whether the first
            # or second number is bigger
            compare = nengo.Ensemble(p.N_compare,1, label='compare', radius=1)
            nengo.Connection(memory.ensembles[0],compare[0],transform=p.evidence_scale)
            nengo.Connection(memory.ensembles[1],compare[0],transform=-p.evidence_scale)


            # create inhibitory gates to control the two reporting systems
            report_gate_f = nengo.Ensemble(50,1,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.1,0.9),
                label='report gate f')
            report_gate_c=nengo.Ensemble(50,1,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.1,0.9),
                label='report gate c')
            nengo.Connection(report_finger,report_gate_f,transform=-10)
            nengo.Connection(report_compare,report_gate_c,transform=-10)
            report_bias=nengo.Node([1], label='bias')
            nengo.Connection(report_bias,report_gate_f)
            nengo.Connection(report_bias,report_gate_c)

            nengo.Connection(report_gate_c, compare.neurons,
                             transform=[[-100.0]]*p.N_compare, synapse=0.01)
            for i in range(p.pointer_count):
                nengo.Connection(report_gate_f, report.ensembles[i].neurons,
                                 transform=[[-100.0]]*p.N_report, synapse=0.01)
                nengo.Connection(report_gate_f, reported.ensembles[i].neurons,
                                 transform=[[-100.0]]*p.N_report, synapse=0.01)

            for ens in memory.all_ensembles + [compare]:
                nengo.Connection(memory_clear, ens.neurons,
                                 transform=[[-10]] * ens.n_neurons,
                                 synapse=0.01)

            self.p_report = nengo.Probe(report.output, synapse=0.01)
            self.p_compare = nengo.Probe(compare, synapse=0.01)
            self.p_memory = nengo.Probe(memory.output, synapse=0.01)

        if p.backend == 'nengo_spinnaker':
            import nengo_spinnaker
            nengo_spinnaker.add_spinnaker_params(model.config)
            for node in model.all_nodes:
                if callable(node.output):
                    if not hasattr(node.output, '_nengo_html_'):
                        model.config[node].function_of_time = True
        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.exp.T)
        self.record_speed(self.exp.T)

        t = sim.trange()

        if p.task == 'fingers':
            scores = np.zeros(p.pointer_count-1, dtype=float)
            count = np.zeros(p.pointer_count-1)
            magnitudes = None

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
                    scores[delta - 1] += 1

            scores = scores / count
        else:
            scores = np.zeros(8, dtype=float)
            count = np.zeros(8)
            mags = np.zeros(8, dtype=float)
            magnitudes = []

            for i in range(len(self.exp.pairs)):
                t_start = i * self.exp.trial_time
                t_end = (i+1) * self.exp.trial_time
                index_start = np.argmax(t > t_start)
                index_end = np.argmax(t > t_end)
                if t_end >= t[-1]:
                    index_end = len(t)
                data = sim.data[self.p_compare][index_start:index_end]
                answer = np.mean(data)
                answer_value = np.max(data) if answer > 0 else np.min(data)

                c = self.exp.pairs[i]


                delta = abs(c[0] - c[1])
                count[delta - 1] += 1
                if (answer < 0 and c[0] < c[1]) or (answer > 0 and c[1] < c[0]):
                    scores[delta - 1] += 1
                    mags[delta - 1] += np.abs(answer_value)
                    magnitudes.append((c[0], c[1], answer_value))
                
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
        return result


if __name__ == '__main__':
    FingerGnosis().run()
else:
    model = FingerGnosis().make_model(task='compare', crosstalk=0, time_clear_mem=0.4)
    
