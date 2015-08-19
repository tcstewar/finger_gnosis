import numpy as np
import nengo
import ctn_benchmark

# define the inputs when doing number comparison task
class NumberExperiment:
    def __init__(self, p):
        self.p = p
    def input0(self, t):
        return [0]*self.p.pointer_count
    def input1(self, t):
        if 0.1<t<0.2: return [compare_numbers[0]]
        if 0.3<t<0.4: return [compare_numbers[1]]
        return [0]
    def pointer_source(self, t):
        return [0, 1]
    def pointer_target(self, t):
        v = [0]*self.p.pointer_count
        if 0.1<t<0.2: v[0]=1
        if 0.3<t<0.4: v[1]=1
        return v
    def report_finger(self, t):
        return [0]
    def report_compare(self, t):
        if 0.5<t<1.0:
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
        self.default('task', task='fingers')

    def model(self, p):
        model = nengo.Network()

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
            area0 = nengo.Ensemble(p.N_input*p.pointer_count, p.pointer_count,
                                   radius=np.sqrt(p.pointer_count))
            area1 = nengo.Ensemble(p.N_input, 1)

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

            pointers = nengo.Network()
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
                                     intercepts=nengo.dists.Uniform(0.1,0.9))
            for i in range(p.pointer_count):
                matrix=[p.crosstalk]*p.pointer_count
                matrix[i]=1.0-p.crosstalk
                pointer = pointers.ensembles[i]
                nengo.Connection(pointer, reference,
                                 function=ref_func,
                                 transform=[[x] for x in matrix])


            # add a memory to integrate the value referenced by the pointers
            memory = nengo.networks.EnsembleArray(p.N_memory, p.pointer_count,
                                                  radius=1)
            nengo.Connection(reference,memory.input,transform=1)
            nengo.Connection(memory.output, memory.input,
                             transform=1, synapse=p.memory_synapse)



            # create a system to report which fingers were pressed
            report = nengo.networks.EnsembleArray(p.N_report,p.pointer_count,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.3,0.9),
                radius=0.3)
            nengo.Connection(memory.output,report.input,transform=1)
            m=[[-10]*p.pointer_count for i in range(p.pointer_count)]
            for i in range(p.pointer_count):
                m[i][i]=0
            nengo.Connection(report.output, report.input,
                             transform=m, synapse=0.01)
            reported = nengo.networks.EnsembleArray(p.N_report, p.pointer_count,
                radius=1, encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.05,0.9))
            nengo.Connection(report.output, reported.input,
                             transform=1, synapse=0.2)
            nengo.Connection(reported.output, report.input, transform=-1)
            nengo.Connection(reported.output, reported.input, transform=1.2)


            # create a system to report whether the first
            # or second number is bigger
            compare = nengo.Ensemble(p.N_compare,1)
            nengo.Connection(memory.ensembles[0],compare,transform=50)
            nengo.Connection(memory.ensembles[1],compare,transform=-50)

            # create inhibitory gates to control the two reporting systems
            report_gate_f = nengo.Ensemble(50,1,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.1,0.9))
            report_gate_c=nengo.Ensemble(50,1,
                encoders=nengo.dists.Choice([[1]]),
                intercepts=nengo.dists.Uniform(0.1,0.9))
            nengo.Connection(report_finger,report_gate_f,transform=-10)
            nengo.Connection(report_compare,report_gate_c,transform=-10)
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

            self.p_report = nengo.Probe(report.output, synapse=0.01)
            self.p_memory = nengo.Probe(memory.output, synapse=0.01)
        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.exp.T)
        self.record_speed(self.exp.T)

        if plt is not None:
            plt.subplot(2,1,1)
            plt.plot(sim.trange(), sim.data[self.p_report])
            plt.subplot(2,1,2)
            plt.plot(sim.trange(), sim.data[self.p_memory])

        return {}


if __name__ == '__main__':
    FingerGnosis().run()
