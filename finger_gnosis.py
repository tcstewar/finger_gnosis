input_count = 2     # how many types of inputs there are (must be 2: finger and magnitude)
N_input = 200       # number of neurons to represent the input
N_pointer = 400     # number of neurons per pointer
N_reference = 1000  # number of neurons to store the value of the pointer (dereferenced)
N_memory = 2000     # number of neurons to store the value of the pointer over time
N_compare = 400     # number of neurons to do magnitude comparison
N_report = 100      # number of neurons to read out the value of memory
pointer_count = 3   # number of pointers
crosstalk = 0.2     # amount of interference between pointers (0=no interference)

compare_numbers = 0.8, 0.7  # the two numbers to compare (should be between 0.5 and 1)
fingers = 0, 2              # the fingers to touch

mode='compare'    # 'fingers' for finger-touching experiment, 
                  # 'compare' for magnitude comparison, 
                  # 'manual' if you want to manually control inputs


import nengo
import math
model = nengo.Network()



# define the inputs when doing number comparison task
class NumberExperiment:
    def input0(self, t):
        return [0]*pointer_count
    def input1(self, t):
        if 0.1<t<0.2: return [compare_numbers[0]]
        if 0.3<t<0.4: return [compare_numbers[1]]
        return [0]
    def pointer_source(self, t):
        return [0, 1]    
    def pointer_target(self, t):
        v = [0]*pointer_count
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
    def input0(self, t):
        r=[0]*pointer_count
        if 0.1<t<0.2: 
            for i in fingers:
                if i<pointer_count: r[i]=1
        return r    
    def input1(self, t):
        return [0]
    def pointer_source(self, t):
        return [1,0]    
    def pointer_target(self, t):
        return [1]*pointer_count
    def report_finger(self, t):
        if 0.3<t<1.0:
            return [1]
        else:
            return [0]    
    def report_compare(self, t):
        return [0]
        
class ManualExperiment:
    def input0(self, t):
        return [0] * pointer_count
    def input1(self, t):
        return [0]
    def pointer_source(self, t):
        return [0] * input_count    
    def pointer_target(self, t):
        return [1] * pointer_count
    def report_finger(self, t):
        return [0]
    def report_compare(self, t):
        return [0]
    
if mode == 'manual':
    exp = ManualExperiment()
elif mode == 'compare':
    exp = NumberExperiment()
elif mode == 'fingers':
    exp = FingerTouchExperiment()

with model:
    input0 = nengo.Node(exp.input0)
    input1 = nengo.Node(exp.input1)
    pointer_source = nengo.Node(exp.pointer_source)
    pointer_target = nengo.Node(exp.pointer_target)
    report_finger = nengo.Node(exp.report_finger)
    report_compare = nengo.Node(exp.report_compare)
    

    # create neural models for the two input areas (fingers and magnitude)
    area0 = nengo.Ensemble(N_input*pointer_count, pointer_count, radius=math.sqrt(pointer_count))
    area1 = nengo.Ensemble(N_input, 1)

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
        for i in range(pointer_count):
            p = nengo.Ensemble(N_pointer,
                               dimensions = input_count*2+1,
                               radius = math.sqrt(input_count*2+1),
                               label='%d' % i)
    for i in range(pointer_count):
        pointer = pointers.ensembles[i]
        nengo.Connection(pointer_source, pointer,
                transform=matrix(input_count, input_count*2+1,
                                 post=[k*2 for k in range(input_count)]))
                                 
    	nengo.Connection(pointer_target,pointer,
    	                transform=matrix(pointer_count,input_count*2+1,
    	                                 pre=[i],post=[input_count*2]))
    	nengo.Connection(area0,pointer,transform=matrix(pointer_count,input_count*2+1,pre=[i],post=[1]))
    	nengo.Connection(area1,pointer,transform=matrix(1,input_count*2+1,pre=[0],post=[3]))



    # define the connections to extract the current value from the pointers
    def ref_func(x):
    	if x[-1]<0.5: return 0
    	sum=0
    	for i in range(input_count):
    		if x[2*i]>0.5: sum+=x[2*i+1]
    	return sum	
    basis=[]
    for i in range(pointer_count):
    	b=[0]*pointer_count
    	b[i]=1
    	basis.append(b)
    	b=[0]*pointer_count
    	b[i]=-1
    	basis.append(b)
    reference=nengo.Ensemble(N_reference,pointer_count,
                             radius=math.sqrt(pointer_count),
                             encoders=nengo.dists.Choice(basis),
                             intercepts=nengo.dists.Uniform(0.1,0.9))
    for i in range(pointer_count):
    	matrix=[crosstalk]*pointer_count
    	matrix[i]=1.0-crosstalk
    	pointer = pointers.ensembles[i]
    	nengo.Connection(pointer,reference,function=ref_func,transform=[[x] for x in matrix])
	
	
    # add a memory to integrate the value referenced by the pointers
    memory = nengo.networks.EnsembleArray(N_memory,pointer_count,radius=1)
    nengo.Connection(reference,memory.input,transform=1)
    nengo.Connection(memory.output,memory.input,transform=1,synapse=0.2)



    # create a system to report which fingers were pressed
    report=nengo.networks.EnsembleArray(N_report,pointer_count,
                                        encoders=nengo.dists.Choice([[1]]),
                                        intercepts=nengo.dists.Uniform(0.3,0.9),
                                        radius=0.3)
    nengo.Connection(memory.output,report.input,transform=1)
    m=[[-10]*pointer_count for i in range(pointer_count)]
    for i in range(pointer_count): 
        m[i][i]=0
    nengo.Connection(report.output,report.input,transform=m,synapse=0.01)
    reported = nengo.networks.EnsembleArray(N_report,pointer_count,
                                 radius=1,encoders=nengo.dists.Choice([[1]]),
                                 intercepts=nengo.dists.Uniform(0.05,0.9))
    nengo.Connection(report.output,reported.input,transform=1,synapse=0.2)
    nengo.Connection(reported.output,report.input,transform=-1)
    nengo.Connection(reported.output,reported.input,transform=1.2)


    # create a system to report whether the first or second number is bigger
    compare = nengo.Ensemble(N_compare,1)
    nengo.Connection(memory.ensembles[0],compare,transform=50)
    nengo.Connection(memory.ensembles[1],compare,transform=-50)


    
    # create inhibitory gates to control the two reporting systems
    report_gate_f=nengo.Ensemble(50,1,encoders=nengo.dists.Choice([[1]]),
                                 intercepts=nengo.dists.Uniform(0.1,0.9))
    report_gate_c=nengo.Ensemble(50,1,encoders=nengo.dists.Choice([[1]]),
                                 intercepts=nengo.dists.Uniform(0.1,0.9))
    nengo.Connection(report_finger,report_gate_f,transform=-10)
    nengo.Connection(report_compare,report_gate_c,transform=-10)
    report_bias=nengo.Node([1])
    nengo.Connection(report_bias,report_gate_f)
    nengo.Connection(report_bias,report_gate_c)

    nengo.Connection(report_gate_c, compare.neurons, 
                     transform=[[-100.0]]*N_compare, synapse=0.01)
    for i in range(pointer_count):
        nengo.Connection(report_gate_f, report.ensembles[i].neurons, 
                         transform=[[-100.0]]*N_report, synapse=0.01)
        nengo.Connection(report_gate_f, reported.ensembles[i].neurons, 
                         transform=[[-100.0]]*N_report, synapse=0.01)
        

