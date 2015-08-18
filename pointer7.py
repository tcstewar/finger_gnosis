input_count=2     # how many types of inputs there are (must be 2: finger and magnitude)
N_input=200       # number of neurons to represent the input
N_pointer=400     # number of neurons per pointer
N_reference=1000  # number of neurons to store the value of the pointer (dereferenced)
N_memory=2000     # number of neurons to store the value of the pointer over time
N_compare=400     # number of neurons to do magnitude comparison
N_report=100      # number of neurons to read out the value of memory
pointer_count=3   # number of pointers
crosstalk=0.2     # amount of interference between pointers (0=no interference)

compare_numbers=0.7,0.8  # the two numbers to compare (should be between 0.5 and 1)
fingers=0,2              # the fingers to touch

mode='fingers'    # 'fingers' for finger-touching experiment, 
                  # 'compare' for magnitude comparison, 
                  # 'manual' if you want to manually control inputs


import nef
import math
net=nef.Network('Pointer')



# define the inputs when doing number comparison task
class NumberExperiment(nef.SimpleNode):
    def origin_input0(self):
        return [0]*pointer_count
    def origin_input1(self):
        t=self.t_start
        if 0.1<t<0.2: return [compare_numbers[0]]
        if 0.3<t<0.4: return [compare_numbers[1]]
        return [0]
    def origin_pointer_source(self):
        return [0,1]    
    def origin_pointer_target(self):
        t=self.t_start
        v=[0]*pointer_count
        if 0.1<t<0.2: v[0]=1
        if 0.3<t<0.4: v[1]=1
        return v
    def origin_report_finger(self):
        return [0]
    def origin_report_compare(self):
        t=self.t_start
        if 0.5<t<1.0:
            return [1]
        else:
            return [0]    
        
# define the inputs when doing the finger touching task        
class FingerTouchExperiment(nef.SimpleNode):
    def origin_input0(self):
        t=self.t_start
        r=[0]*pointer_count
        if 0.1<t<0.2: 
            for i in fingers:
                if i<pointer_count: r[i]=1
        return r    
    def origin_input1(self):
        return [0]
    def origin_pointer_source(self):
        return [1,0]    
    def origin_pointer_target(self):
        return [1]*pointer_count
    def origin_report_finger(self):
        t=self.t_start
        if 0.3<t<1.0:
            return [1]
        else:
            return [0]    
    def origin_report_compare(self):
        return [0]
        
        
        
# connect the experiment inputs to the neural model        
if mode=='manual':
    input0=net.make_input('0',[0]*pointer_count)
    input1=net.make_input('1',[0])
    pointer_source=net.make_input('pointer source',[0]*input_count)
    pointer_target=net.make_input('pointer target',[1]*pointer_count)
    report_finger=net.make_input('report finger',[0])
    report_compare=net.make_input('report compare',[0])
else:
    if mode=='compare':
        exp=NumberExperiment('Experiment')
    elif mode=='fingers':    
        exp=FingerTouchExperiment('Experiment')
    net.add(exp)
    input0=net.make('0',1,pointer_count,mode='direct',quick=True)
    input1=net.make('1',1,1,mode='direct',quick=True)
    pointer_source=net.make('pointer source',1,input_count,mode='direct',quick=True)
    pointer_target=net.make('pointer target',1,pointer_count,mode='direct',quick=True)
    net.connect(exp.getOrigin('input0'),input0)
    net.connect(exp.getOrigin('input1'),input1)
    net.connect(exp.getOrigin('pointer_source'),pointer_source)
    net.connect(exp.getOrigin('pointer_target'),pointer_target)
    report_finger=net.make('report flag',1,1,mode='direct',quick=True)
    report_compare=net.make('report compare',1,1,mode='direct',quick=True)
    net.connect(exp.getOrigin('report_finger'),report_finger)
    net.connect(exp.getOrigin('report_compare'),report_compare)
    


# create neural models for the two input areas (fingers and magnitude)
area0=net.make('area0',N_input*pointer_count,pointer_count,radius=math.sqrt(pointer_count),quick=True)
area1=net.make('area1',N_input,1,quick=True)

net.connect(input0,area0)
net.connect(input1,area1)



# define the connections to create the pointers
def matrix(n,m,pre=None,post=None,value=1):
	m=[[0]*n for i in range(m)]
	if pre is None: pre=range(n)
	if post is None: post=range(m)
	for i in range(max(len(pre),len(post))):
		m[post[i%len(post)]][pre[i%len(pre)]]=value
	return m
for i in range(pointer_count):
	pointer=net.make('pointer%d'%i,N_pointer,input_count*2+1,radius=math.sqrt(input_count*2+1),quick=True)
	net.connect(pointer_source,pointer,transform=matrix(input_count,input_count*2+1,post=[k*2 for k in range(input_count)]))
	net.connect(pointer_target,pointer,transform=matrix(pointer_count,input_count*2+1,pre=[i],post=[input_count*2]))
	net.connect(area0,pointer,transform=matrix(pointer_count,input_count*2+1,pre=[i],post=[1]))
	net.connect(area1,pointer,transform=matrix(1,input_count*2+1,pre=[0],post=[3]))


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
reference=net.make('references',N_reference,pointer_count,radius=math.sqrt(pointer_count),encoders=basis,quick=True,intercept=(0.1,0.9))
for i in range(pointer_count):
	matrix=[crosstalk]*pointer_count
	matrix[i]=1.0-crosstalk
	net.connect('pointer%d'%i,reference,func=ref_func,transform=[[x] for x in matrix])
	
	
	
# add a memory to integrate the value referenced by the pointers
memory=net.make_array('memory',N_memory,pointer_count,radius=1,quick=True)
net.connect(reference,memory,weight=1)
net.connect(memory,memory,weight=1,pstc=0.2)



# create a system to report which fingers were pressed
report=net.make_array('report',N_report,pointer_count,quick=True,encoders=[[1]],intercept=(0.3,0.9),radius=0.3)
net.connect(memory,report,weight=1)
m=[[-10]*pointer_count for i in range(pointer_count)]
for i in range(pointer_count): m[i][i]=0
net.connect(report,report,transform=m,pstc=0.01)
reported=net.make_array('reported',N_report,pointer_count,radius=1,quick=True,encoders=[[1]],intercept=(0.05,0.9))
net.connect(report,reported,weight=1,pstc=0.2)
net.connect(reported,report,weight=-1)
net.connect(reported,reported,weight=1.2)


# create a system to report whether the first or second number is bigger
compare=net.make('compare',N_compare,1,quick=True)
net.connect(memory,compare,weight=50,index_pre=0)
net.connect(memory,compare,weight=-50,index_pre=1)



# create inhibitory gates to control the two reporting systems
report_gate_f=net.make('report gate f',50,1,encoders=[[1]],intercept=(0.1,0.9),quick=True)
report_gate_c=net.make('report gate c',50,1,encoders=[[1]],intercept=(0.1,0.9),quick=True)
net.connect(report_finger,report_gate_f,weight=-10)
net.connect(report_compare,report_gate_c,weight=-10)
report_bias=net.make_input('report bias',[1])
net.connect(report_bias,report_gate_f)
net.connect(report_bias,report_gate_c)

report.addTermination('gate',[[[-100.0]]*N_report]*pointer_count,0.01,False)
reported.addTermination('gate',[[[-100.0]]*N_report]*pointer_count,0.01,False)
compare.addTermination('gate',[[-100.0]]*N_compare,0.01,False)

net.connect(report_gate_f,report.getTermination('gate'))
net.connect(report_gate_f,reported.getTermination('gate'))
net.connect(report_gate_c,compare.getTermination('gate'))





net.add_to(world)
net.view(play=1.5)
