import pointer
import ctn_benchmark
import numpy as np

def task_vary_ref():
    def run():
        for i in range(20):
            N_reference = np.random.randint(10, 200)
            pointer.FingerGnosis().run(seed=i, 
                                       N_reference=N_reference, 
                                       pointer_count=5)

    return dict(actions=[run], verbosity=2)

def task_plot():
    def plot():
        data = ctn_benchmark.Data('data')

        plot = ctn_benchmark.Plot(data)

        plot.vary('_N_reference', ['score1', 'score2', 'score3', 'score4'])

        import pylab
        pylab.show()
    return dict(actions=[plot], verbosity=2)


