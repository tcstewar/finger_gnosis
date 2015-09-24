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

def task_compare():
    def run():
        for i in range(100):
            pointer.FingerGnosis().run(seed=i,
                                       task='compare',
                                       N_reference=300,
                                       N_memory=500,
                                       data_dir='compare',
                                       pointer_count=3)

    return dict(actions=[run], verbosity=2)

def task_plot_compare():
    def plot():
        data = ctn_benchmark.Data('compare')

        plot = ctn_benchmark.Plot(data)

        plot.measures(['score1', 'score2', 'score3', 'score4','score5','score6', 'score7', 'score8'], outlier_cutoff=None,
                ylim=(0,1))

        import pylab
        pylab.show()
    return dict(actions=[plot], verbosity=2)

def task_data_compare():
    def gather():
        data = ctn_benchmark.Data('compare')
        d = []
        for i in range(1, 9):
            d.append(1.0 - data.get('score%d' % i))
        d = np.array(d).T

        for row in d:
            print ', '.join(['%1.3f' % x for x in row])
    return dict(actions=[gather], verbosity=2)

def task_plot():
    def plot():
        data = ctn_benchmark.Data('data')

        plot = ctn_benchmark.Plot(data)

        plot.vary('_N_reference', ['score1', 'score2', 'score3', 'score4'])

        import pylab
        pylab.show()
    return dict(actions=[plot], verbosity=2)


def task_plot2():
    def plot():
        data = ctn_benchmark.Data('data')

        import pylab
        pylab.figure(figsize=(10,5))

        Ns = np.linspace(10, 200, 5)
        for i, N in enumerate(Ns):
            pylab.subplot(1, len(Ns), i+1)
            pylab.title('N=%d' % N)

            sd = 20
            N_refs = data.get('_N_reference')
            weights = np.exp(-(N_refs-N)**2/(2*sd**2))
            weights /= np.sum(weights)

            measures = ['score1', 'score2', 'score3', 'score4']

            low = []
            high = []
            sample = []
            for m in measures:
                d = data.get(m)
                d = 1.0 - d

                ci = ctn_benchmark.stats.bootstrapci(d, np.mean, weights=weights)
                low.append(ci[0])
                high.append(ci[1])
                sample.append(np.average(d, weights=weights))

            x = np.arange(len(measures)) + 1
            pylab.fill_between(x, low, high, color='#aaaaaa')
            pylab.plot(x, sample)
            pylab.ylim(0, 1)
            pylab.xticks(x)
            pylab.xlim(0.5, len(measures)+0.5)



        pylab.tight_layout()
        pylab.show()
    return dict(actions=[plot], verbosity=2)

