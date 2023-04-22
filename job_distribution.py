import numpy as np
import matplotlib.pyplot as plt

class Dist:

    def __init__(self, num_res, max_nw_size, job_len):
        self.num_res = num_res
        self.max_nw_size = max_nw_size
        self.job_len = job_len

        self.job_small_chance = 0.117
        self.job_big_chance = 1 - self.job_small_chance

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 1
        self.job_len_small_upper = job_len / 5

        self.dominant_res_lower = max_nw_size / 2
        self.dominant_res_upper = max_nw_size

        self.other_res_lower = 1
        self.other_res_upper = max_nw_size / 5

        self.counter_1 = 0
        self.counter_2 = 0
    

        # Use for merge distributions
        self.dist_a = self.exp_dist
        self.dist_b = self.bernoulli_dist

        self.dist_a_name = 1
        self.dist_b_name = 2

    def normal_dist(self):

        # new work duration
        nw_len = np.random.randint(1, self.job_len + 9)  # same length in every dimension

        return nw_len, 7

    def poisson_dist(self):
        # new work duration
        nw_len = np.random.poisson(lam=self.job_len-4.004)


        return nw_len, 3
    
    def exp_dist(self):
        # new work duration
        # WARNING: temp hardcode for max job 15
        nw_len = np.random.exponential(scale=13.65)

        return nw_len, 4


    def bernoulli_dist(self):
        # -- job length --
        if np.random.rand() < self.job_small_chance:  # small job
            nw_len = np.random.binomial(self.job_len_small_upper, self.job_small_chance)
        else:  # big job
            nw_len = np.random.binomial(self.job_len_big_upper, self.job_big_chance)
        return nw_len, 5

    def bi_model_dist(self):

        # -- job length --
        if np.random.rand() < 0.09:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)
        return nw_len, 6

    def merged_dist(self):
         simu_len = 500
         if self.counter_1 < simu_len // 2 and self.counter_2 < simu_len // 2:
             # Choose the distribution based on random probability
             if np.random.rand() < 0.5:
                 # distribution 0
                 nw_len, _ = self.dist_a()
                 dist_name = self.dist_a_name
                 self.counter_1 += 1
             else:
                # distribution 1
                 nw_len, _ = self.dist_b()
                 dist_name = self.dist_b_name
                 self.counter_2 += 1
         else:
             # Choose the other distribution to balance the total number of jobs
             if self.counter_1 == simu_len // 2:
                # distribution 1
                 nw_len, _ = self.dist_b()
                 dist_name = self.dist_b_name
             else:
                # distribution 0
                 nw_len, _ = self.dist_a()
                 dist_name = self.dist_a_name

         # self.current_dist = dist_name
         return nw_len, dist_name


def sequence_statistics_flat(seq, output_path, title, dist_name):
    plt.figure()
    plt.hist(seq, bins='auto') 
    plt.title(dist_name + " Total Jobs' Length Distribution") 
    plt.xlabel("Job Length")
    plt.ylabel("Count")
    plt.savefig(output_path + title + ".pdf")
    plt.show()
    print "======================================"

def sequence_statistics_by_example(seq, output_path, outfile, dist_name, hist_title, hist_xaxis):
    f = open(output_path + outfile + ".txt", "w")
    l = "======================================"
    f.write(l + "\n")
    print l
    l = "Statistic on jobs: "
    f.write(l + "\n")
    print l
    l = "======================================"
    f.write(l + "\n")
    print l
    l = "sequence is: " + outfile
    f.write(l + "\n")
    print l
    l = "sequence size is:  {}".format(seq.size)
    f.write(l + "\n")
    print l
    l = "sequence shape is: {}".format(seq.shape)
    f.write(l + "\n")
    print l
    l = "range of value is: {}".format(seq.ptp())
    f.write(l + "\n")
    print l
    l = "max is:  {}".format(seq.max())
    f.write(l + "\n")
    print l
    l = "min is:  {}".format(seq.min())
    f.write(l + "\n")
    print l
    l = "mean is: {}".format(seq.mean())
    f.write(l + "\n")
    print l
    l = "std is:  {}".format(seq.std())
    f.write(l + "\n")
    print l
    l = "var is:  {}".format(seq.var())
    f.write(l + "\n")
    print l
    f.close()

    ### histogram
    num_ex, num_job_in_ex = seq.shape
    labels = []
    for i in range(0,num_ex):
        labels.append("Training Set #" + str(i))
    plt.figure()
    plt.hist(seq, label=labels) 
    plt.legend()
    plt.title(dist_name + hist_title) 
    plt.xlabel(hist_xaxis)
    plt.ylabel("Count")
    plt.savefig(output_path + outfile + ".pdf")
    plt.show()
    print "======================================"

def sequence_statistics_resource_workload(nw_len_seq, nw_size_seq, output_path, dist_name):
    size = nw_size_seq.size
    num_ex, num_job_in_ex = nw_len_seq.shape
    _, _, num_res = nw_size_seq.shape
    workload = np.zeros([num_ex, num_job_in_ex])

    for i in xrange(num_ex):
        for j in range(num_job_in_ex):
            for k in range(num_res):
                workload[i][j] += nw_len_seq[i][j] * nw_size_seq[i][j][k]
        print("Load on # " + str(i) + " resource dimension is " + str(workload[i]))

    # Plot the sum workload 
    sequence_statistics_by_example(workload, output_path, "workload_resource", dist_name, " By Training Set Job Resource Size Histogram", "Total Job Resource")


def generate_sequence_work(pa, seed=42):

    np.random.seed(seed)

    simu_len = pa.simu_len * pa.num_ex

    ##############
    # distribution name
    nw_dist = pa.dist.merged_dist
    nw_dist_name = "Exponential + Bernoulli Distribution"
    ##############

    nw_len_seq = np.zeros(simu_len, dtype=int)
    nw_size_seq = np.ones((simu_len, pa.num_res), dtype=int)

    # nw_dist_seq = np.empty(simu_len, dtype=int)
    nw_dist_seq = np.zeros(simu_len, dtype=int)

    # for i in range(simu_len):

    #     if np.random.rand() < pa.new_job_rate:  # a new job comes

    #         nw_len_seq[i], nw_size_seq[i, :] = nw_dist()
    for i in range(simu_len):
        if i % pa.simu_len == 0:  # Reset the counter at the beginning of each experiment
            pa.dist.counter_1 = 0
            pa.dist.counter_2 = 0

        if np.random.rand() < pa.new_job_rate:  # a new job comes
            nw_len_seq[i], nw_dist_seq[i] = nw_dist()

            #nw_len_seq[i], nw_size_seq[i, :] = nw_dist()

    sequence_statistics_flat(nw_len_seq, "./data/", "nw_len_seq_total", nw_dist_name)

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])
    nw_size_seq = np.reshape(nw_size_seq,
                             [pa.num_ex, pa.simu_len, pa.num_res])
    nw_dist_seq = np.reshape(nw_dist_seq,
                              [pa.num_ex, pa.simu_len])

    sequence_statistics_by_example(nw_len_seq, "./data/", "nw_len_seq_by_example", nw_dist_name, " By Training Set Job Length histogram", "Job Length")
    sequence_statistics_resource_workload(nw_len_seq, nw_size_seq, "./data/", nw_dist_name)

    return nw_len_seq, nw_size_seq, nw_dist_seq
