import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.ticker as ticker

colors = ['r', 'b', 'g', 'y', 'k']

def read_data(filename):
    time, epoch, batch_idx, values = [], [], [], []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        count = 0
        for row in csv_reader:
            if count == 0:
                count += 1
                continue
            count += 1

            time.append(row[0])
            epoch.append(row[1])
            batch_idx.append(row[2])
            values.append(row[3])

    values = np.array(values).astype(float)

    time = np.array(time).astype(float)

    epoch = np.array(epoch).astype(float)
    batch_idx = np.array(batch_idx).astype(float)

    increment = batch_idx[1] - batch_idx[0]
    batches_per_epoch = np.amax(batch_idx) + increment

    idx = epoch + batch_idx / batches_per_epoch

    return time, idx, values

def plot_eigs():
    ref_eig = []
    exp_eig = []
    with open("../eig_results/hessian_eigs/eig_comm-1_bmult-1_cores-16_eig-comm-1") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        count = 0
        for row in csv_reader:
            if count == 0:
                count += 1
                continue
            count += 1

            ref_eig.append(float(row[0]))
            exp_eig.append(float(row[1]))

    plt.scatter(exp_eig, ref_eig)
    plt.ylabel("Eigenvalues with Communication")
    plt.xlabel("Eigenvalues without Communication")
    plt.title("Comparison of Eigenvalues with and without Communication")
    plt.savefig("plots/Eigenvalues.png")
    plt.clf()

def plot_individual(x_vals, y_vals, x_axis, names, title):
    for i in range(len(x_vals)):
        plt.plot(x_vals[i], y_vals[i], label=names[i], color = colors[i])
    plt.title(title)
    plt.ylabel("Accuracy")
    plt.xlabel(x_axis)
    plt.legend(loc="upper left")
    plt.savefig("plots/{}.png".format(title.replace(" ", "_")))
    plt.clf()

def plot_times_bar(times, names, title):
    total_times = [time[-1] for time in times]
    x_pos = [i for i in range(len(names))]
    
    plt.bar(x_pos, total_times)
    plt.xticks(x_pos, names)
    plt.ylabel("Time in seconds to complete 30 epochs")
    plt.title(title)
    plt.savefig("plots/{}.png".format(title.replace(" ", "_")))
    plt.clf()

def plot(files, names, title):
    times = []
    idxs = []
    values = []
    for file in files:
        time, idx, value = read_data(file)
        times.append(time)
        idxs.append(idx)
        values.append(value)

    plot_individual(times, values, "Time in Seconds", names, "{} vs Time".format(title))
    plot_individual(idxs, values, "Epoch", names, "{} vs Epoch".format(title))
    plot_times_bar(times, names, "{} Time to Completion".format(title))

# Overall (Vanilla Hess, Vanilla, Dist, Hess (comm = 1, batch_mult = 1))
plot(
    ["../nd_results/nd_nonhessian/test_acc_cores-1_comm-1_batch-size-128",
     "../nd_results/nd_hessian/test_acc_comm-1_bmult-1_cores-1_eig-comm-1_init-batch-size-128",
     "../dist_results/nonhessian/test_acc_cores-16_comm-1",
     "../dist_results/hessian/test_acc_comm-1_bmult-1_cores-16_eig-comm-1_init-batch-size-128"],
    ["Vanilla", "Vanilla + Hessian", "Distributed", "Distributed + Hessian"],
    "Overall Comparison of Training Methods")

# Num Nodes
plot(
    ["../nd_results/nd_nonhessian/test_acc_cores-1_comm-1_batch-size-128",
     "../dist_results/nonhessian/test_acc_cores-4_comm-1",
     "../dist_results/nonhessian/test_acc_cores-8_comm-1",
     "../dist_results/nonhessian/test_acc_cores-16_comm-1",],
    ["1 Worker (Non-Dist)", "4 Workers", "8 Workers", "16 Workers"],
    "Number of Workers Used")

# Comm Comparison
plot(
    ["../dist_results/hessian/test_acc_comm-1_bmult-1_cores-16_eig-comm-1_init-batch-size-128",
     "../dist_results/hessian/test_acc_comm-5_bmult-1_cores-16_eig-comm-1",
     "../dist_results/hessian/test_acc_comm-10_bmult-1_cores-16_eig-comm-1",],
    ["Comm 1", "Comm 5", "Comm 10"],
    "Communication Intervals")

# Batchmult Comparison
plot(
    ["../dist_results/hessian/test_acc_comm-1_bmult-1_cores-16_eig-comm-1_init-batch-size-128",
     "../dist_results/hessian/test_acc_comm-1_bmult-2_cores-16_eig-comm-1",
     "../dist_results/hessian/test_acc_comm-1_bmult-4_cores-16_eig-comm-1",],
    ["Bmult 1", "Bmult 2", "Bmult 4"],
    "Batch Multipliers")

# Eig Comm vs No
plot(
    ["../dist_results/hessian/test_acc_comm-1_bmult-1_cores-16_eig-comm-1_init-batch-size-128",
     "../dist_results/hessian/test_acc_comm-1_bmult-1_cores-16_eig-comm-0",],
    ["Eig Comm", "No Eig Comm"],
    "Eigenvalue Communcation for Hessian")

plot_eigs()