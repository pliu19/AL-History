__author__ = 'Ping'
from scipy.stats import itemfreq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == "__main__":

    frequency = []
    with open("./rand_LogisticRegression_C=0.1/rand_Trial_1_crosstable.txt") as csv:
        for j in range(25000):
            line = csv.readline()
            int_list = map(float, line.split(','))
            # item = line.split(',')
            # int_list = [int(i) for i in item]
            frequency.append(int_list[1])

    temp = itemfreq(frequency)
    listX = []
    listFrequency = []
    listPercentage = []
    row, column = temp.shape
    column = column + 1
    newarray = np.zeros(shape=(row, column))
    for i in range(row):
        listX.append(temp[i][0])
        listFrequency.append(temp[i][1])
        listPercentage.append(temp[i][1]/25000.)
        print listPercentage[i]

    print listFrequency

    fig = plt.figure(figsize=(9,12))
    ax1 = fig.add_subplot(311)
    plt.suptitle("This is whole title")
    ind = 20

    ax1.hist(frequency,ind,alpha=0.8,cumulative=False)
    ax1.set_title("The frequency of lastcross")

    frequency = np.array(frequency)
    ax2 = fig.add_subplot(312)
    ax2.hist(frequency,ind,alpha=0.8, weights=np.zeros_like(frequency) + 1. / frequency.size,cumulative=False)
    ax2.set_title("The percentage of lastcross - Non-cumulative")

    frequency = np.array(frequency)
    ax3 = fig.add_subplot(313)
    ax3.hist(frequency,ind,alpha=0.8, weights=np.zeros_like(frequency) + 1. / frequency.size,cumulative=True)
    ax3.set_title("The percentage of lastcross - Cumulative")
    # plt.show()

    plt.savefig("foo.png")
