__author__ = 'Ping'
from scipy.stats import itemfreq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


if __name__ == "__main__":

    frequency = []

    with open("./rand_LogisticRegression_C=0.1/rand_Mean_Lastcross.txt") as csv:
        for j in range(25000):
            line = csv.readline()
            print line
            frequency.append(int(line))

    temp = itemfreq(frequency)
    print temp
    print temp.shape
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

    plt.savefig("./rand_LogisticRegression_C=0.1/foo.png")
