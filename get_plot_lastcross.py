__author__ = 'Ping'

from scipy.stats import itemfreq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,LogLocator


if __name__ == "__main__":

    folder_list = ['./rand_LogisticRegression_C=0.1/','./rand_LogisticRegression_C=1.0/',
                   './rand_LogisticRegression_C=10.0/','./rand_LogisticRegression_C=100.0/']
    label = ['C=0.1','C=1.0','C=10.0','C=100.0']
    numberoflines = len(folder_list)

    fig = plt.figure(figsize=(60,15))
    ax = fig.add_subplot(111)
    plt.suptitle("rand_LogisticRegression",fontsize=50)
    plt.gca().set_color_cycle(['red', 'green', 'blue', 'black'])

    for i in range(numberoflines):
        path = folder_list[i] + 'rand_Mean_Lastcross.txt'
        frequency = []
        with open(path) as csv:
            for j in range(25000):
                line = csv.readline()
                int_list = map(float, line.split(','))
                frequency.append(int_list[0])

        temp = itemfreq(frequency)
        listX = []
        listPercentage = []
        row, column = temp.shape

        for j in range(row):
            listX.append(temp[j][0])
            listPercentage.append(temp[j][1]/25000.)

        pc = np.cumsum(listPercentage)
        plt.plot(listX,pc, label = label[i])

    # plot



    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    ax.xaxis.grid(True,'minor')
    ax.yaxis.grid(True,'minor')
    ax.xaxis.grid(True,'major',linewidth=2)
    ax.yaxis.grid(True,'major',linewidth=2)

    plt.legend(loc = 2,handleheight = 4, fontsize = 25, ncol = numberoflines)

    plt.savefig("rand_LogisticRegression.png")
