__author__ = 'Ping'
from scipy.stats import itemfreq
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,LogLocator,LinearLocator


class plot_result(object):

    def __init__(self, folderlist, label, number, strategy, classifier, folder = None):
        self.folderlist = folderlist
        self.label = label
        self.num_test = number
        self.strategy = strategy
        self.classifier = classifier

        self.folderpath = folder

        # self.plot_lastcross()

    def plot_lastcross(self):

        self.num_test = 25000
        title = self.strategy + "_" + self.classifier + "_lastplot"
        numberoflines = len(self.folderlist)

        fig = plt.figure(figsize=(60,15))
        ax = fig.add_subplot(111)
        plt.suptitle(title,fontsize=50)

        # plt.gca().set_color_cycle(['red', 'green', 'blue', 'black'])

        for i in range(numberoflines):
            path = self.folderlist[i] + self.strategy + '_Mean_Lastcross.txt'
            frequency = []
            with open(path) as csv:
                for j in range(self.num_test):
                    line = csv.readline()
                    int_list = map(float, line.split(','))
                    frequency.append(int_list[0])

            temp = itemfreq(frequency)
            listX = []
            listPercentage = []
            row, column = temp.shape

            for j in range(row):
                listX.append(temp[j][0])
                listPercentage.append(temp[j][1]/self.num_test)

            pc = np.cumsum(listPercentage)
            plt.plot(listX,pc, label = self.label[i])

        # plot
        # Y_major_ticks = np.arange(0, 1.5, 0.1)
        # Y_minor_ticks = np.arange(0, 1.5, 0.02)
        # ax.set_yticks(Y_major_ticks)
        # ax.set_yticks(Y_minor_ticks, minor=True)

        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        ax.xaxis.grid(True,'minor')
        ax.yaxis.grid(True,'minor')
        ax.xaxis.grid(True,'major',linewidth=2)
        ax.yaxis.grid(True,'major',linewidth=2)

        plt.legend(loc = 4,handleheight = 4, fontsize = 25, ncol = numberoflines)

        nameofpic = title + '.png'
        plt.savefig(nameofpic)


    def plot_accuracy(self, trial, iteration, plus_title = None ):

        iteration = iteration + 1
        ind = np.arange(1,iteration)

        folderpath = "./" + self.strategy + "_" + self.classifier + "/"

        if not self.folderpath:
            self.folderpath = folderpath

        if not os.path.exists(self.folderpath):
                os.mkdir(self.folderpath)

        NumberofExample = len(self.label)

        for i in range(NumberofExample):

            fig = plt.figure(figsize=(24, 24))

            for j in range(trial):

                ax = fig.add_subplot(trial,1,j+1)
                filename = self.folderlist[i]+ "Accuracy_record_Trial_" + str(j+1) + ".txt"
                current = np.loadtxt(open(filename,"rb"))
                current = current.tolist()
                plt.plot(ind, current)
                plt.ylim([0,1])
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.xaxis.set_minor_locator(MultipleLocator(1))

                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax.yaxis.set_minor_locator(MultipleLocator(0.1))

                ax.xaxis.grid(True,'minor')
                ax.yaxis.grid(True,'minor')
                ax.xaxis.grid(True,'major',linewidth=1)
                ax.yaxis.grid(True,'major',linewidth=1)

            name = folderpath + self.label[i] + ".png"
            plt.savefig(name)

        for i in range(trial):
            fig = plt.figure(figsize=(24, 12))
            ax = fig.add_subplot(111)
            for j in range(NumberofExample):
                filename = self.folderlist[j]+ "Accuracy_record_Trial_" + str(i+1) + ".txt"
                current = np.loadtxt(open(filename,"rb"),delimiter=",")
                current = current.tolist()
                plt.plot(ind, current, label = self.label[j])

            plt.ylim([0,1])
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))

            ax.xaxis.grid(True,'minor')
            ax.yaxis.grid(True,'minor')
            ax.xaxis.grid(True,'major',linewidth=1)
            ax.yaxis.grid(True,'major',linewidth=1)

            plt.legend(loc = 4,handleheight = 4, fontsize = 20, ncol = NumberofExample)

            if plus_title == None:
                name = self.folderpath + "Trial_" + str(i+1) + "_AllLines.png"
            else:
                name = self.folderpath + "Trial_" + str(i+1) + "_AllLines" + str(plus_title) +".png"
                title = "Trial_" + str(i+1) + "_AllLines" + str(plus_title) +".png"
                ax.set_title(title)



            plt.savefig(name)

if __name__ == '__main__':

    folderlist = ['./rand_MultinomialNB_alpha=0.1/', './rand_MultinomialNB_alpha=1.0/',
                   './rand_MultinomialNB_alpha=10.0/', './rand_MultinomialNB_alpha=100.0/']

    label = ["alpha=0.1","alpha=1.0","alpha=10.0","alpha=100.0"]
    test = plot_result(folderlist, label, 25000, 'rand', 'MultinomialNB','./')
    test.plot_lastcross()
    # test.plot_accuracy(10,100,"_(train_C=1000.0)")
    # print folderlist
