__author__ = 'Ping'
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,LogLocator,LinearLocator

if __name__ == "__main__":

    trial = 10
    ind = np.arange(1,101)
    folderpath = "./unc_BernoulliNB/"
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    # ListofFolder = ["./unc_LogisticRegression_C=0.1/","./unc_LogisticRegression_BestClassifier/",
    #                 "./unc_LogisticRegression_BestClassifier_Indices/","./unc_LogisticRegression_BestClassifier_Random/",
    #                 "./unc_LogisticRegression_Best2Classifier/"]
    # ListofParameter = ["C=0.1","BestClassifier", "BestClassifier_Indices","BestClassifier_Random","Best2Classifier"]
    ListofFolder = ["./unc_BernoulliNB_alpha = 0.01/","./unc_BernoulliNB_alpha = 0.1/","./unc_BernoulliNB_alpha = 1.0/",
                    "./unc_BernoulliNB_alpha = 10.0/","./unc_BernoulliNB_alpha = 100.0/","./unc_BernoulliNB_BestClassifier/"]
    ListofParameter = ["alpha=0.01","alpha=0.1","alpha=1","alpha=10","alpha=100","BestClassifier"]
    # ListofFolder = ["./unc_LogisticRegression_C=0.01/","./unc_LogisticRegression_C=0.1/",
    #                 "./unc_LogisticRegression_C=1.0/","./unc_LogisticRegression_C=10.0/",
    #                 "./unc_LogisticRegression_C=100.0/","./unc_LogisticRegression_C=1000.0/",
    #                 "./unc_LogisticRegression_BestClassifier/"]
    # ListofParameter = ["C=0.01","C=0.1","C=1.0","C=10.0","C=100.0","C=1000.0","BestClassifier"]
    NumberofExample = len(ListofParameter)

    for i in range(NumberofExample):

        fig = plt.figure(figsize=(24, 24))

        for j in range(trial):

            ax = fig.add_subplot(trial,1,j+1)
            filename = ListofFolder[i]+ "Accuracy_record_Trial_" + str(j+1) + ".txt"
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

        name = folderpath + ListofParameter[i] + ".png"
        plt.savefig(name)

    for i in range(trial):
        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        for j in range(NumberofExample):
            filename = ListofFolder[j]+ "Accuracy_record_Trial_" + str(i+1) + ".txt"
            current = np.loadtxt(open(filename,"rb"),delimiter=",")
            current = current.tolist()
            plt.plot(ind, current, label = ListofParameter[j])

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

        name = folderpath + "Trial_" + str(i+1) + "_AllLines+BestClassifier.png"
        plt.savefig(name)