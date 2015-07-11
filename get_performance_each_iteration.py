__author__ = 'Ping'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,LogLocator,LinearLocator

if __name__ == "__main__":

    trial = 10
    ind = np.arange(1,101)
    print ind
    ListofFolder = ["./unc_LogisticRegression_C=10.0/","./unc_LogisticRegression_C=100.0/",
                    "./unc_LogisticRegression_C=1000.0/"]
    ListofParameter = ["C=10","C=100","C=1000"]
    NumberofExample = len(ListofParameter)

    for i in range(NumberofExample):

        fig = plt.figure(figsize=(24, 24))

        for j in range(trial):

            ax = fig.add_subplot(trial,1,j+1)
            filename = ListofFolder[i]+ "Accuracy_record_Trial_" + str(j+1) + ".csv"
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

        name = ListofParameter[i] + ".png"
        plt.savefig(name)

