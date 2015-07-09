import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == '__main__':
    count = 0
    base = 0
    results = []
    fig = plt.figure(figsize=(60, 15))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(plt.AutoLocator())
    ax.set_yticks(np.linspace(0,1,7))
    
    path1 = './unc_LogisticRegression_C=0.1/'
    path2 = './unc_LogisticRegression_C=1.0/'
    for trials in range(10):
        t1=np.loadtxt(open(path1 + "Indices_record_Trial_" + str(trials + 1) + ".csv",'rb'), delimiter=',')
        t2=np.loadtxt(open(path2 + "Indices_record_Trial_" + str(trials + 1) + ".csv",'rb'), delimiter=',')
        for i in range(len(t1)):
            base += len(t1[i])
            for j in range(len(t1[i])):
                if t1[i][j] in t2[i]:
                    count+=1
            results.append(float(count)/base)
        xaxislist = range(len(results))
        
        plt.plot(xaxislist, results, marker="o", markersize=15, alpha=0.3)
        plt.savefig(trials+".png")
    
    