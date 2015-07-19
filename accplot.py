import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == '__main__':

    pltcount = 0
    fig = plt.figure(figsize=(240, 30))
    fig.subplots_adjust(hspace = 0.3) #space for titles of subplots
    pathgen = './unc_LogisticRegression_C='

    for trials in range(10):
        for ii in range(1,7): #For six complexities
            for jj in range(ii+1,7): #C 6 2

                c1=str(10000./(10**ii))
                c2=str(10000./(10**jj))
                path1 = pathgen + c1 + '/'
                path2 = pathgen + c2 + '/'
                count = 0
                base = 0
                results = []

                ax = plt.subplot(10, 15, pltcount)
                ax.set_title(c1+'vs'+c2)
                ax.set_yticks(np.linspace(0,1,7))
                ax.xaxis.set_major_locator(plt.AutoLocator())
                t1=np.loadtxt(open(path1 + "Indices_record_Trial_" + str(trials + 1) + ".csv",'rb'), delimiter=',')
                t2=np.loadtxt(open(path2 + "Indices_record_Trial_" + str(trials + 1) + ".csv",'rb'), delimiter=',')

                for i in range(len(t1)):
                    base += len(t1[i])
                    for j in range(len(t1[i])):
                        if t1[i][j] in t2[i]:
                            count+=1
                    percent=float(count)/base
                    results.append(percent)
                    if ((i+1)%10==0):
                        #plt.text(i+1, percent, str(round(percent-1./(i+1),8)))
                        #comment percentages. Above is difference
                        plt.text(i+1, percent, str(round(percent,8)))
                plt.plot(results)
                pltcount+=1
    plt.savefig("1.png")
    