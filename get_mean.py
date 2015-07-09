__author__ = 'Ping'
import numpy as np

def Lastcross(array,lengthoftest):
def Lastcross(array):

    last_cross = []
    for j in range(lengthoftest):
        temp = array[j]

        if temp[0]>0.5:
            lastpred = 0
        else:
            lastpred = 1

        length = len(temp)
        lastcross = 0
        count=0
        for i in range(length-1):
            if temp[i]>0.5:
                flag = 0
            else:
                flag = 1
            if flag != lastpred:
                count += 1
                lastcross = i
            lastpred = flag
        last_cross.append(lastcross)

    return last_cross

if __name__ == '__main__':

    budget = 5000
    bootstrap = 10
    stepsize = 10
    column_file = (budget - bootstrap) / stepsize + 2
    row_file = 10361

    folder = "./rand_MultinomialNB_alpha=100.0/"

    mean_probal = np.zeros(shape=(row_file,column_file))
    for i in range(10):
        path = folder +"rand_Trial_" + str(i+1)+ "_proba.csv"
        temp = np.loadtxt(open(path,"rb"),delimiter=",")
        mean_probal += temp

    mean_probal = mean_probal / 10.

    file_name_mean_prediction = folder + "rand_Mean_proba"
    np.savetxt("%s.csv" %file_name_mean_prediction, mean_probal, delimiter=",")

    last_cross_mean = Lastcross(mean_probal,row_file)
    last_cross_mean = np.array(last_cross_mean)
    print last_cross_mean
    file_name_lastcross_mean = folder + "rand_Mean_Lastcross"
    np.savetxt("%s.txt" %file_name_lastcross_mean, last_cross_mean.astype(int), delimiter=",",fmt='%i')
