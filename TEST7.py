__author__ = 'Ping'

import numpy as np

if __name__ == '__main__':

    previous_path = ['./unc_LogisticRegression_(train_C=0.01)_', './unc_LogisticRegression_(train_C=0.1)_',
                     './unc_LogisticRegression_(train_C=1.0)_', './unc_LogisticRegression_(train_C=10.0)_',
                     './unc_LogisticRegression_(train_C=100.0)_', './unc_LogisticRegression_(train_C=1000.0)_']

    pathlist = ['(test_C=0.01)/','(test_C=0.1)/','(test_C=1.0)/',
                '(test_C=10.0)/','(test_C=100.0)/','(test_C=1000.0)/']

    name = []
    performance = []

    total = len(previous_path) * len(pathlist)
    column = 5
    array = np.zeros(shape=(total, column))

    for i in range(len(previous_path)):
        for j in range(len(pathlist)):
            name.append(previous_path[i] + pathlist[j])
            filename = previous_path[i] + pathlist[j]+ "average_accuracy.txt"
            current = np.loadtxt(open(filename,"rb"),delimiter=",")
            average = np.average(current)
            performance.append(average)

    index = np.argmax(performance)
    print name[index]
    print performance[index]

    performance2 = []
    list123 = ['./unc_LogisticRegression_(train_C=0.01)_(test_C=0.01)/', './unc_LogisticRegression_(train_C=0.1)_(test_C=0.1)/',
               './unc_LogisticRegression_(train_C=1.0)_(test_C=1.0)/', './unc_LogisticRegression_(train_C=10.0)_(test_C=10.0)/',
               './unc_LogisticRegression_(train_C=100.0)_(test_C=100.0)/', './unc_LogisticRegression_(train_C=1000.0)_(test_C=1000.0)/']

    for i in range(len(list123)):

        filename2 = list123[i] + "average_accuracy.txt"
        current = np.loadtxt(open(filename2,"rb"),delimiter=",")
        average2 = np.average(current)
        performance2.append(average2)

    index2 = np.argmax(performance2)
    print list123[index2]

    array[:,-1] = performance
    print total
    count = 0
    name = []
    for i in range(len(previous_path)):
        for j in range(len(pathlist)):
            name.append(previous_path[i] + pathlist[j])
            filename = previous_path[i] + pathlist[j]+ "average_accuracy.txt"
            current = np.loadtxt(open(filename,"rb"),delimiter=",")
            a = np.sum(current[:25])
            b = np.sum(current[:50]) - a
            c = np.sum(current[:75]) - b - a
            d = np.sum(current) - c - b - a
            a = a / 25.
            b = b / 25.
            c = c / 25.
            d = d / 25.
            array[count,:-1] = [a,b,c,d]
            count = count + 1
    print array
    w = np.argmax(array[:,0])
    x = np.argmax(array[:,1])
    y = np.argmax(array[:,2])
    z = np.argmax(array[:,3])
    print w, x, y, z
    print name[w]
    print name[x]
    print name[y]
    print name[z]
    file_name = "performance_of_each_stage"
    np.savetxt("%s.csv" %file_name, array, delimiter=",",fmt='%10.10f')
