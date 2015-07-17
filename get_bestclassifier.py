__author__ = 'Ping'
import numpy as np


if __name__ == "__main__":
    # a = np.array([3,4,1,10,5])
    # print a
    # print np.argmax(a)
    trials = 10
    numbers = 6
    iteration = 100
    listoffolder = ["./unc_LogisticRegression_C=0.01/","./unc_LogisticRegression_C=0.1/",
                    "./unc_LogisticRegression_C=1.0/","./unc_LogisticRegression_C=10.0/",
                    "./unc_LogisticRegression_C=100.0/","./unc_LogisticRegression_C=1000.0/"]

    listofparameter = ["C=0.01","C=0.1","C=1.0","C=10.0","C=100.0","C=1000.0"]

    for i in range(trials):
        temp = np.zeros((iteration,numbers))
        for j in range(numbers):
            filename = listoffolder[j]+ "Accuracy_record_Trial_" + str(i+1) + ".csv"
            temp[:,j] = np.loadtxt(open(filename,"rb"))

        accuracy = []
        for k in range(iteration):
            a = temp[k]
            accuracy.append(np.argmax(a))
        file_index_classifier = "./unc_LogisticRegression/Index_Claasifier_Trial" + str(i+1)
        np.savetxt("%s.txt" %file_index_classifier, accuracy, delimiter=",")