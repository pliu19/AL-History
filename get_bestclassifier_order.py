__author__ = 'Ping'
import numpy as np


if __name__ == "__main__":

    trials = 10
    numbers = 6
    iteration = 100
    # listoffolder = ["./unc_LogisticRegression_C=0.01/","./unc_LogisticRegression_C=0.1/",
    #                 "./unc_LogisticRegression_C=1.0/","./unc_LogisticRegression_C=10.0/",
    #                 "./unc_LogisticRegression_C=100.0/","./unc_LogisticRegression_C=1000.0/"]
    #
    # listofparameter = ["C=0.01","C=0.1","C=1.0","C=10.0","C=100.0","C=1000.0"]
    listoffolder = ["./unc_MultinomialNB_alpha=0.01/","./unc_MultinomialNB_alpha=0.1/","./unc_MultinomialNB_alpha=1.0/",
                     "./unc_MultinomialNB_alpha=10.0/","./unc_MultinomialNB_alpha=100.0/","./unc_MultinomialNB_alpha=1000.0/"]
    listofparameter = ["alpha=0.01","alpha=0.1","alpha=1","alpha=10","alpha=100","alpha=1000"]

    for i in range(trials):
        temp = np.zeros((iteration,numbers))
        for j in range(numbers):
            filename = listoffolder[j]+ "Accuracy_record_Trial_" + str(i+1) + ".txt"
            temp[:,j] = np.loadtxt(open(filename,"rb"),delimiter=',')

        accuracy = []
        for k in range(iteration):
            a = temp[k]
            first2largest = np.argsort(-a)[:2]
            accuracy.append(first2largest)

        file_index_classifier = "./unc_MultinomialNB/Index_2Biggest_Claasifier_Trial" + str(i+1)
        np.savetxt("%s.txt" %file_index_classifier, accuracy, delimiter=",",fmt='%i')