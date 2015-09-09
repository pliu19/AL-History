__author__ = 'Ping'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
import os
from time import time
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from strategies import RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy

def get_classifier(classifier, argus):

    result = str(classifier) + '(' + argus + ')'
    return result

def write_integer(path,array):
    array = np.asarray(array)
    np.savetxt("%s.txt" %path, array.astype(int), delimiter=",",fmt='%i')

def write_float(path,array):
    array = np.asarray(array)
    np.savetxt("%s.txt" %path, array, delimiter=",",fmt='%10.10f')


class BestClassifier():

    def __init__(self, listofparameters, folderlist, trials, bootstrap_size,  step_size, budget, al_strategy, classifier_name):

        self.parameterslist = listofparameters
        self.folderlist = folderlist
        self.length = len(folderlist)
        self.trials = trials
        self.bootstrap_size = bootstrap_size
        self.step_size = step_size
        self.budget = budget
        self.iteration = (budget - bootstrap_size)/step_size + 1
        self.strategy = al_strategy
        self.classifier = classifier_name

        self.get_BestClassifier()


    def get_BestClassifier(self):

        for i in range(self.trials):

            temp = np.zeros((self.iteration,self.length))

            for j in range(self.length):
                filename = self.folderlist[j]+ "Accuracy_record_Trial_" + str(i+1) + ".txt"
                temp[:,j] = np.loadtxt(open(filename,"rb"),delimiter=',')

            accuracy = []
            for k in range(self.iteration):
                a = temp[k]
                accuracy.append(np.argmax(a))

            file_index_classifier = "./" + self.strategy + "_" + self.classifier + "/Index_Claasifier_Trial" + str(i + 1)
            np.savetxt("%s.txt" %file_index_classifier, accuracy, delimiter=",",fmt='%i')

    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, num_trials, path):

        for t in range(num_trials):
            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, budget, t,path)

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, t, folderpath):
        
        file_name_parameter = "./" + al_strategy + "_" + classifier_name + "/Index_Claasifier_Trial" + str(t + 1) + ".txt"
        parameters = np.loadtxt(open(file_name_parameter,"rb"))

        rows = len(y_test)
        column = int(budget/step_size) + 1

        result_prediction = np.zeros(shape=(rows, column))
        result_probas = np.zeros(shape=(rows, column))

        pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        #Choosing strategy
        if al_strategy == 'qbc':
            active_s = QBCStrategy(classifier=classifier_name)
        elif al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        model = None
        labels = np.unique(y_pool)

        #Loop for prediction
        
        indicesInTrail = []
        accuracyInTrail = []

        ite = 0

        while len(trainIndices) < budget and len(pool) >= step_size:

            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:

                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            model = get_classifier(classifier_name, self.parameterslist[int(parameters[ite])])

            indicesInTrail.append(newIndices)

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)

            model = eval(model)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])
            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            tempaccuracy = accuracy_score(y_test,y_pred)
            accuracyInTrail.append(tempaccuracy)
            # Save the prediction and probas

            result_prediction[:, ite] = y_pred
            result_probas[:, ite] = y_probas[:, 0]

            ite = ite + 1

        file_name_indices = folderpath + "Indices_record_Trial" + "_" + str(t+1)
        write_integer(file_name_indices,indicesInTrail)
        file_name_accuracy = folderpath +"Accuracy_record_Trial" + "_" +str(t+1)
        write_float(file_name_accuracy,accuracyInTrail)

        accu = accuracy_score(y_test, result_prediction[:,column-2])
        print "This is the %r-th trial, the accuracy is %r" %(str(t+1),accu)

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        file_name_proba = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, result_probas, delimiter=",")

        file_name_prediction = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, result_prediction, delimiter=",")

class Best2Classifiers(object):

    def __init__(self, listofparameters, folderlist, trials, bootstrap_size,  step_size, budget, al_strategy, classifier_name):

        self.parameterslist = listofparameters
        self.folderlist = folderlist
        self.length = len(folderlist)
        self.trials = trials
        self.bootstrap_size = bootstrap_size
        self.step_size = step_size
        self.budget = budget
        self.iteration = (budget - bootstrap_size)/step_size + 1
        self.strategy = al_strategy
        self.classifier = classifier_name

        self.get_Best2Classifier()

    def get_Best2Classifier(self):

        for i in range(self.trials):
            temp = np.zeros((self.iteration,self.length))
            for j in range(self.length):
                filename = self.folderlist[j]+ "Accuracy_record_Trial_" + str(i+1) + ".txt"
                temp[:,j] = np.loadtxt(open(filename,"rb"),delimiter=',')

            accuracy = []
            for k in range(self.iteration):
                a = temp[k]
                first2largest = np.argsort(-a)[:2]
                accuracy.append(first2largest)

            file_index_classifier = "./" + self.strategy + "_" + self.classifier +"/Index_2Biggest_Claasifier_Trial" + str(i+1)
            np.savetxt("%s.txt" %file_index_classifier, accuracy, delimiter=",",fmt='%i')


    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, num_trials, path,lengthoftxt=None):

        for t in range(num_trials):
            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, budget, t,path, lengthoftxt=None)

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, t, folderpath,lengthoftxt=None):
        """Helper method for running multiple trials."""
        file_name_parameter =  "./" + al_strategy + "_" + classifier_name + "/Index_2Biggest_Claasifier_Trial" + str(t + 1) + ".txt"
        parameters = np.loadtxt(open(file_name_parameter,"rb"),delimiter=',')

        rows = len(y_test)
        column = int(budget/step_size) + 1

        result_prediction = np.zeros(shape=(rows, column))
        result_probas = np.zeros(shape=(rows, column))

        pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        # Choosing strategy
        if al_strategy == 'qbc':
            active_s = QBCStrategy(classifier=classifier_name)
        elif al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        model = None
        labels = np.unique(y_pool)

        # Loop for prediction
        ite = 0
        indicesInTrail = []
        accuracyInTrail = []
        lastaccuray = 0
        while len(trainIndices) < budget and len(pool) >= step_size:

            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:

                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            model = get_classifier(classifier_name,self.parameterslist[int(parameters[ite,0])])

            indicesInTrail.append(newIndices)

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)

            model = eval(model)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])
            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            tempaccuracy = accuracy_score(y_test,y_pred)

            if tempaccuracy < lastaccuray:
                model = get_classifier(classifier_name,self.parameterslist[int(parameters[ite,1])])
                model = eval(model)

                model.fit(X_pool[trainIndices], y_pool[trainIndices])
                y_probas = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
                tempaccuracy = accuracy_score(y_test,y_pred)

            lastaccuray = tempaccuracy

            accuracyInTrail.append(tempaccuracy)
            # Save the prediction and probas

            result_prediction[:,ite] = y_pred
            result_probas[:,ite] = y_probas[:,0]

            ite = ite + 1

        file_name_indices = folderpath + "Indices_record_Trial" + "_" + str(t+1)
        write_integer(file_name_indices,indicesInTrail)
        file_name_accuracy = folderpath +"Accuracy_record_Trial" + "_" +str(t+1)
        write_float(file_name_accuracy,accuracyInTrail)

        accu = accuracy_score(y_test, result_prediction[:,column-2])
        print "This is the %r-th trial, the accuracy is %r" %(str(t+1),accu)

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        file_name_proba = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, result_probas, delimiter=",")

        file_name_prediction = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, result_prediction, delimiter=",")

class BestAndRandom(object):

    def __init__(self, listofparameters, folderlist, trials, bootstrap_size,  step_size, budget, al_strategy, classifier_name):

        self.parameterslist = listofparameters
        self.folderlist = folderlist
        self.length = len(folderlist)
        self.trials = trials
        self.bootstrap_size = bootstrap_size
        self.step_size = step_size
        self.budget = budget
        self.iteration = (budget - bootstrap_size)/step_size + 1
        self.strategy = al_strategy
        self.classifier = classifier_name


    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, num_trials, path,lengthoftxt=None):

        for t in range(num_trials):
            print "trial", t+1
            self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, budget, t,path, lengthoftxt=None)

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, t, folderpath,lengthoftxt=None):
        """Helper method for running multiple trials."""
        file_name_parameter = "./" + al_strategy + "_" + classifier_name + "/Index_Claasifier_Trial" + str(t + 1) + ".txt"
        parameters = np.loadtxt(open(file_name_parameter,"rb"))


        rows = len(y_test)
        column = int(budget/step_size) + 1

        result_prediction = np.zeros(shape=(rows, column))
        result_probas = np.zeros(shape=(rows, column))

        pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        # Choosing strategy
        if al_strategy == 'qbc':
            active_s = QBCStrategy(classifier=classifier_name)
        elif al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        random_active = RandomStrategy(seed=t)

        model = None
        labels = np.unique(y_pool)

        # Loop for prediction
        ite = 0
        indicesInTrail = []
        accuracyInTrail = []
        lastaccuray = 0
        while len(trainIndices) < budget and len(pool) >= step_size:

            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:

                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            model = get_classifier(classifier_name,self.parameterslist[int(parameters[ite])])

            indicesInTrail.append(newIndices)

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)

            model = eval(model)

            model.fit(X_pool[trainIndices], y_pool[trainIndices])
            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            tempaccuracy = accuracy_score(y_test,y_pred)

            if tempaccuracy < lastaccuray:
                pool.update(newIndices)
                for aa in newIndices:
                    trainIndices.remove(aa)

                second_newIndice = random_active.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])
                pool.difference_update(second_newIndice)
                trainIndices.extend(second_newIndice)
                model.fit(X_pool[trainIndices], y_pool[trainIndices])
                y_probas = model.predict_proba(X_test)
                y_pred = model.predict(X_test)
                tempaccuracy = accuracy_score(y_test,y_pred)


            lastaccuray = tempaccuracy

            accuracyInTrail.append(tempaccuracy)
            # Save the prediction and probas

            result_prediction[:,ite] = y_pred
            result_probas[:,ite] = y_probas[:,0]

            ite = ite + 1

        file_name_indices = folderpath + "Indices_record_Trial" + "_" + str(t+1)
        write_integer(file_name_indices,indicesInTrail)
        file_name_accuracy = folderpath +"Accuracy_record_Trial" + "_" +str(t+1)
        write_float(file_name_accuracy,accuracyInTrail)

        accu = accuracy_score(y_test, result_prediction[:,column-2])
        print "This is the %r-th trial, the accuracy is %r" %(str(t+1),accu)

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        file_name_proba = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, result_probas, delimiter=",")

        file_name_prediction = folderpath + al_strategy + "_" + "Trial" + "_" + str(t+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, result_prediction, delimiter=",")