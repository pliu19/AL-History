__author__ = 'Ping'
import glob
import numpy as np
import scipy.sparse as ss
import argparse
from time import time

from sklearn import metrics
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from strategies import LogGainStrategy, RandomStrategy, UncStrategy, BootstrapFromEach, QBCStrategy, ErrorReductionStrategy

class LearningCurve(object):
    """Class - run multiple trials or run trials one at a time"""
    def run_trials(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, num_trials):

        self.history_probas = { }
        self.history_preditcion = { }

        for t in range(num_trials):
            print "trial", t
            self.history_preditcion[t],self.history_probas[t] = self._run_a_single_trial(X_pool, y_pool, X_test, y_test,al_strategy, classifier_name, bootstrap_size,  step_size, budget, t)

        return self.history_preditcion, self.history_probas

    def _run_a_single_trial(self, X_pool, y_pool, X_test, y_test, al_strategy, classifier_name, bootstrap_size,  step_size, budget, t):
        """Helper method for running multiple trials."""

        rows = 25000
        column = int(budget/step_size) + 1

        result_prediction = np.zeros(shape=(rows, column))
        result_probas = np.zeros(shape=(rows, column))

        if len(y_pool) > 10000:
            rs = np.random.RandomState(t)
            indices = rs.permutation(len(y_pool))
            pool = set(indices[:10000])
        else:
            pool = set(range(len(y_pool)))

        trainIndices = []
        bootstrapped = False

        #Choosing strategy
        if al_strategy == 'erreduct':
            active_s = ErrorReductionStrategy(classifier=classifier_name, seed=t)
        elif al_strategy == 'loggain':
            active_s = LogGainStrategy(classifier=classifier_name, seed=t)
        elif al_strategy == 'qbc':
            active_s = QBCStrategy(classifier=classifier_name)
        elif al_strategy == 'rand':
            active_s = RandomStrategy(seed=t)
        elif al_strategy == 'unc':
            active_s = UncStrategy(seed=t)

        model = None

        labels = np.unique(y_pool)

        #Loop for prediction

        ite = 0
        while len(trainIndices) < budget and len(pool) >= step_size:
            if not bootstrapped:
                boot_s = BootstrapFromEach(t)
                newIndices = boot_s.bootstrap(pool, y=y_pool, k=bootstrap_size)
                bootstrapped = True
            else:
                newIndices = active_s.chooseNext(pool, X_pool, model, k=step_size, current_train_indices = trainIndices, current_train_y = y_pool[trainIndices])

            pool.difference_update(newIndices)
            trainIndices.extend(newIndices)
            model = eval(classifier_name)
            model.fit(X_pool[trainIndices], y_pool[trainIndices])

            # Prediction
            y_probas = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            # Save the prediction and probas
            for num in range(rows):
                result_prediction[num][ite] = y_pred[num]
                result_probas[num][ite] = y_probas[num][0]

            ite = ite + 1

        for num in range(rows):
            result_prediction[num][ite] = y_test[num]
            result_probas[num][ite] = y_test[num]

        return result_prediction, result_probas

def load_imdb(path, shuffle=True, random_state=42, \
              vectorizer = CountVectorizer(min_df=5, max_df=1.0, binary=True)):

    print "Loading the imdb reviews data"

    train_neg_files = glob.glob(path+"/train/neg/*.txt")
    train_pos_files = glob.glob(path+"/train/pos/*.txt")

    train_corpus = []
    y_train = []

    for tnf in train_neg_files:
        f = open(tnf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(0)
        f.close()

    # print train_corpus

    for tpf in train_pos_files:
        f = open(tpf, 'r')
        line = f.read()
        train_corpus.append(line)
        y_train.append(1)
        f.close()

    test_neg_files = glob.glob(path+"/test/neg/*.txt")
    test_pos_files = glob.glob(path+"/test/pos/*.txt")

    test_corpus = []

    y_test = []

    for tnf in test_neg_files:
        f = open(tnf, 'r')
        test_corpus.append(f.read())
        y_test.append(0)
        f.close()

    for tpf in test_pos_files:
        f = open(tpf, 'r')
        test_corpus.append(f.read())
        y_test.append(1)
        f.close()

    print "Data loaded."

    print "Extracting features from the training dataset using a sparse vectorizer"
    print "Feature extraction technique is %s." % vectorizer
    t0 = time()

    X_train = vectorizer.fit_transform(train_corpus)


    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_train.shape
    print

    print "Extracting features from the test dataset using the same vectorizer"
    t0 = time()

    X_test = vectorizer.transform(test_corpus)
    # print type(X_test)

    duration = time() - t0
    print("done in %fs" % (duration))
    print "n_samples: %d, n_features: %d" % X_test.shape
    print

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))

        X_train = X_train.tocsr()
        X_train = X_train[indices]
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]

        indices = np.random.permutation(len(y_test))

        X_test = X_test.tocsr()
        X_test = X_test[indices]
        y_test = y_test[indices]

        test_corpus_shuffled = [test_corpus[i] for i in indices]
    else:
        train_corpus_shuffled = train_corpus
        test_corpus_shuffled = test_corpus

    return X_train, y_train, X_test, y_test, train_corpus_shuffled, test_corpus_shuffled

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-path', help='The path to the content file.')

    parser.add_argument('-classifier',choices=['LogisticRegression()','MultinomialNB()'], default='MultinomialNB()',
                        help='The underlying classifier.')
    parser.add_argument("-nt", "--num_trials", type=int, default=10, help="Number of trials (default: 10).")

    parser.add_argument('-num_folds', type=int, default=10, help='The number of folds.')

    parser.add_argument("-f", '--file', type=str, default="aaa.aaa",
                        help='This feature represents the name that will be written with the result. '
                             'If it is left blank, the file will not be written (default: None ).')
    parser.add_argument("-st", "--strategies", choices=['erreduct', 'loggain', 'qbc', 'rand','unc'], nargs='*',default='unc',
                        help="Represent a list of strategies for choosing next samples (default: rand).")
    parser.add_argument("-bs", '--bootstrap', default=10, type=int,
                        help='Sets the Boot strap (default: 10).')
    parser.add_argument("-b", '--budget', default=500, type=int,
                        help='Sets the budget (default: 500).')
    parser.add_argument("-sz", '--stepsize', default=10, type=int,
                        help='Sets the step size (default: 10).')
    parser.add_argument("-sp", '--subpool', default=None, type=int,
                        help='Sets the sub pool size (default: None).')
    args = parser.parse_args()

    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))
    X_tr, y_tr, X_te, y_te, tr_corp, te_corp = load_imdb("C:\\Users\\Ping\\Desktop\\aclImdb", vectorizer=vect)

    learning_api = LearningCurve()
    his_predition, his_probal = learning_api.run_trials(X_tr, y_tr, X_te, y_te, args.strategies, args.classifier, args.bootstrap, args.stepsize, args.budget, args.num_trials)

    for i in his_probal.keys():
        file_name_proba = args.strategies + "_" + "Trial" + "_" + str(i+1) + "_proba"
        np.savetxt("%s.csv" %file_name_proba, his_probal[i], delimiter=",")

    for i in his_predition.keys():
        file_name_prediction = args.strategies + "_" + "Trial" + "_" + str(i+1) + "_prediction"
        np.savetxt("%s.csv" %file_name_prediction, his_predition[i], delimiter=",")