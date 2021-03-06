"""
The :mod:`al.instance_strategies` implements various active learning strategies.
"""
import math
import numpy as np
from collections import defaultdict

import scipy.sparse as ss
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

class RandomBootstrap(object):
    """Class - used if strategy selected is rand"""
    def __init__(self, seed):
        """Instantiate :mod:`al.instance_strategies.RandomBootstrap`

        **Parameters**
        * seed (*int*) - trial number."""

        self.randS = RandomStrategy(seed)

    def bootstrap(self, pool, y=None, k=1):
        """
        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * y - None or possible pool
        * k (*int*) - 1 or possible bootstrap size

        **Returns**
        * randS.chooseNext(pool, k=k) - choose next pool"""

        return self.randS.chooseNext(pool, k=k)

class BootstrapFromEach(object):
    """Class - used if not bootstrapped"""
    def __init__(self, seed):
        """Instantiate :mod:`al.instance_strategies.BootstrapFromEach`

        **Parameters**
        * seed (*int*) - trial number."""

        self.randS = RandomStrategy(seed)

    def bootstrap(self, pool, y, k=1):
        """
        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * y - None or possible pool
        * k (*int*) - 1 or possible bootstrap size

        **Returns**
        * chosen array of indices"""

        data = defaultdict(lambda: [])
        for i in pool:
            data[y[i]].append(i)
        chosen = []
        num_classes = len(data.keys())

        for label in data.keys():
            candidates = data[label]
            # k is bootstrap_size, num_classes is the # of labels, normalize for each
            indices = self.randS.chooseNext(candidates, k=k/num_classes)
            chosen.extend(indices)
        return chosen


class BaseStrategy(object):
    """Class - Base strategy"""
    def __init__(self, seed=0):
        """Instantiate :mod:`al.instance_strategies.BaseStrategy`

        **Parameters**
        * seed (*int*) - 0 or trial number."""

        self.randgen = np.random.RandomState(seed)

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        pass

class RandomStrategy(BaseStrategy):
    """Class - used if strategy is rand, inherits from :mod:`al.instance_strategies.BaseStrategy`"""
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        """Overide method BaseStrategy.chooseNext

        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * X - None or pool.toarray()
        * model - None
        * k (*int*) - 1 or step size
        * current_train_indices - None or array of trained indices
        * current_train_y - None or train_indices specific to y_pool

        **Returns**
        * [list_pool[i] for i in rand_indices[:k]] - array of random permutations given pool"""
        list_pool = list(pool)
        rand_indices = self.randgen.permutation(len(pool))
        return [list_pool[i] for i in rand_indices[:k]]

class UncStrategy(BaseStrategy):
    """Class - used if strategy selected is unc, inherits from :mod:`al.instance_strategies.BaseStrategy`"""
    def __init__(self, seed=0, sub_pool = None):
        """Instantiate :mod:`al.instance_strategies.UncStrategy`

        **Parameters**
        * seed (*int*) - 0 or trial number.
        * sub_pool - None or sub_pool parameter"""

        super(UncStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        """Overide method BaseStrategy.chooseNext

        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * X - None or pool.toarray()
        * model - None
        * k (*int*) - 1 or step size
        * current_train_indices - None or array of trained indices
        * current_train_y - None or train_indices specific to y_pool

        **Returns**
        * [candidates[i] for i in uis[:k]]"""
        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
        else:
            candidates = list(pool)

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()

        probs = model.predict_proba(X[candidates])
        uncerts = np.min(probs, axis=1)
        uis = np.argsort(uncerts)[::-1]
        chosen = [candidates[i] for i in uis[:k]]
        return chosen

class QBCStrategy(BaseStrategy):
    """Class - used if strategy selected is qbc, inherits from :mod:`al.instance_strategies.BaseStrategy`"""
    def __init__(self, classifier, classifier_args=None, seed=0, sub_pool = None, num_committee = 10):
        """Instantiate :mod:`al.instance_strategies.QBCStrategy`

        **Parameters**
        * classifier - Represents the classifier that will be used (default: MultinomialNB).
        * classifier_args - Represents the arguments that will be passed to the classifier (default: '').
        * seed (*int*) - 0 or trial number.
        * sub_pool - None or sub_pool parameter
        * num_committee - 4"""
        super(QBCStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool
        self.num_committee = num_committee
        self.classifier = classifier
        self.classifier_args = classifier_args


    def vote_entropy(self, sample):
        """ Computes vote entropy.
        **Parameters**
        * sample

        **Returns**
        * out (*int*)"""
        votes = defaultdict(lambda: 0.0)
        size = float(len(sample))

        for i in sample:
            votes[i] += 1.0

        out = 0
        for i in votes:
            aux = (float(votes[i]/size))
            out += ((aux*math.log(aux, 2))*-1.)

        return out

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices = None, current_train_y = None):
        """Overide method BaseStrategy.chooseNext

        **Parameters**
        * pool (*int*) - range of numbers within length of pool
        * X - None or pool.toarray()
        * model - None
        * k (*int*) - 1 or step size
        * current_train_indices - None or array of trained indices
        * current_train_y - None or train_indices specific to y_pool

        **Returns**
        * [candidates[i] for i in dis[:k]]"""

        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
        else:
            candidates = list(pool)

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()

        # Create bags
        comm_predictions = []

        for c in range(self.num_committee):
            # Make sure that we have at least one of each label in each bag
            bfe = BootstrapFromEach(seed=c)
            num_labels = len(np.unique(current_train_y))
            initial = bfe.bootstrap(range(len(current_train_indices)), current_train_y, num_labels)
            
            r_inds = self.randgen.randint(0, len(current_train_indices), size=len(current_train_indices)-num_labels)            
            r_inds = np.hstack((r_inds, np.array(initial)))
                        
            bag = [current_train_indices[i] for i in r_inds]
            bag_y = [current_train_y[i] for i in r_inds]
            new_classifier = eval(self.classifier)
            new_classifier.fit(X[bag], bag_y)
            predictions = new_classifier.predict(X[candidates])
            comm_predictions.append(predictions)

        # Compute disagreement for com_predictions

        disagreements = []
        for i in range(len(comm_predictions[0])):
            aux_candidates = []
            for prediction in comm_predictions:
                aux_candidates.append(prediction[i])
            disagreement = self.vote_entropy(aux_candidates)
            disagreements.append(disagreement)

        dis = np.argsort(disagreements)[::-1]
        chosen = [candidates[i] for i in dis[:k]]

        return chosen
