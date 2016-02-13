#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio


class NaiveBayes(object):

    train = None
    trainY = None
    test = None
    testY = None
    _prior = None
    _likelihood = None
    _prediction = None


    """
    load data from .mat file, assuming 'train', 'trainy', 'test', 'testy'
    are in the file.

    @param: {string} path
    @return: None
    """
    def load(self, path):
        data = sio.loadmat(path)
        self.train = data['train']
        self.trainY = data['trainy']
        self.test = data['test']
        self.testY = data['testy']


    """
    use maximum likelihood estimation to train conditional probabilities

    @param: None
    @return: None
    """
    def ml_train(self):
        # class 0 indices
        Y0_indices = np.where(self.trainY == 0)[0]
        # class 1 indices
        Y1_indices = self.trainY.nonzero()[0]
        self._prior = [Y0_indices.size/self.trainY.size,
                       Y1_indices.size/self.trainY.size]
        self._likelihood = [
            # treat count>0 as 1, count=0 as 0
            self.train[Y0_indices, :].getnnz(0)/Y0_indices.size,
            self.train[Y1_indices, :].getnnz(0)/Y1_indices.size,
        ]


    """
    use maximum a posterior estimation and take dirichlet distribution as
    conjugate prior, to train conditional probabilities

    @param: None
    @return: None
    """
    def map_train(self):
        # class 0 indices
        Y0_indices = np.where(self.trainY == 0)[0]
        # class 1 indices
        Y1_indices = self.trainY.nonzero()[0]
        alpha = 1.02
        self._prior = [Y0_indices.size/self.trainY.size,
                       Y1_indices.size/self.trainY.size]
        self._likelihood = [
            (self.train[Y0_indices, :].getnnz(0)+alpha-1)/(Y0_indices.size+2*alpha-2),
            (self.train[Y1_indices, :].getnnz(0)+alpha-1)/(Y1_indices.size+2*alpha-2),
        ]


    """
    compute predictions and store them in an array with 1/0 value

    @param: None
    @return: None
    """
    def predict(self):
        # convert to zero/one matrix
        self.train = self.train.astype('bool')
        posterior = [
            # compute all posteriors=likelihood*prior of class 0
            np.array([
                np.multiply(
                    np.power(self._likelihood[0], self.test[i, :].toarray()),
                    np.power(1-self._likelihood[0], 1-self.test[i, :].toarray()),
                ).prod()
                for i in range(self.test.shape[0])
            ]) * self._prior[0],
            # compute all posteriors=likelihood*prior of class 1
            np.array([
                np.multiply(
                    np.power(self._likelihood[1], self.test[i, :].toarray()),
                    np.power(1-self._likelihood[1], 1-self.test[i, :].toarray()),
                ).prod()
                for i in range(self.test.shape[0])
            ]) * self._prior[1],
        ]

        self.prediction = posterior[0].__lt__(posterior[1]).astype('int')


    """
    compute accuracy of the predictions

    @param: None
    @return: {double} accu
    """
    def accuracy(self):
        accu = np.count_nonzero(self.prediction.__eq__(self.testY.T).astype('int')) / self.testY.size
        return accu


    """
    print predictions, one label per line

    @param: None
    @return: None
    """
    def print_prediction(self):
        for x in np.nditer(self.prediction):
            print(x, end='\n')
