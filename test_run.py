#!/usr/bin/env python
# -*- coding: utf-8 -*-

from naive_bayes import NaiveBayes

def main():
    a = NaiveBayes()
    a.load('reuters.mat')
    a.ml_train()
    a.predict()
    print("Maximum Likelihood:")
#    print(a.accuracy())
    a.print_prediction()
    print()
    a.map_train()
    a.predict()
    print("Maximum A Posterior:")
#    print(a.accuracy())
    a.print_prediction()



if __name__ == '__main__':
    main()

