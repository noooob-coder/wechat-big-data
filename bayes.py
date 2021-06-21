#!/usr/bin/python
# coding=utf-8
#!/usr/bin/python
# coding=utf-8

import numpy
import random
import scipy.special as special
import pandas as pd


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for clk_rt in sample:
            imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)
def data_load():
    ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
    for action in ACTION_LIST:
        for item in ['feedid','userid']:
            path=f'{item}ctr_data_for_{action}.csv'
            df=pd.read_csv(path)
            c=item+action+'1'
            i=item+action
            # print(df)
            I=numpy.array(df[i])
            C=numpy.array(df[c])
            # print(I)
            # print(C)
            bs = BayesianSmoothing(1, 1)
            bs.update(I,C,len(I), 0.0000000001)
            df['bayes']=(df[c]+bs.alpha)/(df[i]+bs.alpha+bs.beta)
            df.to_csv(f'bayes_{item}ctr_data_for_{action}.csv')
data_load()


