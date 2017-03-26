# _*_ coding: utf-8 _*_
import numpy as np
import pandas as pd

# Data Cleanup

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
# The rank feature is categorical, the numbers don't encode any sort of
# relative values.
data = pd.concat([admissions, pd.get_dummies(admissions['rank'],
                                             prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standardize the GRE and GPA
# We'll also need to standardize the GRE and GPA data, which means to scale
# the values such they have zero mean and a standard deviation of 1.
#  == 평균 0이고 표준편차 1인 표준정규분포(standard normal distribution) 구하기.
# This is necessary because the sigmoid function squashes really small and
# really large inputs. The gradient of really small and large inputs is zero,
# which means that the gradient descent step will go to zero too.
# Since the GRE and GPA values are fairly large, we have to be really careful
# about how we initialize the weights or the gradient descent steps will die
# off and the network won't train. Instead, if we standardize the data, we can
# initialize the weights easily and everyone is happy.
for field in ['gre', 'gpa']:
    # 평균
    mean = data[field].mean()
    print("mean=", mean)
    # 표준편차
    std = data[field].std()
    print("standard deviation=", std)
    # Mahalanobis distance
    # Mahalanobis distance는 어떤 값이 얼마나 일어나기 힘든 값인지,
    # 또는 얼마나 이상한 값인지를 수치화하는 한 방법이다.
    # http://darkpgmr.tistory.com/41
    # http://math7.tistory.com/47
    data.loc[:, field] = (data[field] - mean)/std

# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)
print("sample data size = ", len(test_data))

# Split into features and targets
# original data
features = data.drop('admit', axis=1)
targets = data['admit']
# sample test data
features_test= test_data.drop('admit', axis=1)
targets_test = test_data['admit']
