#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from math import ceil, floor

#%% Load data

fnames = ['car', 'dancing', 'running', 'stairs', 'walking']
column_names = ['acc_x', 'acc_y', 'acc_z']
k = 10

dataframes = []
for i, fname in enumerate(fnames):
    fpath = os.path.join('data', f'{fname}.csv')
    tmp = pd.read_csv(
        fpath, usecols = column_names, skiprows = lambda x : x > 18000)
    tmp.insert(3, "activity", i, True)
    dataframes.append(tmp)
df = pd.concat(dataframes)

#%% Extract features

def extract_features(data, time_step = 0.01, frame_size = 1, overlap = 0.5, test_size = 0.2):
    activities = np.unique(data['activity'])
    features = []
    for act in activities:
        acc = data[data['activity'] == act].iloc[:,:-1]
        step_size = ceil(frame_size*(1-overlap)/time_step)
        fsize = int(frame_size/time_step)
        i = 0; idx = np.array([0, fsize-1], dtype=int)
        T = len(acc)
        num_samples = floor((T - fsize)/step_size) + 1
        ft = np.zeros((num_samples, 3*3+1)) # (mean, std, ZCR) \times (X, Y, Z)
        while idx[1] < T:
            frame = acc.iloc[idx[0]:idx[1],:]
            ft[i,0:3] = np.mean(frame)
            ft[i,3:6] = np.std(frame)
            ft[i,6:9] = np.sum(np.diff(np.sign(frame), axis = 0) > 0, axis = 0)
            i += 1
            idx += step_size
        ft[:,-1] = act
        features.append(ft)
    features = np.concatenate(features)
    x_train, x_test, y_train, y_test = train_test_split(
        features[:,:-1], features[:,-1], test_size=test_size, random_state=24)
    return x_train, x_test, y_train, y_test


#%% Frame size selection

frame_sizes = np.linspace(1,25,20)
scores = np.zeros((len(frame_sizes), 2))
for i, fs in enumerate(frame_sizes):
    X, X_test, Y, Y_test = extract_features(df, frame_size = fs)
    clf = svm.SVC(kernel = 'linear', C = 0.01, decision_function_shape = 'ovr')
    gnb = GaussianNB(var_smoothing = 0.04)
    clf_scores = cross_val_score(clf, X, Y, cv=k)
    gnb_scores = cross_val_score(gnb, X, Y, cv=k)
    scores[i,:] = [clf_scores.mean(), gnb_scores.mean()]

plt.plot(frame_sizes, scores[:,0], label = 'SVM')
plt.plot(frame_sizes, scores[:,1], label = 'GNB')
plt.xlabel('Frame size [s]')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% Feature Extraction

fs = 10
X, X_test, Y, Y_test = extract_features(df, frame_size = fs)
X2, X_val, Y2, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=42)

#%% Parameter Tuning and Validation

# SVM
C = np.power(np.linspace(2e-2, 2e-1, num=k), 2)
SVM_cv_scores = np.zeros(k*k)
SVM_split_scores = np.zeros(k)
plot_c = np.repeat(C, k)
for i, c in enumerate(C):
    clf = svm.SVC(kernel = 'linear', C = c, decision_function_shape = 'ovr')

    # 10-fold cross-validation
    cv_scores = cross_val_score(clf, X, Y, cv=k)
    SVM_cv_scores[i*k:i*k+k] = cv_scores

    # Split test/train
    clf.fit(X2, Y2)
    Y_pred = clf.predict(X_val)
    SVM_split_scores[i] = accuracy_score(Y_val, Y_pred)

ax = sns.relplot(x = plot_c, y = SVM_cv_scores, kind='line')
plt.plot(C,SVM_split_scores, 'orange', linewidth = 1.5)
plt.xlabel("$C$")
plt.ylabel("Accuracy")
ax.set(ylim = (0.85, 1))
ax.set(xlim = (C[0], C[-1]))
plt.legend(['10-fold', 'test/train'])
plt.grid()
avg_SVM_cv_scores = SVM_cv_scores.reshape((k,k)).mean(axis = 1)
C = C[np.argmax(avg_SVM_cv_scores)]

# Gaussian Naive-Bayes
var_smoothing = np.power(np.linspace(1e-9, 0.3, num=k), 2)
GNB_cv_scores = np.zeros(k*k)
GNB_split_scores = np.zeros(k)
plot_v = np.repeat(var_smoothing, k)
for i, v in enumerate(var_smoothing):
    gnb = GaussianNB(var_smoothing = v)

    # 10-fold cross-validation
    cv_scores = cross_val_score(gnb, X, Y, cv=k)
    GNB_cv_scores[i*k:i*k+k] = cv_scores

    # Split test/train
    gnb.fit(X2, Y2)
    Y_pred = gnb.predict(X_val)
    GNB_split_scores[i] = accuracy_score(Y_val, Y_pred)

ax = sns.relplot(x = plot_v, y = GNB_cv_scores, kind='line')
plt.plot(var_smoothing, GNB_split_scores, 'orange', linewidth = 1.5)
plt.xlabel("Var. Smoothing")
plt.ylabel("Accuracy")
plt.legend(['10-fold', 'test/train'])
plt.grid()
ax.set(ylim = (0.95, 1))
ax.set(xlim = (1e-9, var_smoothing[-1]))
avg_GNB_cv_scores = GNB_cv_scores.reshape((k,k)).mean(axis = 1)
V = var_smoothing[np.argmax(avg_GNB_cv_scores)]

print(f'SVM C param = {C}')
print(f'GNB var. smoothing param = {V}')
# %% Test
clf = svm.SVC(kernel = 'linear', C = 0.01, decision_function_shape = 'ovr')
clf.fit(X, Y)
Y_pred = clf.predict(X_test)
clf_test_score = accuracy_score(Y_test, Y_pred)

gnb = GaussianNB(var_smoothing = 0.04)
gnb.fit(X, Y)
Y_pred = gnb.predict(X_test)
gnb_test_score = accuracy_score(Y_test, Y_pred)

print('Test Scores')
print(f'SVM: {clf_test_score}')
print(f'GNB: {gnb_test_score}')
# %% Confusion Matrix

plot_confusion_matrix(clf, X_test, Y_test,
                        display_labels=fnames,
                        cmap=plt.cm.Blues,
                        normalize = 'true')
plot_confusion_matrix(gnb, X_test, Y_test,
                        display_labels=fnames,
                        cmap=plt.cm.Blues,
                        normalize = 'true')

# %%
