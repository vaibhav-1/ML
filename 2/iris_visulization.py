import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris as iris
import itertools
data = iris()
#sice iris is a numpy array using accordingly
feature = data.data
feature_name = data.feature_names
target = data.target
target_names = data.target_names
slength = feature[:,0]
plength = feature[:,2]
swidth = feature[:,1]
pwidth =feature[:,3]
plt.figure(2,figsize=(8,6))
plt.clf()
fig,axes = plt.subplots(2,3)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
color_markers = [
        ('r', '>'),
        ('g', 'o'),
        ('b', 'x'),
        ]
labels = target_names[target]

#code for graph craetion
"""for i ,(p0,p1) in enumerate(pairs):
    ax = axes.flat[i]
    for t in range(3):
        c,marker = color_markers[t]
        ax.scatter(feature[target==t,p0],feature[target==t,p1],marker=marker,c=c)
        ax.set_xlabel(feature_name[p0])
        ax.set_ylabel(feature_name[p1])
        ax.set_xticks([])
        ax.set_yticks([])
fig.savefig('fig1.png')"""
is_setosa = (labels == 'setosa')
max_setosa = plength[is_setosa].max()
#fancy array addressing of numpy
min_non_setosa =plength[~is_setosa].min()
feature = feature[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels=='viginica')

best_acc=-1
for fi in range(feature.shape[1]):
    thresh = feature[:,fi]
    for t in thresh:
        feature_i= feature[:,fi].copy()
        pred =(feature_i>t)
        acc =(pred==is_virginica).mean()
        rev_acc = (pred == ~is_virginica).mean()
        if rev_acc>acc:
            reverse = True
            acc =rev_acc
        else:
            reverse=False
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse
def is_virginica_test(fi, t, reverse, example):
    'Apply threshold model to a new example'
    test = example[fi] > t
    if reverse:
        test = not test
    return test
#from threshold import fit_model, predict

# ning accuracy was 96.0%.
# ing accuracy was 90.0% (N = 50).
