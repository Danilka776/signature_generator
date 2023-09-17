import time
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from main import ShortPatternMiner, get_replace
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pylab import rc, plot
import itertools

with open("dataset_1670944872.114844.csv", "r") as data:
    df = pd.read_csv(data)
df1 = df.drop(columns=['anomalous'], axis=1)
df1 = shuffle(df1)
train_df = df1.iloc[[i for i in range(500)]]
test_df = df1.iloc[[i for i in range(501, 1000)]]

available_coarse_clusterization = ["protocol", "method", "raw_uri", "x-location", "src_port", "src_ip"]
available_string_distance = ["Levenshtein", "MetricLCS", "JaroWinkler", "Cosine"]
available_clusterization = ["kmedoids", "hierarchical"]

dict_test = []
for dist in available_string_distance:
    for coars in available_coarse_clusterization:
        for clust in available_clusterization:
            try:
                a = ShortPatternMiner(coverage=0.1, coarse_clusterization=[coars], string_distance=dist, clusterization=clust, max_clusters=10)
                start = time.time()
                dict_test.append(a.generate_pattern(train_df))
                end = time.time()
                print("time on", coars, clust, dist, "is - ", end - start)
            except Exception:
                print(coars, clust, dist, "was COLAPSED")

df1 = shuffle(df)
test_df1 = df1.iloc[[i for i in range(500, 1000)]]
y_true = [int(i) for i in test_df1["anomalous"].to_list()]
y_true

strs = test_df1['body'].to_list()
res = []
m = 0
m_arr = []
m_i = 0
for j in range(len(dict_test)):
    for i in range(len(dict_test[j]['body'])):
        res = []
        pattern = get_replace(dict_test[j]['body'][i])
        for s in strs:
            if re.search(pattern, s):
                res.append(1)
            else:
                res.append(0)
        cm = confusion_matrix(y_true, res)
        tn, fp, fn, tp = cm.ravel()
        accur = (tp + tn) / (tp + tn + fp + fn)
        if m < accur:
            m = accur
            m_i = i
    m_arr.append([m, m_i])
    print(m, m_i)


y_pred_class = res
cm = confusion_matrix(y_true, y_pred_class)
tn, fp, fn, tp = cm.ravel()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


font = {'size' : 15}

plt.rc('font', **font)

plt.figure(figsize=(10, 8))
plot_confusion_matrix(cm, classes=['Non-marked', 'Marked'],
                      title='Confusion matrix')
plt.savefig("conf_matrix.png")
plt.show()
