
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np

path = sys.argv[1]
output_name = sys.argv[2]
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


true_k = 100

dataset = []
f = open(path + 'title_StackOverflow.txt','r')
for line in f:
  dataset.append(line)

#print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset)

#print("done in %fs" % (time() - t0))
#print("n_samples: %d, n_features: %d" % X.shape)
#print()

t0 = time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

#print("done in %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
#print("Explained variance of the SVD step: {}%".format(
 #   int(explained_variance * 100)))

print()

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=20,
                verbose=False)

#print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
Y = km.predict(X)
test = np.genfromtxt(path + "check_index.csv",delimiter = ",")
test = np.delete(test,0,0)
test = np.delete(test,0,1)

output = open(output_name,'w')
output.write("ID,Ans\n")

for i in range(len(test)):
#  if i % 10000 == 0 :
#    print (i)
  A = Y[int(test[i][0])]
  B = Y[int(test[i][1])]
  if A == B : 
    output.write(str(i) + ',1\n')
  else:
    output.write(str(i) + ',0\n') 

