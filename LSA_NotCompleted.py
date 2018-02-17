# coding: utf-8


from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.spatial.distance import *
import io
import matplotlib.pyplot as plt
import numpy
from time import time
from pprint import pprint


def get_unigrams(file_name):
    unigrams = {}
    with io.open(file_name, encoding='utf8', errors='ignore') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                token = token.lower()
                try:
                    unigrams[token]
                except:
                    unigrams[token] = 0
                unigrams[token] += 1

    return unigrams


def index_unigrams(unigrams):
    new_unigrams = {}
    reverse_unigrams = {}
    for index, unigram in enumerate(unigrams):
        new_unigrams[unigram] = index
        reverse_unigrams[index] = unigram
    return new_unigrams, reverse_unigrams


file_name = "sample_corpus"
unigrams = get_unigrams(file_name)
print 'UNIGRAMS'
print unigrams
iunigrams, runigrams = index_unigrams(unigrams)
print 'IUNIGRAMS'
print iunigrams
print 'RUNIGRAMS'
print runigrams
unigrams = sorted(unigrams.items(), key=lambda x: x[1], reverse=True)
print 'UNIGRAMS'
print unigrams

# pprint.pprint(iunigrams) # Figure out non-stop words
dimensions = [x[0] for x in unigrams[100:3100]]
print '---DIMENSIONS---'
print dimensions
idimensions = {x: index for index, x in enumerate(dimensions)}
# pprint(dimensions)


cmatrix = numpy.memmap("lsa.cmatrix", dtype='float32', mode='w+', shape=(len(unigrams), len(dimensions)))
print(cmatrix.shape)


def populate_cmatrix(file_name, cmatrix, iunigrams, idimensions, window=5):
    e = 0
    s = 0
    with io.open(file_name, encoding='utf-8', errors='ignore') as f:
        for index, line in enumerate(f):
            tokens = line.strip().split()
            for indexj, token in enumerate(tokens):
                token = token.lower()
                lcontext = tokens[(indexj - window):(indexj)]
                rcontext = tokens[indexj + 1:index + window]
                context = [tok.lower() for tok in lcontext + rcontext]

                try:
                    # print 'in try'
                    unigram_index = iunigrams[token]
                    for d in context:
                        # print 'd->',d
                        if d in idimensions:
                            j = idimensions[d]
                            # print j
                            cmatrix[unigram_index][j] += 1
                            s += 1
                except:
                    # print 'in except'
                    e += 1

    print(e, s)


start_time = time()
populate_cmatrix(file_name, cmatrix, iunigrams, idimensions)
end_time = time()
print(end_time - start_time)

w1 = 'eat'
w2 = 'drink'
w3 = 'print'
id1 = iunigrams[w1]
id2 = iunigrams[w2]
id3 = iunigrams[w3]

# Store all word ids of words
id = []
for i, j in iunigrams.items():
    id.append(j)

print id1, id2, id3
v1 = cmatrix[id1]
v2 = cmatrix[id2]
v3 = cmatrix[id3]

print v1, v2, v3

print(euclidean(v1, v2))
print(cosine(v2, v3))
print(cosine(v1, v3))

start_time = time()
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(cmatrix)
twod_cmatrix = svd.transform(cmatrix)
end_time = time()
print(end_time - start_time)

v1_2d, v2_2d = twod_cmatrix[id1], twod_cmatrix[id2]
id3 = iunigrams[w3]
v3_2d = twod_cmatrix[id3]
print(v1_2d, v2_2d, v3_2d)
print(cosine(v1_2d, v2_2d), cosine(v1_2d, v3_2d))

# get_ipython().magic(u'pylab inline')

# baks implement v_2d
v_2d = []
mapp = {}

for i in range(len(id)):
    # print id[i]
    v_2d.append(twod_cmatrix[id[i]])
    # print twod_cmatrix[id[i]]
    # print runigrams[id[i]]
    mapp[runigrams[id[i]]] = twod_cmatrix[id[i]]

np_v_2d = numpy.array(v_2d)
# print np_v_2d
kmeans = KMeans(n_clusters=20, random_state=1337).fit(np_v_2d)

print 'labels->'

final_mapping = {}
for i in range(kmeans.labels_.shape[0]):
    # print kmeans.labels_[i]
    if kmeans.labels_[i] not in final_mapping:
        final_mapping[kmeans.labels_[i]] = [mapp.keys()[i]]
    else:
        final_mapping[kmeans.labels_[i]].append(mapp.keys()[i])

print final_mapping


for key in final_mapping.keys():
    print key
    with open(str(key)+'.txt','a') as f:
        # f.write(str(key)+'\n')
        print len(final_mapping[key])
        val = (','.join(final_mapping[key])).encode('ascii','ignore')
        f.write(val + '\n')
    f.close()

v1_2d = v1_2d / numpy.linalg.norm(v1_2d)
v2_2d = v2_2d / numpy.linalg.norm(v2_2d)
v3_2d = v3_2d / numpy.linalg.norm(v3_2d)
print ([v1_2d, v2_2d, v3_2d])
colors = ['r', 'b', 'g']
fig, axs = plt.subplots(1, 1)
for i, x in enumerate([v1_2d, v2_2d, v3_2d]):
    a = plt.plot([0, x[0]], [0, x[1]], colors[i] + '-')
plt.show()
