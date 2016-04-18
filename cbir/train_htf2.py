import os
import django

# import project settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'cbirproject.settings'
django.setup()

# file dependencies
import re
import numpy as np
import cv2
import itertools
from sklearn import preprocessing
from cbir.cvclasses.searcher import Searcher
from skimage.feature import greycomatrix, greycoprops

from cbir.models import Image


def euclidean_distance(a, b):
    return np.linalg.norm(a-b)


def get_htf_features(image, props):
    features = []
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = greycomatrix(grey, [1], [0, 90, 45, 135], 256, symmetric=True, normed=True)
    for p in props:
        if p == 'entropy':
            entropy = np.apply_over_axes(np.sum, g * (-np.log1p(g)), axes=(0, 1))[0, 0]
            eu = np.mean(entropy)
            er = np.ptp(entropy)
            features.append([eu, er])
        else:
            gp = greycoprops(g, p)
            u = np.mean(gp)
            r = np.ptp(gp)
            features.append([u, r])

    features = np.array(features, dtype=np.float64)

    if len(props) != 1:
        for (x, y), value in np.ndenumerate(features):
            features[x, y] = (value - np.mean(features[:,y]))/np.std(features[:,y])

    return features.flatten()


def get_htf_matches(query, props, files):
    q = get_htf_features(cv2.imread(query.file.path), props)
    htf_matches = {}
    for file in files:
        f = get_htf_features(cv2.imread(file.path), props)
        htf_matches[str(file)] = euclidean_distance(q, f)

    return htf_matches


def combine(a, b, c, w):
    matches = {}

    # split dictionaries into keys and values
    al = [x for x in a.items()]
    ak, av = zip(*al)
    bl = [x for x in b.items()]
    bk, bv = zip(*bl)
    cl = [x for x in c.items()]
    ck, cv = zip(*cl)

    # scale the values in the range 0-1
    a_scaled = preprocessing.minmax_scale(av, feature_range=(0, 1))
    b_scaled = preprocessing.minmax_scale(bv, feature_range=(0, 1))
    c_scaled = preprocessing.minmax_scale(cv, feature_range=(0, 1))

    # build numpy structured arrays combining scaled values and original keys
    names = ['keys', 'values']
    formats = ['S225', 'f8']
    dtype = dict(names=names, formats=formats)
    anp = np.array(list(zip(ak,a_scaled)), dtype=dtype)
    bnp = np.array(list(zip(bk,b_scaled)), dtype=dtype)
    cnp = np.array(list(zip(ck,c_scaled)), dtype=dtype)

    # iterate over numpy structures creating a weighted average between values with the same key
    for i, t1 in np.ndenumerate(anp):
        for j, t2 in np.ndenumerate(bnp):
            if anp['keys'][i] == bnp['keys'][j]:
                for k, t3 in np.ndenumerate(cnp):
                    if anp['keys'][i] == cnp['keys'][k]:
                        stack = np.vstack((anp['values'][i], bnp['values'][j]))
                        stack = np.vstack((stack, cnp['values'][k]))
                        matches[anp['keys'][i].decode("utf-8")] = np.average(stack, axis=0, weights=w)[0]
                        break
                break

    return matches

s_group = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy']
combinations = []

for s in range(len(s_group)+1):
    for c in itertools.combinations(s_group, s):
        if len(c) != 0 and len(c) <= (len(s_group)+1):
            combinations.append(list(c))

classes = list(range(1,11))
ranges = {
    '1': (0,100),
    '2': (100,200),
    '3': (200,300),
    '4': (300,400),
    '5': (400,500),
    '6': (500,600),
    '7': (600,700),
    '8': (700,800),
    '9': (800,900),
    '10': (900,1000)
}
# w_range = [x / 10.0 for x in range(0, 11, 1)]
# weights = [list(x) for x in list(itertools.product(w_range, w_range, w_range)) if x[0] + x[1] + x[2] == 1]
w = [.4,.5,.1]
db = Image.objects.all()
files = []
for instance in db:
    files.append(instance.file)

precision_per_comb = []
for props in combinations:
    precision_per_class = []
    precision_hist = []
    count = 0
    for m, query in enumerate(db):   # m axis
        Tp = 0
        Fp = 0

        s = Searcher(query)
        c = s.colour()
        t = s.lbpatterns()
        h = get_htf_matches(query, props, files)
        matches = combine(c, t, h, w)

        dict_sorted = sorted([(v, k) for (k, v) in matches.items()])

        best_matches = []
        for d, path in dict_sorted[:8]:
            num = int(re.findall('\d+', path)[0])
            for ke, va in ranges.items():
                if num in range(va[0], va[1]):
                    best_matches.append([d, int(ke)])
                    break

        for match in best_matches:
            if match[1] == classes[count]:
                Tp += 1
            else:
                Fp += 1

        precision = Tp/(Tp + Fp)
        precision_hist.append(precision)

        if ((m + 1) % 10) == 0:                                             # increment class by multiples of 10
            count += 1
            precision_per_class.append(np.mean(precision_hist))
            precision_hist = []

    # print(precision_per_class)
    # print(np.mean(precision_per_class))
    # precision_per_weight['(' + str(w[0]) + ',' + str(w[1]) + ',' + str(w[2]) + ')'] = np.mean(precision_per_class)
    precision_per_comb.append(np.mean(precision_per_class))
    print(np.mean(precision_per_class), props)
print(max(precision_per_comb))
    # for pre in precision_per_weight.items():
    #     print(pre)
    # print(max(precision_per_weight, key=precision_per_weight.get))
