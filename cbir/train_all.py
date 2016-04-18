import os
import django

# import project settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'cbirproject.settings'
django.setup()

# file dependencies
import re
import numpy as np
import itertools
from sklearn import preprocessing
from cbir.cvclasses.searcher import Searcher

from cbir.models import Image


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
w_range = [x / 10.0 for x in range(0, 11, 1)]
# weights = [list(x) for x in list(itertools.product(w_range, w_range, w_range)) if x[0] + x[1] + x[2] == 1]
weights = [[.4,.6,.0], [.4,.5,.1], [.4,.4,.2], [.3,.5,.2]]
db = Image.objects.all()
history = {}

precision_per_weight = {}
for w in weights:
    precision_per_class = []
    precision_hist = []
    count = 0
    for m, query in enumerate(db):   # m axis
        Tp = 0
        Fp = 0

        s = Searcher(query)
        c = s.colour()
        t = s.lbpatterns()
        h = s.texture()
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

    print(precision_per_class)
    print(np.mean(precision_per_class))
    precision_per_weight['(' + str(w[0]) + ',' + str(w[1]) + ',' + str(w[2]) + ')'] = np.mean(precision_per_class)
# for pre in precision_per_weight.items():
#     print(pre)
# print(max(precision_per_weight, key=precision_per_weight.get))
