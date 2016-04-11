import os
import django

# import project settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'cbirproject.settings'
django.setup()

# file dependencies
import numpy as np
import cv2
import itertools
from sklearn import preprocessing
from cbir.cvclasses.searcher import Searcher

from cbir.models import Image


def combine(a, b, w):
    matches = {}

    # split dictionaries into keys and values
    al = [x for x in a.items()]
    ak, av = zip(*al)
    bl = [x for x in b.items()]
    bk, bv = zip(*bl)

    # scale the values in the range 0-1
    a_scaled = preprocessing.minmax_scale(av, feature_range=(0,1))
    b_scaled = preprocessing.minmax_scale(bv, feature_range=(0,1))

    # build numpy structured arrays combining scaled values and original keys
    names = ['keys', 'values']
    formats = ['S225', 'f8']
    dtype = dict(names=names, formats=formats)
    anp = np.array(list(zip(ak,a_scaled)), dtype=dtype)
    bnp = np.array(list(zip(bk,b_scaled)), dtype=dtype)

    # iterate over numpy structures creating a weighted average between values with the same key
    for i, t1 in np.ndenumerate(anp):
        for j, t2 in np.ndenumerate(bnp):
            if anp['keys'][i] == bnp['keys'][j]:
                stack = np.vstack((anp['values'][i], bnp['values'][j]))
                matches[anp['keys'][i].decode("utf-8")] = np.average(stack, axis=0, weights=w)[0]   # python dictionary

    return matches

w_range = [x / 10.0 for x in range(1, 10, 1)]
weights = [list(x) for x in list(itertools.product(w_range, w_range)) if x[0] + x[1] == 1]

history = {}
for w in weights:
    error = 0
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250)]
    count = 0
    for m, query in enumerate(Image.objects.all()):   # m axis
        if (m + 1) % 50:
            count += 1
        same = []
        s = Searcher(query)
        c = s.colour()
        t = s.texture()
        matches = combine(c, t, w)

        for n, value in enumerate(matches.values()):
            if n in range(ranges[count][0], ranges[count][1]):
                same.append(value)
            else:
                for x in same:
                    if value < x:
                        error += 1

    history['(' + str(w[0]) + ',' + str(w[1]) + ')'] = error

print(min(history, key=history.get))
