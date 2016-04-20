import os
import django

# import project settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'cbirproject.settings'
django.setup()

# file dependencies
import numpy as np
import cv2

from cbir.models import Image
from cbir.cvclasses.localbinarypatterns import LocalBinaryPatterns


def euclidean_distance(a, b):
    return np.linalg.norm(a-b)


def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])


db = Image.objects.all()
grey_images = []
ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
          (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
# a = list(range(1, 30))
# b = list(range(1, 20))
# params = list(itertools.product(a,b))
# paramsb = [(8, 1), (16, 2), (24, 3), (24, 8)]
precision_per_param = {}

for image in db:
    read = cv2.imread(image.file.path)
    grey_images.append(cv2.cvtColor(read, cv2.COLOR_BGR2GRAY))

# for p in params:
count = 0
precision_per_class = []
hists = []
# lbp = LocalBinaryPatterns(p[0], p[1])
lbp = LocalBinaryPatterns(28, 3)
for g in grey_images:
    hists.append(lbp.describe(g))

precision_hist = []
for i, query in enumerate(hists):
    distances = []
    for j, other in enumerate(hists):
        d = chi2_distance(query, other)                            # calculate distance between texture features
        if j in range(ranges[count][0], ranges[count][1]):              # if other in same class as query...
            distances.append([d, 1])                                    # ...mark as 1...
        else:
            distances.append([d, 0])                                    # ...otherwise mark as 0

    matches = sorted(distances)                                         # sort the distances and

    Tp = 0                                                              # initialise true positive
    Fp = 0                                                              # initialise false positive

    for m in matches[:8]:                                              # for the top 10 matches
        if m[1] == 1:                                                   # if marked 1...
            Tp += 1                                                     # ... increase true positive count...
        else:
            Fp += 1                                                     # ...otherwise increase false positive count

    precision = Tp/(Tp + Fp)
    precision_hist.append(precision)                                    # save precision for that class

    if ((i + 1) % 10) == 0:                                             # increment class by multiples of 10
        count += 1
        precision_per_class.append(np.mean(precision_hist))

print(precision_per_class)
print(np.mean(precision_per_class))
# precision_per_param['(' + str(p[0]) + ',' + str(p[1]) + ')'] = np.mean(precision_per_class)
# print(precision_per_param)
# print(max(precision_per_param, key=precision_per_param.get))
