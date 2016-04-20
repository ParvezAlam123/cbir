import os
import django

# import project settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'cbirproject.settings'
django.setup()

# file dependencies
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
import itertools

from cbir.models import Image


def euclidean_distance(a, b):
    return np.linalg.norm(a-b)


def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])


def get_texture(image, props):
    # props = ['dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
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

db = Image.objects.all()
ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
          (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]

s_group = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy']
combinations = []

for s in range(len(s_group)+1):
    for c in itertools.combinations(s_group, s):
        if len(c) != 0 and len(c) <= (len(s_group)+1):
            combinations.append(list(c))

combinations = [['dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'entropy']]

read_images = []
for image in db:
    read = cv2.imread(image.file.path)
    read_images.append(read)

m_error_comb = []
precision_per_comb = []
for comb in combinations:
    m_error_hist = []
    precision_per_class = []
    texture_features = []
    count = 0
    for iid in read_images:
        texture = get_texture(iid, comb)
        texture_features.append(texture)

    precision_hist = []
    for i, query in enumerate(texture_features):
        m_error = 0
        distances = []
        for j, other in enumerate(texture_features):
            d = euclidean_distance(query, other)                            # calculate distance between texture features
            if count == 8:
                if j in range(20, 40):
                    distances.append([d, 2])
                elif j in range(ranges[count][0], ranges[count][1]):
                    distances.append([d, 1])                                    # ...mark as 1...
                else:
                    distances.append([d, 0])
            else:
                if j in range(ranges[count][0], ranges[count][1]):              # if other in same class as query...
                    distances.append([d, 1])                                    # ...mark as 1...
                else:
                    distances.append([d, 0])                                    # ...otherwise mark as 0

        matches = sorted(distances)                                         # sort the distances and

        Tp = 0                                                              # initialise true positive
        Fp = 0                                                              # initialise false positive

        for m in matches[:8]:                                              # for the top 10 matches
            if m[1] == 2:
                m_error += 1
            if m[1] == 1:                                                   # if marked 1...
                Tp += 1                                                     # ... increase true positive count...
            else:
                Fp += 1                                                     # ...otherwise increase false positive count

        precision = Tp/(Tp + Fp)
        precision_hist.append(precision)                                    # save precision for that class

        if count == 8:
            m_error_hist.append(m_error)

        if ((i + 1) % 10) == 0:                                             # increment class by multiples of 10
            count += 1
            precision_per_class.append(np.mean(precision_hist))

    precision_per_comb.append(precision_per_class)
    m_error_comb.append(np.mean(m_error_hist))
    print(precision_per_class)
    print(np.mean(precision_per_class))
# overall = []
# weak = []
# for prec in precision_per_comb:
#     store = []
#     for n, pr in enumerate(prec):
#         if ((n+1) == 3) or ((n+1) == 9):
#             store.append(pr)
#
#     print(np.mean(prec), np.mean(store))
#     overall.append(np.mean(prec))
#     weak.append(np.mean(store))
# print('best overall: ' + str(max(overall)), 'best across weaker classes: ' + str(max(weak)))
# for m in m_error_comb:
#     print(m)
