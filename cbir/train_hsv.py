import os
import django

# import project settings
os.environ['DJANGO_SETTINGS_MODULE'] = 'cbirproject.settings'
django.setup()

# file dependencies
import numpy as np
import cv2

from cbir.models import Image


def euclidean_distance(a, b):
    return np.linalg.norm(a-b)


def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])


def colour_extractor(image):
    features = []
    weights = [0.5, 0.1, 0.1, 0.15, 0.15]
    hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for mask in splitImage(hsvimg):
        features.append(clrHistogram(hsvimg, mask, [8, 12, 3]))

    return np.dot(np.array(weights, dtype=np.float64), np.array(features, dtype=np.float64))


def clrHistogram(image, mask, bins):
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    return hist.flatten()


def splitImage(image):
    masks = []
    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))
    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
    (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
    ellipMask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)
    masks.append(ellipMask)
    for (startX, endX, startY, endY) in segments:
        cornerMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
        masks.append(cv2.subtract(cornerMask, ellipMask))

    return masks


db = Image.objects.all()
colour_features = []
ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
          (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
count = 0
precision_per_class = []

for image in db:
    read = cv2.imread(image.file.path)
    colour = colour_extractor(read)
    colour_features.append(colour)


precision_hist = []
for i, query in enumerate(colour_features):
    distances = []
    for j, other in enumerate(colour_features):
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
