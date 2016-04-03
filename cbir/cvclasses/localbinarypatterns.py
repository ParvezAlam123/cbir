from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, points, radius):
        self.points = points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.points, self.radius, method="uniform")

        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.points + 2), range=(0, self.points + 1))

        # normalise the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist
