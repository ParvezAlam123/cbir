import numpy as np
from cbir.models import Image


class Searcher:
    def __init__(self, query):
        self.q = query

    @staticmethod
    def euclidean_distance(a, b):
        return np.linalg.norm(a-b)

    @staticmethod
    def chi2_distance(histA, histB, eps=1e-10):
        return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    def colour(self):
        matches = {}
        for instance in Image.objects.all():
            if str(instance) != str(self.q.file) and instance.hsvHist is not None:
                matches[str(instance)] \
                    = self.chi2_distance(np.fromstring(self.q.hsvHist, dtype=np.float32),
                                         np.fromstring(instance.hsvHist, dtype=np.float32))
                self.check_duplicate(self.q, instance)
        return matches

    def texture(self):
        matches = {}
        for instance in Image.objects.all():
            if str(instance) != str(self.q.file) and instance.texture is not None:
                matches[str(instance)] \
                    = self.chi2_distance(np.fromstring(self.q.texture, dtype=np.float64),
                                         np.fromstring(instance.texture, dtype=np.float64))
                self.check_duplicate(self.q, instance)

        return matches

    def lbpatterns(self):
        matches = {}
        for instance in Image.objects.all():
            if str(instance) != str(self.q.file) and instance.lbpHist is not None:
                matches[str(instance)] \
                    = self.chi2_distance(np.fromstring(self.q.lbpHist, dtype=np.float64),
                                         np.fromstring(instance.lbpHist, dtype=np.float64))
                self.check_duplicate(self.q, instance)

        return matches

    @staticmethod
    def check_duplicate(query, instance):
        if np.allclose(np.fromstring(query.bgrHist, dtype=np.float32),
                       np.fromstring(instance.bgrHist, dtype=np.float32)):
            Image.objects.filter(id=query.id).delete()
