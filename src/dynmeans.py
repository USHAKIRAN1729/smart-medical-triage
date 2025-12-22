import numpy as np

class DynMeans:
    def __init__(self, lambda_dist=0.8, max_inactive=3):
        self.lambda_dist = lambda_dist
        self.max_inactive = max_inactive
        self.centers = []
        self.counts = []
        self.inactive = []

    def fit_batch(self, X):
        labels = []

        for x in X:
            if not self.centers:
                self.centers.append(x.copy())
                self.counts.append(1)
                self.inactive.append(0)
                labels.append(0)
                continue

            dists = [np.linalg.norm(x - c) for c in self.centers]
            idx = int(np.argmin(dists))

            if dists[idx] < self.lambda_dist:
                self.counts[idx] += 1
                eta = 1 / self.counts[idx]
                self.centers[idx] = (1 - eta) * self.centers[idx] + eta * x
                self.inactive[idx] = 0
                labels.append(idx)
            else:
                self.centers.append(x.copy())
                self.counts.append(1)
                self.inactive.append(0)
                labels.append(len(self.centers) - 1)

            for i in range(len(self.inactive)):
                if i != idx:
                    self.inactive[i] += 1

        self._prune()
        return labels

    def _prune(self):
        keep = [i for i, a in enumerate(self.inactive) if a <= self.max_inactive]
        self.centers = [self.centers[i] for i in keep]
        self.counts = [self.counts[i] for i in keep]
        self.inactive = [self.inactive[i] for i in keep]

    def get_centers(self):
        return np.array(self.centers)
