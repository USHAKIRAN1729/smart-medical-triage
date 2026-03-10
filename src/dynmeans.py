import numpy as np

class DynMeans:
    def __init__(self, lambda_dist=0.75, max_inactive=10):
        self.lambda_dist = lambda_dist
        self.max_inactive = max_inactive
        self.centers = []
        self.counts = []
        self.inactive = []

    def fit_batch(self, X):
        labels = []
        for x in X:
            if len(self.centers) == 0:
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
        return labels

    def get_centers(self):
        return np.array(self.centers)
