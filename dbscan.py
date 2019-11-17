import numpy as np
import math


class Dbscan():

    def __init__(self, epsilon, minimum_points):
        self.eps = epsilon
        self.min_pts = minimum_points
        self.data = []
        self.clusters = {}

    def load_data(self, input_data):
        self.data = input_data
        self.labels = [0]*len(self.data)

    def distance(self, pointA, pointB):
        sum_square = 0.0
        for i in range(len(pointA)):
            sum_square += (pointA[i] - pointB[i]) ** 2

        return math.sqrt(sum_square)

    def neighbor_idxs(self, idx):
        result_idxs = []
        for i in range(len(self.data)):
            if i != idx and self.distance(self.data[idx], self.data[i]) < self.eps:
                result_idxs.append(i)

        return result_idxs

    def is_outlier(self, idx):
        return len(self.neighbor_idxs(idx)) < self.min_pts

    def expand_cluster(self, idx):
        queue = [idx]
        i = 0
        while i < len(queue):
            neighbor_idxs = self.neighbor_idxs(idx)
            if len(neighbor_idxs) >= self.min_pts:
                for n_idx in neighbor_idxs:
                    if self.labels[n_idx] == -1:
                        self.labels[n_idx] = self.labels[idx]
                    elif self.labels[n_idx] == 0:
                        self.labels[n_idx] = self.labels[idx]
                        queue.append(n_idx)
            i += 1

    def train(self):
        latest_labels = 0
        for i in range(len(self.data)):
            if self.labels[i] == 0:
                if self.is_outlier(i):
                    self.labels[i] = -1
                else:
                    latest_labels += 1
                    self.labels[i] = latest_labels
                    self.expand_cluster(i)

        self.save_clusters()

    def save_clusters(self):
        for i in range(len(self.labels)):
            if self.labels[i] in self.clusters:
                self.clusters[self.labels[i]].append(self.data[i])
            else:
                self.clusters[self.labels[i]] = [self.data[i]]

    def predict(self, data_test):
        predict_labels = []
        for item_test in data_test:
            minIdx = None
            minDistance = 1000000000
            for i in range(len(self.data)):
                distance = self.distance(item_test, self.data[i])
                if distance < minDistance:
                    minDistance = distance
                    minIdx = i
            predict_labels.append(self.labels[minIdx])
        return predict_labels
