import numpy as np
import math


class KMeans():

    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def load_data(self, input_data):
        self.data = input_data
        self.labels = [0]*len(self.data)
        self.centroids = []

        np.random.seed(100)
        for i in range(self.k):
            idx = np.random.randint(0, len(self.data))
            self.centroids.append(self.data[idx])

    def distance(self, pointA, pointB):
        sum_square = 0.0
        for i in range(len(pointA)):
            sum_square += (pointA[i] - pointB[i]) ** 2

        return math.sqrt(sum_square)

    def find_nearest_cluster(self, point):
        distances = []

        for i in range(self.k):
            distances.append(self.distance(point, self.centroids[i]))

        return distances.index(min(distances))

    def update_centroids(self):
        for i in range(len(self.clusters)):
            self.centroids[i] = np.average(self.clusters[i], axis=0)
        
    def is_converge(self):
        return False

    def print_centroid(self):
        for i in range(3):
            print(self.centroids[i])

    def train(self):
        for i in range(self.max_iterations):
            self.clusters = []

            for i in range(self.k):
                self.clusters.append([])

            for i in range(len(self.data)):
                nearest_cluster = self.find_nearest_cluster(self.data[i])
                self.clusters[nearest_cluster].append(self.data[i])

            self.pervious = self.centroids
            self.update_centroids()

    
    def predict(self, data):
        labels = []
        for i in range(len(data)):
            labels.append(self.find_nearest_cluster(data[i]))
        
        return labels
