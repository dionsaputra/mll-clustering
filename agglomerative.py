import math
import numpy as np
from scipy.spatial import distance

class Agglomerative:
    def __init__(self, data, k, linkage = 'single'):
        self.data = np.array(data)
        self.k = k
        self.data_length = len(data)
        self.group = []
        self.set_linkage(linkage)
        self.fit()

    def distance(self, pointA, pointB):
        sum_square = 0.0
        for i in range(len(pointA)):
            sum_square += (pointA[i] - pointB[i]) ** 2

        return math.sqrt(sum_square)

    def set_linkage(self, linkage):
        if linkage == 'single':
            self.linkage_function =  self.single
        elif linkage == 'complete':
            self.linkage_function =  self.complete
        elif linkage == 'average':
            self.linkage_function =  self.average
        elif linkage == 'average_group':
            self.linkage_function =  self.average_group
    
    # Initialize all point with a group
    def init_group(self):
        self.num_of_group = self.data_length
        self.group = [[i] for i in range(self.data_length)]

    # Initialize distance of each point in group
    # and then iterate distance for all point in each group
    def init_dist(self):
        self.dist_array = []
        i = 0
        while i < self.data_length:
            j = i + 1
            while j < self.data_length:
                self.dist_array.append((self.linkage_function(self.group[i], self.group[j]), i, j))
                j += 1
            i += 1
        self.dist_array = sorted(self.dist_array, key = lambda x : x[0])
    
    #Find for min distance for linkage = single
    def single(self, pointA, pointB, singleB = False):
        min_dist = math.inf
        if singleB :
            for idx_a in pointA:
                min_dist = min(min_dist, self.distance(self.data[idx_a], pointB))
        else:
            for idx_a in pointA:
                for idx_b in pointB:
                    min_dist = min(min_dist, self.distance(self.data[idx_a], self.data[idx_b]))
        return min_dist

    #Find for max distance for linkage = complete
    def complete(self, pointA, pointB, singleB = False):
        max_dist = 0
        if singleB :
            for idx_a in pointA:
                max_dist = max(max_dist, self.distance(self.data[idx_a], pointB))
        else:
            for idx_a in pointA:
                for idx_b in pointB:
                    max_dist = max(max_dist, self.distance(self.data[idx_a], self.data[idx_b]))
        return max_dist

    #Find for average distance for each point in group for linkage = average
    def average(self, pointA, pointB, singleB = False):
        dist = 0
        if singleB :
            num_of_pair = len(pointA)
            for idx_a in pointA:
                dist += self.distance(self.data[idx_a], pointB) / num_of_pair
        else:
            num_of_pair = len(pointA) * len(pointB)
            for idx_a in pointA:
                for idx_b in pointB:
                    dist += self.distance(self.data[idx_a], self.data[idx_b]) / num_of_pair
        return dist

    #Find for average point for each group and calculate distance for linkage = average_group
    def average_group(self, pointA, pointB, singleB = False):
        idx = len(self.data[0])
        avgA = np.array([0.0 for _ in range(idx)])
        avgB = np.array([0.0 for _ in range(idx)])
        for idx_a in pointA:
            avgA += self.data[idx_a] / len(pointA)
        if singleB:
            avgB = pointB
        else:
            for idx_b in pointB:
                avgB += self.data[idx_b] / len(pointB)
        return self.distance(avgA, avgB)

    # Fit data from init and then joining each group untul it has only k group
    # If it has only k groups then we set cluster
    def fit(self):
        self.init_group()
        self.init_dist()

        while self.num_of_group > self.k:
            curr = self.dist_array.pop(0)
            self.join_group(curr[1], curr[2])
            self.num_of_group -= 1

        self.set_cluster()

    # Join group by remove one group and update the dist_array
    def join_group(self, idxGroup1, idxGroup2):
        self.group[idxGroup1] = self.group[idxGroup1] + self.group[idxGroup2]
        self.group[idxGroup2] = []
        self.dist_array = [item for item in self.dist_array if item[1] != idxGroup2 and item[2] != idxGroup2]
        for i in range(len(self.dist_array)):
            if self.dist_array[i][1] == idxGroup1 or self.dist_array[i][2] == idxGroup1:
                self.dist_array[i] = (self.linkage_function(
                    self.group[self.dist_array[i][1]],
                    self.group[self.dist_array[i][2]]
                ), self.dist_array[i][1], self.dist_array[i][2])
        self.dist_array = sorted(self.dist_array, key = lambda x : x[0])
    
    # Set cluster for all group that have been described before
    def set_cluster(self):
        temp = []
        for i in self.group:
            if(len(i)>0):
                temp.append(i)
        self.group = temp
        # self.group = filter(lambda x: len(x) > 0, self.group)
        self.clusters = [None for _ in range(self.data_length)]
        curr_cluster = 0
        for pointgroup in self.group:
            for member_id in pointgroup:
                self.clusters[member_id] = curr_cluster
            curr_cluster += 1

    # Return all clusters
    def get_cluster(self):
        return self.clusters

    # Predict function
    def predict(self, point):
        min_dist = math.inf
        cluster = 0
        point = np.array(point)
        for i in range(self.k):
            distance = self.linkage_function(self.group[i], point, True)
            if distance < min_dist:
                min_dist = distance
                cluster = i

        return cluster
