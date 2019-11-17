import pandas as pd
import numpy as np
from dbscan import Dbscan
from kmeans import Kmeans
from agglomerative import Agglomerative
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans



class Tester():

    def __init__(self):
        self.filename = "iris.data"
        self.load_data()

    def load_data(self):
        data_columns = ["f1", "f2", "f3", "f4", "label"]
        self.data = pd.read_csv(self.filename, names=data_columns)
        self.features = self.data.drop(["label"], axis=1)

        label_1 = "Iris-setosa"
        label_2 = "Iris-versicolor"
        label_3 = "Iris-virginica"

        self.exact_labels = []
        for item in self.data["label"]:
            if item == label_1:
                self.exact_labels.append(3)
            elif item == label_2:
                self.exact_labels.append(1)
            else:
                self.exact_labels.append(2)

    def test_kmeans(self):
        exact_labels = []
        label_1 = "Iris-setosa"
        label_2 = "Iris-versicolor"
        label_3 = "Iris-virginica"

        for item in self.data["label"]:
            if item == label_1:
                exact_labels.append(0)
            elif item == label_2:
                exact_labels.append(1)
            else:
                exact_labels.append(2)

        k = 3
        kmeans = Kmeans(k)

        X_train, X_test, y_train, y_test = train_test_split(
            self.features, exact_labels, test_size=0.33, random_state=42
        )

        kmeans.load_data(X_train.to_numpy().tolist())
        kmeans.train()
        labels = kmeans.predict(X_test.to_numpy().tolist())

        accurate_sum = 0
        for i in range(len(labels)):
            if labels[i] == y_test[i]:
                accurate_sum += 1

        print("Akurasi K-Means: ", accurate_sum/len(labels))

        kmeans_sklearn = KMeans(n_clusters=3)
        kmeans_sklearn.fit(X_train)

        sklearn_accurate_sum = 0
        for i in range(len(labels)):
            if kmeans_sklearn.labels_[i] == y_test[i]:
                sklearn_accurate_sum += 1

        print("Akurasi K-Means sklearn: ", sklearn_accurate_sum/len(labels))


    def test_dbscan(self):
        exact_labels = []
        label_1 = "Iris-setosa"
        label_2 = "Iris-versicolor"
        label_3 = "Iris-virginica"

        for item in self.data["label"]:
            if item == label_1:
                exact_labels.append(2)
            elif item == label_2:
                exact_labels.append(3)
            else:
                exact_labels.append(1)

        epsilon = 2
        min_pts = 2
        dbscan = Dbscan(epsilon, min_pts)

        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.exact_labels, test_size=0.33, random_state=42
        )
        dbscan.load_data(X_train.to_numpy().tolist())
        dbscan.train()
        labels = dbscan.predict(X_test.to_numpy().tolist())

        accurate_sum = 0
        for i in range(len(labels)):
            if labels[i] == y_test[i]:
                accurate_sum += 1

        print("Akurasi DBScan: ", accurate_sum/len(labels))

        clustering_labels = DBSCAN(
            eps=epsilon, min_samples=min_pts).fit_predict(X_train)
        clustering_labels = [c + 3 for c in clustering_labels]

        sklearn_accurate_sum = 0
        for i in range(len(labels)):
            if clustering_labels[i] == y_test[i]:
                sklearn_accurate_sum += 1

        print("Akurasi DBScan sklearn: ", sklearn_accurate_sum/len(labels))
    
    def get_mapping_to_label(self, n_cluster, y, label):
        uc = np.unique(np.array(y))
        cluster = [{} for i in range(len(uc))]
        label = label.reset_index(drop=True)
        for i in range(len(y)):
            if y[i] is not None and y[i] >= 0:
                loc = np.where(uc == y[i])[0][0]
                if label[i] in cluster[loc]:
                    cluster[loc][label[i]] += 1
                else:
                    cluster[loc][label[i]] = 0
        map = {}
        for i in range(len(uc)):
            if cluster[i]:
                map[uc[i]] = max(cluster[i], key=cluster[i].get)
        return map

    def apply_map_to_cluster(self, y, map):
        return [map[i] for i in y]
    
    def get_accuracy(self, y_pred, y_test):
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                count += 1
        return count / len(y_pred)

    def test_agglomerative(self):

        df = pd.read_csv('iris.data', header=None)

        x = df.drop([4], axis=1)
        y = df[4]  

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=79
        )

        agglomerative_single_model = Agglomerative(X_train, 3)
        agglomerative_complete_model = Agglomerative(
            X_train, 3, 'complete')
        agglomerative_average_model = Agglomerative(
            X_train, 3, 'average')
        agglomerative_average_group_model = Agglomerative(
            X_train, 3, 'average_group')

        agglomerative_single_label = agglomerative_single_model.get_cluster()
        agglomerative_complete_label = agglomerative_complete_model.get_cluster()
        agglomerative_average_label = agglomerative_average_model.get_cluster()
        agglomerative_average_group_label = agglomerative_average_group_model.get_cluster()

        agglomerative_single_map = self.get_mapping_to_label(3, agglomerative_single_label, y_train)
        agglomerative_complete_map = self.get_mapping_to_label(3, agglomerative_complete_label, y_train)
        agglomerative_average_map = self.get_mapping_to_label(3, agglomerative_average_label, y_train)
        agglomerative_average_group_map = self.get_mapping_to_label(3, agglomerative_average_group_label, y_train)


        agglomerative_single_pred = []
        agglomerative_complete_pred = []
        agglomerative_average_pred = []
        agglomerative_average_group_pred = []
        for i, row in X_test.iterrows():
            agglomerative_single_pred.append(agglomerative_single_model.predict(row))
            agglomerative_complete_pred.append(agglomerative_complete_model.predict(row))
            agglomerative_average_pred.append(agglomerative_average_model.predict(row))
            agglomerative_average_group_pred.append(agglomerative_average_group_model.predict(row))

        print(
            'akurasi agglomerative dengan single linkage = ',
            self.get_accuracy(self.apply_map_to_cluster(agglomerative_single_pred, agglomerative_single_map), y_test.reset_index(drop=True))
        )

        print(
            'akurasi agglomerative dengan complete linkage = ',
            self.get_accuracy(self.apply_map_to_cluster(agglomerative_complete_pred, agglomerative_complete_map), y_test.reset_index(drop=True))
        )

        print(
            'akurasi agglomerative dengan average linkage = ',
            self.get_accuracy(self.apply_map_to_cluster(agglomerative_average_pred, agglomerative_average_map), y_test.reset_index(drop=True))
        )

        print(
            'akurasi agglomerative dengan average_group linkage = ',
            self.get_accuracy(self.apply_map_to_cluster(agglomerative_average_group_pred, agglomerative_average_group_map), y_test.reset_index(drop=True))
        )
if __name__ == "__main__":
    tester = Tester()
    tester.test_dbscan()
    tester.test_kmeans()
    tester.test_agglomerative()
