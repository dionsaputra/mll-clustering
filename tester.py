import pandas as pd
from dbscan import Dbscan
from sklearn import datasets


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
                self.exact_labels.append(1)
            elif item == label_2:
                self.exact_labels.append(2)
            else:
                self.exact_labels.append(3)

    def test_dbscan(self):
        epsilon = 2
        min_pts = 2
        dbscan = Dbscan(epsilon, min_pts)
        dbscan.load_data(self.features.to_numpy().tolist())
        dbscan.train()

        accurate_sum = 0
        for i in range(len(dbscan.labels)):
            if dbscan.labels[i] == self.exact_labels[i]:
                accurate_sum += 1

        print("akurasi: ", accurate_sum/len(self.exact_labels))


if __name__ == "__main__":
    tester = Tester()
    tester.test_dbscan()
