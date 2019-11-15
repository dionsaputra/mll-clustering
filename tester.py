import pandas as pd
from dbscan import Dbscan
from agglomerative import Agglomerative
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

        print("akurasi dbscan: ", accurate_sum/len(self.exact_labels))

    def test_agglomerative(self):
        agglomerative_single_model = Agglomerative(self.features, 3)
        agglomerative_complete_model = Agglomerative(self.features, 3, 'complete')
        agglomerative_average_model = Agglomerative(self.features, 3, 'average')
        agglomerative_average_group_model = Agglomerative(self.features, 3, 'average_group')

        agglomerative_single_label = agglomerative_single_model.get_cluster()
        agglomerative_complete_label = agglomerative_complete_model.get_cluster()
        agglomerative_average_label = agglomerative_average_model.get_cluster()
        agglomerative_average_group_label = agglomerative_average_group_model.get_cluster()

        accurate_sum_1 = 0
        for i in range(len(agglomerative_single_label)):
            # print(agglomerative_single_label[i], " ", self.exact_labels[i])
            if agglomerative_single_label[i] == self.exact_labels[i]:
                accurate_sum_1 += 1

        accurate_sum_2 = 0
        for i in range(len(agglomerative_complete_label)):
            # print(agglomerative_complete_label[i], " ", self.exact_labels[i])
            if agglomerative_complete_label[i] == self.exact_labels[i]:
                accurate_sum_2 += 1
        
        accurate_sum_3 = 0
        for i in range(len(agglomerative_average_label)):
            # print(agglomerative_average_label[i], " ", self.exact_labels[i])
            if agglomerative_average_label[i] == self.exact_labels[i]:
                accurate_sum_3 += 1

        accurate_sum_4 = 0
        for i in range(len(agglomerative_average_group_label)):
            # print(agglomerative_average_group_label[i], " ", self.exact_labels[i])
            if agglomerative_average_group_label[i] == self.exact_labels[i]:
                accurate_sum_4 += 1

        print("akurasi agglomerative dengan single linkage: ", accurate_sum_1/len(self.exact_labels))
        print("akurasi agglomerative dengan complete linkage: ", accurate_sum_2/len(self.exact_labels))
        print("akurasi agglomerative dengan average linkage: ", accurate_sum_3/len(self.exact_labels))
        print("akurasi agglomerative dengan average complete linkage: ", accurate_sum_4/len(self.exact_labels))


if __name__ == "__main__":
    tester = Tester()
    tester.test_dbscan()
    tester.test_agglomerative()