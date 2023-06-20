import csv
from dataset_est.cifar_est import *
from dataset_est.mnist_est import *
def save_results_to_csv(file_path, data):
    with open (file_path,'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data.items())

class Entropy_result:
    def __init__(self):
        self.entropy = {}

    def save_result(self, key, value):
        self.entropy[key] = value

    def get_value(self, key):
        return self.entropy.get(key)


entropy_holder = Entropy_result()
cifar10_data = entropy_cifar10() #get entropy from cifar10 dataset
entropy_holder.save_result("cifar10_data", cifar10_data)
mnist_data = entropy_mnist() #get entropy from mnist dataset
entropy_holder.save_result("mnist_data", mnist_data)

save_results_to_csv('dataset_entropy', entropy_holder.entropy)