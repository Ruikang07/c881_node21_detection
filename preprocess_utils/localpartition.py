# script for loca partition by Yuchuan Li

import os
import numpy as np
import pandas as pd
import random
import shutil

import argparse

root_path = 'E:/CISC881/Team/dataset_node21/cxr_images/proccessed_data/'

parser = argparse.ArgumentParser(prog='localpartition.py', description='Local partition for the dataset ')

parser.add_argument('--root_dir', default=root_path, help="input directory to process")
parser.add_argument('--output_dir', default=root_path, help="output directory for results")
parser.add_argument('--random', action='store_false', help="whether to use the predefined partition")

parsed_args = parser.parse_args()

image_path = os.path.join(parsed_args.root_dir, 'images/')
meta_csv = os.path.join(parsed_args.root_dir, 'metadata.csv')
train_path = os.path.join(parsed_args.root_dir, 'train/')
if not os.path.exists(train_path):
    os.makedirs(train_path)
test_path = os.path.join(parsed_args.root_dir, 'test/')
if not os.path.exists(test_path):
    os.makedirs(test_path)
train_csv = os.path.join(parsed_args.root_dir, 'train.csv')
test_csv = os.path.join(parsed_args.root_dir, 'test.csv')


def random_partition():
    original_data = pd.read_csv(meta_csv)
    image_list = list(sorted(os.listdir(image_path)))
    negative_list = image_list[0:3748]
    positive_list = image_list[3748:]
    test_positive_list = random.sample(positive_list, 166)
    test_negative_list = random.sample(negative_list, 115)

    train_positive_list = list(set(positive_list) - set(test_positive_list))
    train_negative_list = list(set(negative_list) - set(test_negative_list))
    train_list = train_positive_list + train_negative_list
    test_list = test_positive_list + test_negative_list

    data = np.array(original_data.loc[:, :])
    labels = list(original_data.columns.values)
    labels[0] = ''
    train_csv_array = np.array(labels)
    test_csv_array = np.array(labels)
    test_csv_array = test_csv_array.reshape(1, -1)
    train_csv_array = train_csv_array.reshape(1, -1)

    for i in range(0, 5224):

        temp_filename = data[i, 2]
        if temp_filename in test_list:

            test_csv_array = np.concatenate((test_csv_array, data[i, :].reshape(1, -1)), axis=0)
        else:
            train_csv_array = np.concatenate((train_csv_array, data[i, :].reshape(1, -1)), axis=0)

    pd_data = pd.DataFrame(train_csv_array)
    pd_data.to_csv(train_csv, header=0, index=0)
    pd_data = pd.DataFrame(test_csv_array)
    pd_data.to_csv(test_csv, header=0, index=0)


def file_copy():
    train_data = np.array(pd.read_csv(train_csv).loc[:, :])
    test_data = np.array(pd.read_csv(test_csv).loc[:, :])

    for i in range(0, train_data.shape[0]):
        source_path = os.path.join(image_path, train_data[i, 2])
        destination_path = os.path.join(train_path, train_data[i, 2])
        shutil.copy(source_path, destination_path)

    for i in range(0, test_data.shape[0]):
        source_path = os.path.join(image_path, test_data[i, 2])
        destination_path = os.path.join(test_path, test_data[i, 2])
        shutil.copy(source_path, destination_path)


if __name__ == "__main__":

    if parsed_args.random:
        random_partition()

    file_copy()
