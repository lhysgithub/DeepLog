import numpy as np


def main():
    with open("data/one_hot_abnormal_ground_truth.csv", "w") as w:
        # with open("data/hdfs_train") as f:
        # with open("data/hdfs_test_normal") as f:
        with open("data/hdfs_test_abnormal") as f:
            lines = f.readlines()

            # get max_id
            max_id = 0
            for line in lines:
                ids = line.split(" ")
                for id in ids:
                    if id == "\n" or id == "":
                        continue
                    idi = int(id)
                    max_id = max(idi, max_id)

            # generate one hot
            for line in lines:
                one_hot_line = np.zeros(max_id)
                ids = line.split(" ")
                for id in ids:
                    if id == "\n" or id == "":
                        continue
                    idi = int(id)
                    one_hot_line[idi-1]=1
                # one-hot to str
                one_hot_str = ",".join([str(int(i)) for i in one_hot_line])
                # write file
                w.write(one_hot_str+"\n")


def filter_by_mask(line: str):
    mask = "0. 1. 1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 1. 1"
    mask = [int(i) for i in mask.split(".")]
    line_list = line.split()
    for i in range(len(mask)):
        if mask[i] == 0:
            target = str(int(i+1))
            for j in range(len(line_list)):
                if line_list[j] == target:
                    line_list[j] = ""
    return " ".join(line_list)


def filter_data():
    # with open("data/hdfs_test_abnormal_filtered", "w") as w:
    # with open("data/hdfs_test_normal_filtered", "w") as w:
    with open("data/hdfs_train_filtered", "w") as w:
    #     with open("data/hdfs_test_abnormal") as f:
    #     with open("data/hdfs_test_normal") as f:
        with open("data/hdfs_train") as f:
            lines = f.readlines()
            for line in lines:
                alt_line = filter_by_mask(line)
                w.write(alt_line+"\n")


if __name__ == '__main__':
    # main()
    filter_data()
