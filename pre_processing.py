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


if __name__ == '__main__':
    main()