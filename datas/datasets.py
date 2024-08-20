import numpy as np
import pickle as pkl
import scanpy as sc

#save information
def pkl_save(root,info):
    ff = open(root, 'wb')
    pkl.dump(info, ff)
    ff.close()
#read pickle informat
def pkl_read(root):
    with open(root, "rb") as f:
        info = pkl.load(f)
    return info
def load_BBC():
    def creatSet(data, labels):
        data_new = []
        labels_new = []
        for i in [0, 1, 2, 3, 4]:
            count = 0
            for j, n in enumerate(data):
                if labels[j] == i:
                    count = count + 1
                    data_new.append(data[j])
                    labels_new.append(labels[j])
        return np.array(data_new), labels_new
    file = open('datas/BBC/bbc.mtx')
    val_list = file.readlines()
    data = np.zeros((9635, 2225))
    for string in val_list[2:]:
        each_lines = string.strip("\n").split(" ")
        data[int(each_lines[0]) - 1][int(each_lines[1]) - 1] += float(each_lines[2])
    data = np.transpose(data)
    labels = []
    with open('datas/BBC/bbc.classes') as file2:
        labels_pre = file2.readlines()[4:]
    for each in labels_pre:
        each = each.strip("\n").split(" ")
        labels.append(int(each[1]))

    with open('datas/BBC/bbc.terms', 'r') as f_w:
        content = f_w.read()
        term_list = content.split('\n')
        f_w.close()
    x_train, labels = creatSet(data, labels)
    labels = np.array(labels)
    return x_train, labels,term_list

def load_BBC_sport():
    adata = sc.read('datas/BBCSport/bbcsport.mtx')
    data = adata.X
    data = data.todense()
    data = data.A
    data = np.transpose(data)
    labels = []
    with open('datas/BBCSport/bbcsport.classes') as file2:
        labels_pre = file2.readlines()[4:]
    for each in labels_pre:
        each = each.strip("\n").split(" ")
        labels.append(int(each[1]))
    labels = np.array(labels)
    with open('datas/BBCSport/bbcsport.terms', 'r') as f_w:
        content = f_w.read()
        term_list = content.split('\n')
        f_w.close()
    return data,labels,term_list


def load_data(dataset_name):
    if dataset_name == 'BBC':
        return load_BBC()
    elif dataset_name == 'BBCSport':
        return load_BBC_sport()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)
