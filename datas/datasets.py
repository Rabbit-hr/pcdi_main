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
    if dataset_name == 'BBCSport':
        return load_BBC_sport()
    else:
        print('Not defined for loading', dataset_name)
        exit(0)
