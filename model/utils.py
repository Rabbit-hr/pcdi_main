import pickle as pkl
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

#save information
def pkl_save(root,info):
    ff = open(root, 'wb')
    pkl.dump(info, ff)
    ff.close()

#read pickle informat
# ion
def pkl_read(root):
    with open(root, "rb") as f:
        info = pkl.load(f)
    return info

def correct_spelling_for_term(word):
    spell = SpellChecker()
    corrected_word = spell.correction(word)
    corrections = (word, corrected_word)
    print(corrections)
    return corrected_word

def tf_idf_toarray(data):
    tf_idf_transformer = TfidfTransformer()
    data_tf_idf = tf_idf_transformer.fit_transform(data)
    data1 = data_tf_idf.toarray()
    return data1

def softmax_nonzero_rows(matrix):
    # 创建一个与输入矩阵相同形状的零矩阵
    result = np.zeros_like(matrix, dtype=float)
    # 对每一行进行 softmax 操作，只对非零值进行处理
    for row in range(matrix.shape[0]):
        # 获取当前行的非零元素索引
        row_nonzero_indices = np.nonzero(matrix[row, :])[0]
        # 提取当前行的非零元素
        row_values = matrix[row, row_nonzero_indices]
        # 对非零元素进行 softmax 操作
        softmax_values = np.exp(row_values) / np.sum(np.exp(row_values))
        # 将 softmax 后的值放回原矩阵的相应位置
        result[row, row_nonzero_indices] = softmax_values
    return result