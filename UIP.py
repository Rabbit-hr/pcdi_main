import os
import sys
sys.path.append('datas')
import keras
from model.utils import pkl_read, pkl_save, correct_spelling_for_term,tf_idf_toarray, softmax_nonzero_rows
from gensim.models import KeyedVectors
import keras.backend as K
from keras.layers import Multiply
from keras.layers import Input, Dense, LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
from datas.datasets import *

#the intention semantic space.
def creat_intention_semantic_space(word_list, model_path='English_wiki/wiki.en.vec'):
    wiki_word_vectors = KeyedVectors.load_word2vec_format(model_path)

    word_list_remove_numbers = ['i' if any(char.isdigit() for char in x) else x for x in word_list]
    word_vectors = []
    for each_word in word_list_remove_numbers:
        if each_word in wiki_word_vectors:
            each_word_vec = wiki_word_vectors[each_word]
        else:
            each_word = correct_spelling_for_term(each_word)
            if each_word in wiki_word_vectors:
                each_word_vec = wiki_word_vectors[each_word]
            else:
                each_word_vec = np.zeros(wiki_word_vectors.vector_size)
        word_vectors.append(each_word_vec)
    word_vectors = np.array(word_vectors)
    return word_vectors

#R_ut adj
def calculate_R_ut(vector_path,query_list, term_list, model_path,data):
    if not os.path.exists(vector_path):
        t_vectors = creat_intention_semantic_space(term_list, model_path)
        pkl_save(vector_path, t_vectors)
    else:
        t_vectors = pkl_read(vector_path)
    word_vectors = KeyedVectors.load_word2vec_format(model_path, limit=10000)
    u_vectors = np.array([word_vectors[word] for word in query_list])
    # 计算余弦相似度矩阵
    epsilon = 1e-8
    R_ut = np.array([[np.dot(u_i, t_i) / (
                (np.linalg.norm(u_i) + epsilon) * (np.linalg.norm(t_i) + epsilon))
                                   for t_i in t_vectors]
                                  for u_i in u_vectors])
    nan_indices = np.isnan(R_ut)
    R_ut[nan_indices] = -1
    R_ut_adj = np.sum(R_ut,axis=0)/len(query_list)
    final_R_ut = np.tile(R_ut_adj, (data.shape[0], 1))
    return final_R_ut

#R_td adj
def calculate_R_td(clustering_data):
    R_td = tf_idf_toarray(clustering_data)
    R_td_adj = softmax_nonzero_rows(R_td)
    return R_td_adj

#creat_learnable beta in Eq.(2)
def train_beta(dims, act=LeakyReLU(alpha=0.1), init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # inputs
    ones_layer = Input(shape=(dims[0],))#ones
    R_ut_layer = Input(shape=(dims[0],)) #input: R_ut
    ones = ones_layer
    R_ut = R_ut_layer

    # generate learnable weights beta for each relation of user clustering intent descriptors and terms:
    for i in range(n_stacks - 1):
        ones = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(ones)
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(ones)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)
    beta = Dense(dims[0], kernel_initializer=init, name='decoder_0')(h)

    # beta * R_ut
    beta_multi_R_ut = Multiply()([beta, R_ut])
    beta_calculator = Model(inputs=[ones_layer,R_ut_layer], outputs= beta_multi_R_ut)
    o = keras.optimizers.Adam(lr = 1e-3,beta_1=0.9, beta_2=0.999)
    beta_calculator.compile(optimizer=o,loss=loss_cla)
    return beta_calculator

#Eq.(4) in paper:L_cla
def loss_cla(R_ut, beta_multi_R_ut):
    numerator = beta_multi_R_ut/0.1
    non_zero_indices = K.cast(K.not_equal(numerator, 0), K.floatx())
    row_softmax = K.exp(numerator) / K.sum(K.exp(numerator), axis=-1, keepdims=True) * non_zero_indices
    loss_cla = -K.mean(R_ut*K.log(row_softmax + K.epsilon()))
    return loss_cla

#creat_learnable gamma in Eq.(3)
def train_gamma(dims, act=LeakyReLU(alpha=0.1), init='glorot_uniform'):
    n_stacks = len(dims) - 1
    ones_layer = Input(shape=(dims[0],))  # creat: ones
    R_td_layer = Input(shape=(dims[0],))  # input: R_td
    ones = ones_layer
    R_td = R_td_layer

    # generate learnable weights gamma for relation between terms and documents:
    for i in range(n_stacks - 1):
        ones = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(ones)
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(ones)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)
    gamma = Dense(dims[0], kernel_initializer=init, name='decoder_0')(h)

    # gamma * R_td
    gamma_multi_R_td = Multiply()([gamma, R_td])
    gamma_calculator = Model(inputs=[ones_layer, R_td_layer], outputs=gamma_multi_R_td)
    o = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    gamma_calculator.compile(optimizer=o, loss=loss_rela)
    return gamma_calculator

#Eq.(5) in paper:L_rela
def loss_rela(R_ut, gamma_multi_R_td):
    numerator = gamma_multi_R_td/0.1
    non_zero_indices = K.cast(K.not_equal(numerator, 0), K.floatx())
    row_softmax = K.exp(numerator) / K.sum(K.exp(numerator), axis=-1, keepdims=True) * non_zero_indices
    loss_rela = -K.mean(R_ut * K.log(row_softmax + K.epsilon()))
    return loss_rela

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='BBCSport')
    args = parser.parse_args()
    print(args.dataset)

    if args.dataset == 'BBCSport':
        x,y,term_list = load_data(args.dataset)
        clustering_intent_descriptors = ['athletics','cricket','football','tennis','rugby']
    model_path = 'model/English_wiki/wiki.en.vec'

    R_ut = calculate_R_ut(vector_path='datas/'+args.dataset+'/terms_vectors.pkl', query_list=clustering_intent_descriptors, term_list=term_list, model_path=model_path,data=x)
    R_td = calculate_R_td(x)

    #Eq.(1) in paper, user guidance information I
    dims = [x.shape[-1], 500, 500, 256, 128]
    ones = np.ones_like(x)

    #train gamma
    I_td = train_gamma(dims)
    I_td.fit([ones, R_td], R_ut, epochs=50, batch_size=64)
    gamma_multi_R_td = I_td.predict([ones, R_td])
    #train beta
    I_ut = train_beta(dims)
    I_ut.fit([ones, R_ut], R_ut, epochs=50, batch_size=64)
    beta_multi_R_ut = I_ut.predict([ones, R_ut])

    I = gamma_multi_R_td * beta_multi_R_ut
    pkl_save('datas/'+args.dataset+'/user_guidance_information.pkl', I)
