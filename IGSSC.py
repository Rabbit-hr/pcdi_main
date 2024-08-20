import keras.backend as K
from keras.layers import Dense, Input
from keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa

def Dual_channel_network(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # dual-chanel network, one channel is devoted to encoding intent guidance information,
    # while the other focuses on capturing intrinsic features of text documents.
    x_i = Input(shape=(dims[0],), name='input_intention_guidance')#input_guidance_information
    x = Input(shape=(dims[0],),name='input_text_document')#'input_document'
    z_i = x_i
    z_x = x
    #dual_encoder
    for i in range(n_stacks - 1):
        z_i = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_q%d' % i)(z_i)
        z_x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(z_x)
    # hidden layer(embedded representation)
    z_i = Dense(dims[-1], kernel_initializer=init, name='encoder_q%d' % (n_stacks - 1))(z_i)
    z_x = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(z_x)
    i_hat = z_i
    x_hat = z_x
    for i in range(n_stacks - 1, 0, -1):
        i_hat = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_q%d' % i)(i_hat)
        x_hat = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x_hat)
    i_hat = Dense(dims[0], kernel_initializer=init, name='decoder_q_0')(i_hat)
    x_hat = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x_hat)
    dc_encoder = Model(inputs=[x_i,x], outputs=[z_i,z_x])
    dcn = Model(inputs=[x_i,x], outputs=[i_hat,x_hat])

    #create loss:Eq.(6) use MSE
    xent_loss_i = 0.5 * K.mean((x_i - i_hat) ** 2, 0)
    xent_loss_x = 0.5 * K.mean((x - x_hat) ** 2, 0)

    dcn_loss = K.mean(xent_loss_i + xent_loss_x)
    dcn.add_loss(dcn_loss)
    dcn.compile(optimizer='adam')

    return dcn,dc_encoder

def normalize(output):
    return (output - K.mean(output, axis=0)) / K.std(output, axis=0)

#construct loss L_G, Eq.(7)
def cross_corr_matrix(z_a_norm, z_b_norm,batch_size):
    return (K.transpose(z_a_norm) @ z_b_norm)/batch_size

def get_off_diag(c):
    zero_diag = K.zeros(c.shape[-1])
    return tf.linalg.set_diag(c, zero_diag)

def cross_corr_matrix_loss(c):
    lambda_amt =5e-3
    #lambda_amt = 5e-4
    s = tf.linalg.diag_part(c)
    c_diff = K.pow(s -1,2)
    off_diag = K.pow(get_off_diag(c),2)
    loss = tf.reduce_sum(c_diff) + lambda_amt * tf.reduce_sum(off_diag)
    return loss

def IGSSC_network(batch_size,dims, act='relu', init='glorot_uniform'):
    dual_chanel,dc_encoder = Dual_channel_network(dims, act='relu', init='glorot_uniform')
    IGSSC_encoder = Model(inputs = dc_encoder.inputs,outputs = dc_encoder.outputs)
    z_i,z_x = IGSSC_encoder.outputs[0],IGSSC_encoder.outputs[1]

    z_a_norm, z_b_norm = normalize(z_i), normalize(z_x)
    c = cross_corr_matrix(z_a_norm, z_b_norm,batch_size)
    con_loss = cross_corr_matrix_loss(c)

    con_loss = K.mean(con_loss)
    cqae_loss = con_loss

    IGSSC_encoder.add_loss(cqae_loss)
    o = tfa.optimizers.LAMB()
    IGSSC_encoder.compile(optimizer='sgd')

    return IGSSC_encoder,dual_chanel,dc_encoder


