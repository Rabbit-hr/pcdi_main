from keras import callbacks
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from keras.layers import Layer, InputSpec,add
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from model.metrics import *
from IGSSC import IGSSC_network
from model.utils import pkl_read
from datas.datasets import load_data

#map to the same value range
def map_to(matrix1, matrix2):
    abs_sum1 = np.mean(np.abs(matrix1))
    abs_sum2 = np.mean(np.abs(matrix2))
    scaling_factor = abs_sum1 / abs_sum2
    normalized_matrix2 = matrix2 * scaling_factor
    print(scaling_factor)
    return normalized_matrix2

class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PCDI(object):
    def __init__(self, dims, true_label,n_clusters, encode_epochs,cross_epochs, encode_batch_size, init='glorot_uniform'):
        super(PCDI, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1
        self.true_label = true_label
        self.n_clusters = n_clusters
        self.encode_epochs =encode_epochs
        self.cross_epochs = cross_epochs
        self.encode_batch_size = encode_batch_size
        self.IGSSC_encoder,self.dual_chanel,self.dc_encoder = IGSSC_network(batch_size=encode_batch_size,dims=self.dims,init=init)

        # prepare with L_C
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(add(self.dc_encoder.output))
        self.model = Model(inputs=self.dc_encoder.input, outputs=clustering_layer)

    def pretrain(self, i, x, batch_size=256, save_dir='results/temp'):
        # dual_chanel_encoder_training_epochs = 100,
        # optimize_with_cross_epochs = 50,
        csv_logger = callbacks.CSVLogger(save_dir + '/pretrain_log.csv')
        print('...learning Z_I and Z_X...')
        cb = [csv_logger]
        self.dual_chanel.fit([i, x],shuffle=True,epochs= 100,batch_size=batch_size,verbose=2)
        print('PCDI...using loss L_G for optimizing...')
        self.IGSSC_encoder.fit([i, x],shuffle=True,epochs=50,batch_size=batch_size,verbose=2)
        self.IGSSC_encoder.save_weights(save_dir + '/encoder_weights.h5')
        print('Pretrained weights are saved to %s/encoder_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):  # load weights
        self.model.load_weights(weights)

    def extract_features(self, x,i):
        return self.IGSSC_encoder.predict([i, x])

    def predict(self, x,i):  # predict cluster labels using the output of clustering layer
        q = self.model.predict([i, x], verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    #optimize with loss_C
    def fit(self, x,i, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp'):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(add(self.IGSSC_encoder.predict([i, x])))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        import csv
        logfile = open(save_dir + '/pcdi_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict([i, x], verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                y_pred = q.argmax(1)
                if y is not None:
                    acc_r = np.round(acc(y, y_pred), 5)
                    nmi_r = np.round(nmi(y, y_pred), 5)
                    ari_r = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict1 = dict(iter=ite, acc=acc_r, nmi=nmi_r, ari=ari_r, loss=loss)
                    logwriter.writerow(logdict1)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc_r, nmi_r, ari_r), ' ; loss=', loss)
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=[i[idx],x[idx]], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', save_dir + '/pcdi_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/pcdi_model_' + str(ite) + '.h5')
            ite += 1
        # renew sumpplement_feature
        sumpplement_feature = np.copy(add(self.IGSSC_encoder.predict([i, x], verbose=0)))
        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/pcdi_model_final.h5')
        self.model.save_weights(save_dir + '/pcdi_model_final.h5')

        return y_pred,sumpplement_feature

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='BBCSport',
                        choices=['BBCSport', 'BBC'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--encode_epochs', default=100, type=int)
    parser.add_argument('--cross_epochs', default=50, type=int)
    parser.add_argument('--encode_batch_size', default=256, type=int)
    parser.add_argument('--update_interval', default=None, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)

    x,y,word_list = load_data(args.dataset)
    n_clusters = len(np.unique(y))
    I = pkl_read('datas/'+args.dataset+'/user_guidance_information.pkl')
    i = map_to(x, I)

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    if args.dataset == 'BBCSport':
        update_interval = 100
        pretrain_epochs = 100
        pretrain_optimizer = Adam(lr=0.001)
    elif args.dataset == 'BBC':
        update_interval = 100
        pretrain_epochs = 100
    if args.update_interval is not None:
        update_interval = args.update_interval
    if args.encode_epochs is not None:
        encode_epochs = args.encode_epochs

    # prepare the PCDI model
    pcdi = PCDI(dims=[i.shape[-1], 500, 500, 256, 128], n_clusters=n_clusters, true_label=y, encode_epochs=args.encode_epochs,cross_epochs=args.cross_epochs,init=init,encode_batch_size=args.encode_batch_size)
    if args.ae_weights is None:
        pcdi.pretrain(x=x,i=i, batch_size=args.batch_size,save_dir=args.save_dir)
    else:
        pcdi.IGSSC_encoder.load_weights(args.ae_weights)

    pcdi.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    print('PCDI...using loss L_C for optimizing...')
    y_pred,y_embed = pcdi.fit(x=x,i=i, y=y, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=update_interval, save_dir=args.save_dir)

    print("final clustering results")
    print(' ' * 8 + '|==> Accuracy: %.4f, acc: %.4f,  nmi: %.4f  ,  ari: %.4f <==|'
          % (Accuracy(y,y_pred), acc(y, y_pred), nmi(y, y_pred), ari(y, y_pred)))