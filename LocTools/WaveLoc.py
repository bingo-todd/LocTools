import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import logging
import configparser
import time
import tensorflow as tf
from tensorflow.keras import Model
import gammatone.filters as gt_filters
from BasicTools import get_fpath, Dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use('Agg')


class WaveLoc_NN(Model):
    def __init__(self, config):
        super(WaveLoc_NN, self).__init__()
        self.fs = np.int32(config['fs'])
        self.cf_low = np.int32(config['cf_low'])
        self.cf_high = np.int32(config['cf_high'])
        self.n_band = np.int32(config['n_band'])
        self.filter_len = np.int32(config['filter_len'])
        self.n_azi = np.int32(config['n_azi'])
        self.n_unit_fcn = np.int32(config['n_unit_fcn'])
        self._build_model()

    def get_gtf_kernel(self):
        cfs = gt_filters.erb_space(self.cf_low, self.cf_high, self.n_band)
        sample_times = np.arange(0, self.filter_len, 1)/self.fs
        irs = np.zeros((self.filter_len, self.n_band), dtype=np.float32)
        EarQ = 9.26449
        minBW = 24.7
        order = 1
        N = 4
        for band_i in range(self.n_band):
            b = 1.019*((cfs[band_i]/EarQ)**order+minBW**order)**(1/order)
            numerator = np.multiply(sample_times**(N-1),
                                    np.cos(2*np.pi*cfs[band_i]*sample_times))
            denominator = np.exp(2*np.pi*b*sample_times)
            irs[:, band_i] = np.divide(numerator, denominator)
        gain = np.max(np.abs(np.fft.fft(irs, axis=0)), axis=0)
        irs_gain_norm = np.divide(np.flipud(irs), gain)
        kernel = np.concatenate((irs_gain_norm,
                                 np.zeros((self.filter_len, self.n_band))),
                                axis=0)
        self.cfs = cfs
        return kernel

    def _concat_layers(self, layer_all, x):
        input_tmp = x
        for layer in layer_all:
            output_tmp = layer(input_tmp)
            input_tmp = output_tmp
        return output_tmp

    def _max_normalization(self, x):
        amp_max = tf.reduce_max(
                    tf.reduce_max(
                        tf.reduce_max(
                            tf.abs(x),
                            axis=1, keepdims=True),
                        axis=2, keepdims=True),
                    axis=3, keepdims=True)
        return tf.divide(x, amp_max)

    def _build_model(self):
        tf.random.set_seed(1)
        kernel_initializer = tf.constant_initializer(self.get_gtf_kernel())
        gtf_kernel_len = 2*self.filter_len

        # Gammatome filter layer
        gt_layer = tf.keras.layers.Conv2D(
                                      filters=self.n_band,
                                      kernel_size=[gtf_kernel_len, 1],
                                      strides=[1, 1],
                                      padding='same',
                                      kernel_initializer=kernel_initializer,
                                      trainable=False, use_bias=False)
        gt_layer_norm = self._max_normalization
        gt_layer_pool = tf.keras.layers.MaxPool2D([2, 1], [2, 1])
        gt_layer_all = [gt_layer, gt_layer_norm, gt_layer_pool]

        # convolve layer
        band_layer_all = []
        for i in range(self.n_band):
            band_layer1 = tf.keras.layers.Conv2D(filters=6,
                                                 kernel_size=[18, 2],
                                                 strides=[1, 1],
                                                 activation=tf.nn.relu)
            band_layer1_pool = tf.keras.layers.MaxPool2D([4, 1], [4, 1])
            #
            band_layer2 = tf.keras.layers.Conv2D(filters=12,
                                                 kernel_size=[6, 1],
                                                 strides=[1, 1],
                                                 activation=tf.nn.relu)
            band_layer2_pool = tf.keras.layers.MaxPool2D([4, 1], [4, 1])
            #
            band_layer3 = tf.keras.layers.Flatten()
            band_layer_all.append([band_layer1, band_layer1_pool,
                                   band_layer2, band_layer2_pool,
                                   band_layer3])

        # frequency channel weight
        weight_layer1 = tf.keras.layers.Dense(units=self.n_band, activation=tf.nn.relu)
        weight_layer2 = tf.keras.layers.Dense(units=self.n_band, activation=tf.nn.sigmoid)
        weight_layer_all = [weight_layer1, weight_layer2]

        #
        fcn_layer1 = tf.keras.layers.Dense(units=self.n_unit_fcn,
                                           activation=tf.nn.relu)
        fcn_layer1_drop = tf.keras.layers.Dropout(rate=0.5)
        fcn_layer2 = tf.keras.layers.Dense(units=self.n_unit_fcn,
                                           activation=tf.nn.relu)
        fcn_layer2_drop = tf.keras.layers.Dropout(rate=0.5)
        ouptut_layer = tf.keras.layers.Dense(units=self.n_azi,
                                             activation=tf.nn.softmax)
        fcn_layer_all = [fcn_layer1, fcn_layer1_drop,
                         fcn_layer2, fcn_layer2_drop,
                         ouptut_layer]

        self.gt_layer_all = gt_layer_all
        self.band_layer_all = band_layer_all
        self.weight_layer_all = weight_layer_all
        self.fcn_layer_all = fcn_layer_all

    def call(self, x):
        gt_layer_output = self._concat_layers(self.gt_layer_all, x)
        band_out_all = []
        for band_i in range(self.n_band):
            band_out = self._concat_layers(
                                self.band_layer_all[band_i],
                                tf.expand_dims(
                                    gt_layer_output[:, :, :, band_i],
                                    axis=-1))
            band_out_all.append(band_out)
        band_out_all_concat = tf.concat(band_out_all, axis=1)
        weight = self._concat_layers(self.weight_layer_all, band_out_all_concat)
        band_out_all_weighted = []
        for band_i in range(self.n_band):
            band_out_all_weighted.append(band_out_all[band_i]*tf.expand_dims(weight[:, band_i], axis=1))
        band_out_all_weighted_concat = tf.concat(band_out_all_weighted, axis=1)
        y_est = self._concat_layers(self.fcn_layer_all, band_out_all_weighted_concat)
        return y_est


class WaveLoc(object):
    def __init__(self, file_reader, log_path, config_fpath=None):
        # constant settings
        config = configparser.ConfigParser()
        config.read(config_fpath)
        self._load_cfg(config_fpath)
        self.epsilon = 1e-20
        self.file_reader = file_reader
        self.nn = WaveLoc_NN(config['model'])

        logger = logging.getLogger()
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        self.logger = logger

    def _load_cfg(self, config_fpath):
        if config_fpath is not None and os.path.exists(config_fpath):
            config = configparser.ConfigParser()
            config.read(config_fpath)
            # settings for model
            self.fs = np.int32(config['model']['fs'])
            self.n_band = np.int32(config['model']['n_band'])
            self.cf_low = np.int32(config['model']['cf_low'])
            self.cf_high = np.int32(config['model']['cf_high'])
            self.frame_len = np.int32(config['model']['frame_len'])
            self.shift_len = np.int32(config['model']['shift_len'])
            self.filter_len = np.int32(config['model']['filter_len'])
            self.n_azi = np.int32(config['model']['n_azi'])
            # settings for training
            self.batch_size = np.int32(config['train']['batch_size'])
            self.max_epoch = np.int32(config['train']['max_epoch'])
            self.is_print_log = config['train']['is_print_log'] == 'True'
            self.train_set_dir = config['train']['train_set_dir'].split(';')
            self.valid_set_dir = config['train']['valid_set_dir'].split(';')
            if self.valid_set_dir[0] == '':
                self.valid_set_dir = None
        else:
            print(config_fpath)
            raise OSError

    def _cal_cross_entropy(self, y_est, y_gt):
        cross_entropy = -tf.reduce_mean(
                            tf.reduce_sum(
                                tf.multiply(
                                    y_gt, tf.math.log(y_est+1e-20)),
                                axis=1))
        return cross_entropy

    def _cal_mse(self, y_est, y_gt):
        mse = tf.reduce_mean(tf.reduce_sum((y_gt-y_est)**2, axis=1))
        return mse

    def _cal_loc_rmse(self, y_est, y_gt):
        azi_est = tf.argmax(y_est, axis=1)
        azi_gt = tf.argmax(y_est, axis=1)
        diff = tf.cast(azi_est - azi_gt, tf.float32)
        return tf.sqrt(tf.reduce_mean(diff**2))

    def _cal_cp(self, y_est, azi_gt):
        equality = tf.equal(tf.argmax(y_est, axis=1),
                            tf.argmax(y_est, axis=1))
        cp = tf.reduce_mean(tf.cast(equality, tf.float32))
        return cp

    def load_model(self, model_dir):
        """load model"""
        model_fpath = tf.train.latest_checkpoint(model_dir)
        self.nn.load_weights(model_fpath)
        print(f'load model from {model_fpath}')

    def _train_record_init(self, model_dir, is_load_model):
        if is_load_model:
            record_info = np.load(os.path.join(model_dir, 'train_record.npz'))
            valid_loss_record = record_info['valid_loss_record']
            lr_value = record_info['lr']
            best_epoch = record_info['best_epoch']
            min_valid_loss = record_info['min_valid_loss']
            last_epoch = np.nonzero(valid_loss_record)[0][-1]
        else:
            valid_loss_record = np.zeros(self.max_epoch)
            lr_value = 1e-3
            min_valid_loss = np.infty
            best_epoch = 0
            last_epoch = -1
        return [valid_loss_record, lr_value,
                min_valid_loss, best_epoch, last_epoch]

    def _get_fpath(self, set_dir):
        if isinstance(set_dir, list):
            dir_all = set_dir
        else:
            dir_all = [set_dir]
        fpath_all = []
        for dir_tmp in dir_all:
            print(dir_tmp)
            fpath_all_tmp = get_fpath(dir_tmp, '.wav', is_absolute=True)
            fpath_all.extend(fpath_all_tmp)
        if len(fpath_all) < 1:
            raise Exception(f'empty dir: {set_dir}')
        return fpath_all

    def run_optimization(self, x, y, optimizer):
        with tf.GradientTape() as g:
            y_est = self.nn(x)
            loss = self._cal_cross_entropy(y_est, y)
        gradients = g.gradient(loss, self.nn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return loss

    def train_model(self, model_dir, is_load_model=False):

        self.logger.info('Train set')
        [self.logger.info('\t{}'.format(item))
         for item in self.train_set_dir]

        self.logger.info('Valid set')
        [self.logger.info('\t{}'.format(item))
         for item in self.valid_set_dir]

        if is_load_model:
            self.load_model(model_dir)

        [valid_loss_record,
         lr, min_valid_loss,
         best_epoch, last_epoch] = self._train_record_init(model_dir,
                                                           is_load_model)

        print('start training')
        self.dataset_train = Dataset(self.file_reader,
                                     self._get_fpath(self.train_set_dir),
                                     self.batch_size*5,
                                     self.batch_size,
                                     [[self.frame_len, 2, 1],
                                      [self.frame_len, 2, 1],
                                      [self.n_azi]])
        self.dataset_valid = Dataset(self.file_reader,
                                     self._get_fpath(self.valid_set_dir),
                                     self.batch_size*5,
                                     self.batch_size,
                                     [[self.frame_len, 2, 1],
                                      [self.frame_len, 2, 1],
                                      [self.n_azi]])
        for epoch in range(last_epoch+1, self.max_epoch):
            t_start = time.time()
            optimizer = tf.optimizers.Adam(lr)
            self.dataset_train.reset()
            while not self.dataset_train.is_finish():
                wav_d, wav_r, y_loc = self.dataset_train.next_batch()
                self.run_optimization(wav_r, y_loc, optimizer)
            # model test
            valid_loss_record[epoch] = self.validate()
            # write to log
            iter_time = time.time()-t_start
            self.logger.info(' '.join((f'epoch:{epoch}',
                                    f'lr:{lr}',
                                    f'time:{iter_time:.2f}\n')))
            self.logger.info(f'\t loss_loc:{valid_loss_record[epoch]}')

            # update min_valid_loss
            if min_valid_loss > valid_loss_record[epoch]:
                self.logger.info('find new optimal\n')
                best_epoch = epoch
                min_valid_loss = valid_loss_record[epoch]

            # save in each epoch in case of interruption
                self.nn.save_weights(f"{model_dir}/cp-{epoch:04d}.ckpt")
                np.savez(os.path.join(model_dir, 'train_record'),
                         valid_loss_record=valid_loss_record,
                         lr=lr,
                         best_epoch=best_epoch,
                         min_valid_loss=min_valid_loss)

            # early stop
            n_epoch_stop = 5
            if epoch-best_epoch > n_epoch_stop:
                print(epoch, best_epoch)
                print('early stop\n', min_valid_loss)
                self.logger.info('early stop{}\n'.format(min_valid_loss))
                break

            # learning rate decay
            n_epoch_decay = 2
            if epoch >= n_epoch_decay:  # no better performance in 2 epoches
                min_valid_loss_local = np.min(
                    valid_loss_record[epoch-n_epoch_decay+1:epoch+1])
                if valid_loss_record[epoch-n_epoch_decay] < min_valid_loss_local:
                    lr = lr*.5

        if True:
            fig, ax = plt.subplots(1, 1, sharex=True, tight_layout=True)
            ax.plot(valid_loss_record, label='loc')
            ax.legend()
            ax.set_ylabel('cost')
            fig_path = os.path.join(model_dir, 'train_curve.png')
            plt.savefig(fig_path)

    def validate(self):
        loss_loc = 0.
        n_sample = 0
        self.dataset_valid.reset()
        while not self.dataset_valid.is_finish():
            wav_d, wav_r, y_loc = self.dataset_valid.next_batch()
            n_sample_tmp = wav_d.shape[0]
            y_loc_est = self.nn(wav_r)
            loss_loc_tmp = self._cal_cross_entropy(y_loc_est, y_loc)
            loss_loc = loss_loc + loss_loc_tmp*n_sample_tmp
            n_sample = n_sample + n_sample_tmp
        loss_loc = loss_loc/n_sample
        return loss_loc

    def evaluate(self, set_dir, log_path):
        """
        """
        logger = open(log_path, 'w')
        file_path_all = self._get_fpath(set_dir)
        for file_i, file_path in enumerate(file_path_all):
            wav_d, wav_r, y_loc = self.file_reader(file_path)

            y_loc_est = self.nn(wav_r)

            logger.write(f'{file_i}; {file_path}')
            n_sample = wav_r.shape[0]
            prob_str_all = []
            for sample_i in range(n_sample):
                prob_str = ' '.join([str(item) for item in y_loc_est[sample_i]])
                logger.write(f'; {prob_str}')
            logger.write('\n')
