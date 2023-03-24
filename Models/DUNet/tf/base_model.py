import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers.experimental.preprocessing import Hashing
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dense, Concatenate, Input, Embedding, Dropout, LSTM, \
    Flatten, Permute, Lambda


class CatTower(layers.Layer):
    def __init__(self, hash_bin=1e5, emb_dim=32, tower_config=[32, 64]):
        """
        CatTower for string features.

        Args:
            hash_bin (int, optional): [Nums of hash bins]. Defaults to 10000.
            emb_dim (int, optional): [Embedding dimentions]. Defaults to 32.
            tower_config (list, optional): [config of dense]. Defaults to [32, 64].
        """
        super(CatTower, self).__init__()

        initializer = tf.keras.initializers.HeUniform()

        self.tower_config = tower_config

        self.hn1 = Hashing(num_bins=hash_bin)
        self.emb1 = Embedding(hash_bin, emb_dim, name='embedding')
        self.flatten = Flatten()

        self.dense_layers = tf.keras.Sequential()
        for out_dim in tower_config:
            self.dense_layers.add(
                Dense(out_dim, activation='relu', kernel_initializer=initializer))

    def call(self, inputs):
        x_cat = self.hn1(inputs)
        x_cat = self.emb1(x_cat)
        outputs = self.dense_layers(x_cat)
        outputs = self.flatten(outputs)

        return outputs


class NumTower(layers.Layer):
    def __init__(self, tower_config=[512, 256, 256, 256]):
        """
        NumTower for numerical features

        Args:
            tower_config (list, optional): [config of dense]. Defaults to [512, 256, 256, 256].
        """
        super(NumTower, self).__init__()
        initializer = tf.keras.initializers.HeUniform()
        self.tower_config = tower_config
        self.dense_layers = tf.keras.Sequential()
        for out_dim in tower_config:
            self.dense_layers.add(
                Dense(out_dim, activation='relu', kernel_initializer=initializer))

    def call(self, inputs):
        outputs = self.dense_layers(inputs)
        return outputs


class SeqTower(layers.Layer):
    def __init__(self, emb_dim=71, time_steps=30, hidden=64, out_dim=32):
        """
        SeqTower BiLSTM

        Args:
            emb_dim (int, optional): [input embedding dim]. Defaults to 71.
            time_steps (int, optional): [sequenct lenth]. Defaults to 30.
            hidden (int, optional): [hidden embedding dim]. Defaults to 64.
            out_dim (int, optional): [output dim]. Defaults to 32.
        """
        super(SeqTower, self).__init__()

        self.rnn = tf.keras.Sequential()
        self.rnn.add(
            layers.Bidirectional(layers.LSTM(hidden),
                                 input_shape=(time_steps, emb_dim))
        )
        self.rnn.add(layers.Dense(out_dim))

    def call(self, inputs):
        inputs = tf.transpose(tf.reshape(inputs, [-1, 71, 30]), [0, 2, 1])
        return self.rnn(inputs)


class BaseNet(layers.Layer):

    def __init__(self, hash_bin=1e6, emb_dim=4, cat_tower_config=[32], num_tower_config=[256], merge_config=[256],
                 **kwarg):
        """
        BaseNet

        Input: {
                'cat_features': shape(19, ), dtype=tf.string,
                'num_features': shape(408, ), dtype=tf.float32,
                'seq_features': shape(2130, ), dtype=tf.float32,
                }

        Args:
            hash_bin ([type], optional): [Nums of hash bins]. Defaults to 1e6.
            emb_dim (int, optional): [Embedding dimentions]. Defaults to 4.
            cat_tower_config (list, optional): [config of dense]. Defaults to [32].
            num_tower_config (list, optional): [config of dense]. Defaults to [256].
            merge_config (list, optional): [config of dense]. Defaults to [256].
        """
        super(BaseNet, self).__init__()
        self.cat_tower = CatTower(hash_bin, emb_dim,
                                  cat_tower_config)
        self.num_tower = NumTower(num_tower_config)
        #         self.seq_tower = SeqTower(
        #             emb_dim=71, time_steps=30, hidden=64, out_dim=32)

        self.merge_layers = tf.keras.Sequential()
        for out_dim in merge_config:
            self.merge_layers.add(Dense(
                out_dim, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()))

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()

    def call(self, inputs):
        x_cat = self.cat_tower(inputs['cat_features'])

        x_num = tf.math.log(inputs['num_features'])
        x_num = self.num_tower(self.bn1(x_num))

        #         x_seq = tf.math.log(inputs['seq_features'])
        #         x_seq = self.seq_tower(self.bn2(x_seq))

        #         x_merge = self.bn3(tf.concat([x_cat, x_num, x_seq], 1))
        x_merge = self.bn3(tf.concat([x_cat, x_num], 1))
        x_merge = self.merge_layers(x_merge)

        return x_merge
