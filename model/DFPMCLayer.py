import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2

from model.selfAttLayer import SelfAttention_Layer
from model.linearLayer import LinearLayer


class PMFC_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(PMFC_Layer, self).__init__()
        self.linear = LinearLayer(units=1)
        self.att = SelfAttention_Layer()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs, **kwargs):
        user_info, seq_info, pos_info, mask, mode, mode2, distance = inputs
        # if mode == 'inner':
        #     # MF
        #     # transpose函数式返回一个转置的tensor，[0,2,1]是Viu的维数排列
        #     #  将矩阵ui和矩阵iu转置相乘
        #     mf = tf.matmul(Vui, tf.transpose(Viu, [0, 2, 1]))  # [b,1,S]
        #     # squeeze去除维度为1的向量
        #     mf = tf.squeeze(mf, 1)  # [b, S]
        #     # # PMF
        #     # pmf = tf.matmul(Vil, tf.transpose(Vli, [0, 2, 1]))  # [b,S,L]
        #     # # reduce_mean计算张量上的各个平均值。
        #     # pmf = tf.reduce_mean(pmf, -1)  # [b,S,1]
        #     att_dim = self.att([Vli, Vli, Vli, mask])
        #     att_dim = tf.expand_dims(att_dim, axis=1)
        #     pmf = tf.matmul(att_dim, tf.transpose(Vil, [0, 2, 1]))
        #     pmf = tf.squeeze(pmf, 1)
        if mode == '1':
            # MF
            mf = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1)
            pmf = tf.reduce_sum(tf.multiply(seq_info, pos_info), axis=-1)
            pmf = tf.reduce_mean(pmf, axis=-1)
            pmf = tf.expand_dims(pmf, axis=1)
        else:
            # MF
            mf = tf.reduce_sum(tf.multiply(user_info, pos_info), axis=-1)
            # PMF
            Vli = self.att([seq_info, seq_info, seq_info, mask])
            Vli = tf.expand_dims(Vli, axis=1)
            pmf = tf.reduce_sum(tf.multiply(Vli, pos_info), axis=-1)
        # a = tf.nn.sigmoid(d)
        # x = a*pmf + (1-a)*mf + d  # [B,S]
        if mode2 == '0':
            x = self.dense(tf.concat([pmf, mf], axis=-1))
        elif mode2 == '2':
            d = tf.pow(distance, -1)
            x = self.dense(tf.concat([pmf, mf, d], axis=-1))
        else:
            d = tf.pow(distance, -1)
            x = self.dense(tf.concat([pmf, d], axis=-1))
        return x
