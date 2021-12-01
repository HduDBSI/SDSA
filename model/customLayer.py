import tensorflow as tf
from geopy.distance import geodesic
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2
from model.DFPMCLayer import PMFC_Layer


class LinearModel(Model):
    # 0表示张量分解模型，1表示引入空间距离，2表示引入空间距离权重
    def __init__(self, emb_size, feature_columns, len_Seq, embed_reg=1e-6, mode='inner', mode2='2', **kwargs):
        super(LinearModel, self).__init__(**kwargs)
        self.emb_size = emb_size
        self.user_fea_col, self.item_fea_col = feature_columns
        self.len_Seq = len_Seq
        self.mode = mode
        self.mode2 = mode2
        self.ui_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                      input_length=1,
                                      output_dim=self.user_fea_col['embed_dim'],
                                      mask_zero=False,
                                      embeddings_initializer='random_normal',
                                      embeddings_regularizer=l2(embed_reg))
        # item embedding
        # self.iu_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
        #                               input_length=1,
        #                               output_dim=self.item_fea_col['embed_dim'],
        #                               mask_zero=True,
        #                               embeddings_initializer='random_normal',
        #                               embeddings_regularizer=l2(embed_reg))
        # # item2 embedding, not share embedding
        # self.li_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
        #                               input_length=1,
        #                               output_dim=self.item_fea_col['embed_dim'],
        #                               mask_zero=True,
        #                               embeddings_initializer='random_normal',
        #                               embeddings_regularizer=l2(embed_reg))
        # self.il_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
        #                               input_length=1,
        #                               output_dim=self.item_fea_col['embed_dim'],
        #                               mask_zero=True,
        #                               embeddings_initializer='random_normal',
        #                               embeddings_regularizer=l2(embed_reg))
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.PMFC = PMFC_Layer()

    def loss_function(self, X_uti, X_utj):
        # reduce_mean 降维平均值，将高维的数组变为一个数，该数是数组中所有元素的均值
        # 学习率为0.01时出现nan值，考虑增加一个最小值，调用tf.clip_by_value()函数
        if self.mode == 'inner':
            return - 1 * tf.reduce_mean(tf.math.log(tf.nn.sigmoid(X_uti - X_utj)))
        else:
            return tf.reduce_sum(tf.nn.relu(X_uti - X_utj + 0.5))

    def loss_SASRec(self, pos, neg):
        return tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos)) - tf.math.log(1 - tf.nn.sigmoid(neg))) / 2

    def call(self, inputs, **kwargs):
        # input
        user_inputs, seq_inputs, pos_inputs, neg_inputs, distance_pos, distance_neg = inputs
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)  # (None, maxlen)
        ui_embed = self.ui_embedding(user_inputs)  # (None, dim)
        seq_embed = self.item_embedding(seq_inputs)
        # pos embedding
        pos_embed = self.item_embedding(pos_inputs)
        pos_score = self.PMFC([ui_embed, seq_embed, pos_embed, mask, self.mode, self.mode2, distance_pos])
        # neg embedding
        neg_embed = self.item_embedding(neg_inputs)
        neg_score = self.PMFC([ui_embed, seq_embed, neg_embed, mask, self.mode, self.mode2, distance_neg])

        self.add_loss(self.loss_function(pos_score, neg_score))
        # score = tf.concat([pos_score, neg_score], axis=-1)
        return pos_score, neg_score

    def summary(self):
        seq_inputs = Input(shape=(self.len_Seq,), dtype=tf.int32)
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        distance_pos = Input(shape=(1,), dtype=tf.float32)
        distance_neg = Input(shape=(1,), dtype=tf.float32)
        Model(inputs=[user_inputs, seq_inputs, pos_inputs, neg_inputs, distance_pos, distance_neg],
              outputs=self.call([user_inputs, seq_inputs, pos_inputs, neg_inputs,
                                 distance_pos, distance_neg])).summary()
