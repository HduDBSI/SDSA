import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w1 = self.add_variable(name='w', shape=[input_shape[0][-1], self.units], initializer=tf.zeros_initializer())
        self.w2 = self.add_variable(name='w', shape=[input_shape[1][-1], self.units], initializer=tf.zeros_initializer())
        self.w3 = self.add_variable(name='w', shape=[input_shape[2][-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b', shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        mf, pmf, d = inputs
        output = tf.matmul(mf, self.w1)+tf.matmul(mf, self.w2)+tf.matmul(mf, self.w3) + self.b
        return output
