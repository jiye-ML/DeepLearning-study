'''
用tensorflow评估模型，
tf.metrics.mean_cosine_distance: 计算labels and predictions 的 cosine distance
'''
import tensorflow as tf


# 数据输入
num_classes = 2
value_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)
value2_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)


# mean_cosine_distance 函数创建两个局部变量，total and count 来计算 mean cosine distance
# 均值被 weights 加权,  mean_cosine_distance: 幂等变换 divides total by count.
mean_tensor, mean_tensor_update = tf.metrics.mean_cosine_distance(value_tensor, value2_tensor, -1)


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        mean = sess.run(mean_tensor_update)
        print("iterator: {}, mean={}".format(batch, mean))

    print("precision_at_k: {}".format(sess.run(mean_tensor)))