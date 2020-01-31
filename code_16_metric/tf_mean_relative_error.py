'''
用tensorflow评估模型，
tf.metrics.mean_relative_error: 通过正则给定的值计算平均 relative error ;
'''
import tensorflow as tf


# 数据输入
num_classes = 2
value_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.float64)
value2_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.float64)
norm_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.float64)


# mean_per_class_accuracy: abs(a - b) / norm
# 如果 norm是0 返回label值
mean_tensor, mean_tensor_update = tf.metrics.mean_relative_error(value_tensor, value2_tensor, norm_tensor)


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        mean = sess.run(mean_tensor_update)
        print("iterator: {}, mean={}".format(batch, mean))

    print("precision_at_k: {}".format(sess.run(mean_tensor)))