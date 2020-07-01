'''
用tensorflow评估模型，
tf.metrics.mean_per_class_accuracy:  计算每一类的平均正确率,
'''
import tensorflow as tf


# 数据输入
num_classes = 2
value_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)
value2_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)


# mean_per_class_accuracy:
mean_tensor, mean_tensor_update = tf.metrics.mean_per_class_accuracy(value_tensor, value2_tensor, num_classes)


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        mean = sess.run(mean_tensor_update)
        print("iterator: {}, mean={}".format(batch, mean))

    print("precision_at_k: {}".format(sess.run(mean_tensor)))