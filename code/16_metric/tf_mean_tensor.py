'''
用tensorflow评估模型，
tf.metrics.mean_tensor: 计算给定值的平均 element-wise
'''
import tensorflow as tf


# 数据输入
num_classes = 2
value_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)


# 和mean不同, 返回值是元素级别的mean
mean_tensor, mean_tensor_update = tf.metrics.mean_tensor(value_tensor)



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        value, mean = sess.run([value_tensor, mean_tensor_update])
        print("iterator: {}, value={}, mean={}".format(batch, value, mean))

    print("precision_at_k: {}".format(sess.run(mean_tensor)))