'''
用tensorflow评估模型，
tf.metrics.percentage_below: 计算值落在 threshold 以下的百分比
'''
import tensorflow as tf


# 数据输入
num_classes = 10
value_tensor = tf.random_uniform(shape=[10], minval=0, maxval=num_classes, dtype=tf.int64)


# 函数创建两个局部变量, total and count 来计算值落在 threshold 以下的百分比
mean_tensor, mean_tensor_update = tf.metrics.percentage_below(value_tensor, 5)



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        value, mean = sess.run([value_tensor, mean_tensor_update])
        print("iterator: {}, value={}, mean={}".format(batch, value, mean))

    print("precision_at_k: {}".format(sess.run(mean_tensor)))