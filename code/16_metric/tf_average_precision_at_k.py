'''
用tensorflow评估模型， 计算预测值和离散标签平均 precision_k 
幂等操作： s *s = s
'''
import tensorflow as tf


# 数据输入
num_classes = 3
labels_tensor = tf.random_uniform(shape=[1], minval=0, maxval=num_classes, dtype=tf.int64)
predictions_tensor = tf.random_uniform(shape=[1, num_classes], minval=0, maxval=1, dtype=tf.float64)


# average_precision_at_k方法创建两个局部变量 average_precision_at_<k>/total 和 average_precision_at_<k>/max,
# 使用它们来计算频率，, 计算的到的频率最后返回最为： divides average_precision_at_<k>/total by average_precision_at_<k>/max.
# mean = total / max

# 为了评估数据流, 函数创建 update_op 操作来更新变量,返回 precision_at_<k>.
# 内部， a top_k 操作计算一个tensor的top_k下标，设置操作 top_k and labels 计算 true positives and false positives
# 然后更新操作增加 true_positive_at_<k> and false_positive_at_<k> 使用这些值。
pre_k_tensor, pre_k_update_tensor = tf.metrics.average_precision_at_k(labels_tensor, predictions_tensor, 1, name='pre_k')


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        labels, predictions, pre_k = sess.run([labels_tensor, predictions_tensor, pre_k_update_tensor])
        print("iterator: {}, labels={}, predictions={} precision_at_k: {}".format(batch, labels, predictions, pre_k))

    print("precision_at_k: {}".format(sess.run([pre_k_tensor])))
