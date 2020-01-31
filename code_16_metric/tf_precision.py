'''
用tensorflow评估模型， 
tf.metrics.precision: 计算预测和标签间的准确率, 两个变量是同一样类型的值
# precision_at_k： prediction 比标签多一个维度， 在预测的之中最大的前k的类中，是label标签的频率是多大。
# precision_at_thresholds， 二分类问题中， 对于predict大于阈值=1, 否则=0, 然后计算
# precision_at_top_k, 计算标签和top_k之后的预测
'''
import tensorflow as tf


# 数据输入
num_classes = 10
labels_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)
predictions_tensor = tf.random_uniform(shape=[3, num_classes], minval=0, maxval=1, dtype=tf.float64)

# 函数创建两个局部变量, true_positives and false_positives, divides true_positives by the sum of true_positives and false_positives.
mean_tensor, mean_tensor_update = tf.metrics.precision_at_k(labels_tensor, predictions_tensor)



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        mean = sess.run(mean_tensor_update)
        print("iterator: {}, mean={}".format(batch, mean))

    print("precision_at_k: {}".format(sess.run(mean_tensor)))

    pass