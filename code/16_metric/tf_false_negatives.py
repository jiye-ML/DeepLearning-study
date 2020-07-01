'''
用tensorflow评估模型， 计算总的 false negative 的数目
false_negatives_at_thresholds : 加入阈值
'''
import tensorflow as tf


# 数据输入
num_classes = 2
labels_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)
predictions_tensor = tf.random_uniform(shape=[3], minval=0, maxval=num_classes, dtype=tf.int64)
predictions_tensor2 = tf.random_uniform(shape=[3], minval=0, maxval=1, dtype=tf.float64)



#  labels: ground truth, 维度和 predictions一样. 将被转换为bool
# predictions: 预测值, 任意维度. 将被转换为bool
# 必须是二分类，返回0和1，这样才能计算一共有多少个 False
false_negative_tensor, false_negative_tensor_update = tf.metrics.false_negatives(labels_tensor, predictions_tensor)

# thresholds: A python list or tuple of float thresholds in [0, 1]
# 这里可以让预测为小数, 先将predict根据threshold转换为0,1，然后和真实标签是1的作比较，如果不一样就+1
false_negative_thresholds_tensor, false_negative_thresholds_tensor_update = tf.metrics.false_negatives_at_thresholds(labels_tensor, predictions_tensor2, [0.5])



with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        labels, predictions, pre_k = sess.run([labels_tensor, predictions_tensor2, false_negative_thresholds_tensor_update])
        print("iterator: {}, labels={}, predictions={} precision_at_k: {}".format(batch, labels, predictions, pre_k))

    print("precision_at_k: {}".format(sess.run(false_negative_thresholds_tensor)))