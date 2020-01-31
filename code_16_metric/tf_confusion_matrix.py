"""
分类结果混淆矩阵
"""
import tensorflow as tf

# labels是可以转换成bool的值
labels = tf.greater_equal(tf.random_uniform(shape=[2]), 0.5)
# prediction是样本属于正样本的概率，range [0, 1]
predictions = tf.random_uniform(shape=[2])

# 1.Computes the total number of false negatives.
fn_value, fn_update_op = tf.metrics.false_negatives(labels, tf.greater_equal(predictions, 0.5), name='fn')
fn_t_value, fn_t_update_op = tf.metrics.false_negatives_at_thresholds(labels, predictions,
                                                                      thresholds=[0.2, 0.5, 0.8], name='fn_t')
# 2.Sum the weights of false positives.
fp_value, fp_update_op = tf.metrics.false_positives(labels, tf.greater_equal(predictions, 0.5), name='fp')
fp_t_value, fp_t_update_op = tf.metrics.false_positives_at_thresholds(labels, predictions,
                                                                      thresholds=[0.2, 0.5, 0.8], name='fp_t')
# 3.Sum the weights of true_positives.
tp_value, tp_update_op = tf.metrics.true_positives(labels, tf.greater_equal(predictions, 0.5), name='tp')
tp_t_value, tp_t_update_op = tf.metrics.true_positives_at_thresholds(labels, predictions,
                                                                     thresholds=[0.2, 0.5, 0.8], name='tp_t')
# 4.Computes true negatives at provided threshold values.
tn_t_value, tn_t_update_op = tf.metrics.true_negatives_at_thresholds(labels, predictions,
                                                                     thresholds=[0.2, 0.5, 0.8], name='tn_t')

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    for i in range(10):
        fn_value_result, fn_t_value_result = sess.run([fn_update_op, fn_t_update_op])
        print("{} fn:{},fn_t:{}".format(i, fn_value_result, fn_t_value_result))
    fn_value_result, fn_t_value_result = sess.run([fn_value, fn_t_value])
    print("final fn:{},fn_t:{}".format(fn_value_result, fn_t_value_result))


