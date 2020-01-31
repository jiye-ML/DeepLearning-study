'''
用tensorflow评估模型， AUC和 PR曲线
'''

import tensorflow as tf

# labels是可以转换成bool的值：ROC或PR适用于二分类（正负样例）
labels = tf.less_equal(tf.random_uniform(shape=[2]), 0.5)
# prediction是样本属于正样本的概率，range [0, 1]
predictions = tf.random_uniform(shape=[2])

# auc:曲线下方的面积。
# ROC：横轴是FPR(假正率),纵轴是TPR(真正率)
#   1.四个特殊点的含义
#   2.画ROC曲线
#   3.AUC of ROC的含义
roc_auc, roc_auc_update_op = tf.metrics.auc(labels, predictions, curve="ROC", name="roc")
# PR：横轴是recall(查全率),纵轴是precision(查准率)
#   1.recall和precision的计算
#   2.平衡点（BEP,recall=precision）
pr_auc, pr_auc_update_op = tf.metrics.auc(labels, predictions, curve="PR", name="pr")

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    for i in range(10000):
        roc_auc_result = sess.run(roc_auc_update_op)
        pr_auc_result = sess.run(pr_auc_update_op)
        print("{}, roc auc={}".format(i, roc_auc_result))
        print("{}, pr auc={}".format(i, pr_auc_result))
    print("final roc auc={}".format(sess.run(roc_auc)))
    print("final pr auc={}".format(sess.run(pr_auc)))
