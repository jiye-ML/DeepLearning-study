'''
用tensorflow评估模型， 准确率
'''

import tensorflow as tf


# 数据输入
labels = tf.random_uniform(shape=[3], minval=1, maxval=3, dtype=tf.int32)
predictions = tf.random_uniform(shape=[3], minval=1, maxval=3, dtype=tf.int32)

# 准确率, 返回当前准确率和更新操作，
accuracy, update_op_acc = tf.metrics.accuracy(predictions, labels, name='prediction')

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        # 运行更新操作节点返回，当前的准确率。
        accuracy_value = sess.run(update_op_acc)
        print("iterator: {}, accuracy1: {}".format(batch, accuracy_value))

    accuracy = sess.run([accuracy])
    print("accuracy: {}".format(accuracy))


