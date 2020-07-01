import tensorflow as tf
import tensorflow.contrib.metrics as tcm


# 数据输入
labels = tf.random_uniform(shape=[3], minval=1, maxval=10, dtype=tf.int32)
predictions = tf.random_uniform(shape=[3], minval=1, maxval=10, dtype=tf.int32)
predictions2 = tf.random_uniform(shape=[3], minval=1, maxval=10, dtype=tf.int32)

# 准确率 和 MAE（https://en.wikipedia.org/wiki/Mean_absolute_error）
accuracy, update_op_acc = tcm.streaming_accuracy(predictions, labels, name='prediction1')
# when evaluating the same metric multiple times on different inputs,
# one must specify the scope of each metric to avoid accumulating the results together:
accuracy2, update_op_acc2 = tcm.streaming_accuracy(predictions2, labels, name='prediction2')
error, update_op_error = tcm.streaming_mean_absolute_error(labels, predictions)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        accuracy_value, error_value, accuracy_value2 = sess.run([update_op_acc, update_op_error, update_op_acc2])
        print("iterator: {}, accuracy1 : {}, error1: {}, accuracy2: {}".
              format(batch, accuracy_value, error_value, accuracy_value2))

    accuracy, mean_absolute_error = sess.run([accuracy, error])
    print("accuracy : {}, mean_absolute_error: {}".format(accuracy, mean_absolute_error))


