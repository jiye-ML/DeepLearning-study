"""
1.miou:mean Intersection-Over-Union.
2.mpca:mean of the per-class accuracies.
"""
import tensorflow as tf

labels = tf.random_uniform(shape=[5, 20], minval=0, maxval=3, dtype=tf.int32)
predictions = tf.random_uniform(shape=[5, 20], minval=0, maxval=3, dtype=tf.int32)

# 1.miou:mean Intersection-Over-Union.
miou, miou_update_op = tf.metrics.mean_iou(labels, predictions, num_classes=3)
# 2.mpca:mean of the per-class accuracies.
mpca, mpca_update_op = tf.metrics.mean_per_class_accuracy(labels, predictions, num_classes=3)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    for i in range(10):
        miou_result, mpca_result = sess.run([miou_update_op, mpca_update_op])
        print("{} miou={}, mpca={}".format(i, miou_result, mpca_result))
        pass
    miou_result, mpca_result = sess.run([miou, mpca])
    print("final miou={}, mpca={}".format(miou_result, mpca_result))
