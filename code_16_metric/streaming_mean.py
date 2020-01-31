import tensorflow as tf
import tensorflow.contrib.metrics as tcm

values = tf.random_uniform(shape=[2], minval=1, maxval=5, dtype=tf.int32)

# 1.Initialization: initializing the metric state.
# Calling streaming_mean creates a pair of state variables that will contain：
#   (1) the running sum and (2) the count of the number of samples in the sum.
# tf.local_variables_initializer() sets the sum and count variables to zero.
mean_value, update_op = tcm.streaming_mean(values, weights=[0.2, 0.8])

with tf.Session() as sess:
    # The streaming metrics use local variables.
    sess.run(tf.local_variables_initializer())

    for i in range(10):
        # 2.Aggregation: updating the values of the metric state.
        # Aggregation is performed by examining the current state of values
        #       and incrementing the state variables appropriately.
        mean_value_result, value_result = sess.run([update_op, values])
        print("mean after {}: {} {}".format(i, value_result, mean_value_result))
    # 3.Finalization: computing the final metric value.
    final_mean_value_result = mean_value.eval()
    print("final mean: {}".format(final_mean_value_result))
