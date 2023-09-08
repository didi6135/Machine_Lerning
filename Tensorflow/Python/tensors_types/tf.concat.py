import tensorflow as tf


# Tensors concat
def concatTensors():
    #  Create tow array with numbers
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8, 9, 10]

    # Make concat to the arrays with axis 0
    # that bring one array with both of them
    concat = tf.concat([arr1, arr2], 0)

    # Make concat with axis 1
    #  that bring back tensor2d with shape 2, 10
    concat1 = tf.concat([[arr1, arr2], [arr1, arr2]], 1)
    print(concat)
    print(concat1)


# concatTensors()

# c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
#
# # Find the largest value
# print(tf.reduce_max(c))
# # Find the index of the largest value
# print(tf.math.argmax(c))
# # Compute the softmax
# print(tf.nn.softmax(c))

