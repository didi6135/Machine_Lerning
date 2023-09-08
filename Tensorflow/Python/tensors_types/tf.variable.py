import tensorflow as tf


my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# Here creating a tensor
tf_var = tf.Variable([1, 2, 3, 4, 5, 6])
# Here we assign a new data into it
a = tf_var.assign([1, 1, 1, 1, 1, 1])
print(a)
# print(my_variable)

# Here we create two tensors with the same name
my_tensor = tf.constant([1, 2, 3, 4, 5])
a_test = tf.Variable(my_tensor, mane="David")
# Here we add 1 to each number in tensor with broadcast
b_test = tf.Variable(my_tensor + 1, mane="David")

# When print we get false even if it's the same name
# print(a_test)
# print(b_test)
# print(a_test == b_test)

step_counter = tf.Variable([1], trainable=False)
# print(step_counter)


with tf.device('CPU:0'):

    a = tf.Variable([
        [1, 2, 3],
        [4, 5, 6]])
    b = tf.constant([
        [1, 2],
        [3, 4],
        [5, 6]])

    #  The calculate work like this
    # it's multi the a with shape 0 (1, 2, 3) with b with shape 0 (1, 3, 5)
    # c[0, 0] = (1 * 1) + (2 * 3) + (3 * 5) = 1 + 6 + 15 = 22

    # it's multi the a with shape 0 (1, 2, 3) with b with shape 1 (2, 4, 6)
    # c[0, 1] = (1 * 2) + (2 * 4) + (3 * 6) = 2 + 8 + 18 = 28

    # it's multi the a with shape 1 (4, 5, 6) with b with shape 0 (1, 3, 5)
    # c[1, 0] = (4 * 1) + (5 * 3) + (6 * 5) = 4 + 15 + 30 = 49

    # it's multi the a with shape 1 (4, 5, 6) with b with shape 1 (2, 4, 6)
    # c[1, 1] = (4 * 2) + (5 * 4) + (6 * 6) = 8 + 20 + 36 = 64
    c = tf.matmul(a, b)

# print(a)
# print(b)
# print(c)




