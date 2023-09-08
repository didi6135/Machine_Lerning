import tensorflow as tf

# Create tensor with
# 2 batch -- define how many batches you have
# 3 Height -- define what will be the height of each batch
# 4 Width -- define what will be the width of each batch
# 5 Features -- define how many feature inside each width

# The rank of this example will be 4 because it's a tensor
# with 2 batch with height of 3 each one, with width 4 each one,
#  with 5 feature each one
zeros_tensor = tf.zeros([2, 3, 4, 5])
# print(zeros_tensor)


# print('type of every element: ', zeros_tensor.dtype)
# print('number of axes: ', zeros_tensor.ndim)
# print('Elements along axis 0 of tensor: ', zeros_tensor.shape[0])
# print('Total number of elements (2*3*4*5): ', tf.size(zeros_tensor).numpy())

tf.rank(zeros_tensor)

# ---------------------------- Indexing -------------------------------
# Get elements from tensor with rank 1
# it's look like the list in python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
# print(rank_1_tensor.numpy())
# print("First:", rank_1_tensor[0].numpy())
# print("Second:", rank_1_tensor[1].numpy())
# print("Last:", rank_1_tensor[-1].numpy())
# print("Everything:", rank_1_tensor[:].numpy())
# print("Before 4:", rank_1_tensor[:4].numpy())
# print("From 4 to the end:", rank_1_tensor[4:].numpy())
# print("From 2, before 7:", rank_1_tensor[2:7].numpy())
# print("Every other item:", rank_1_tensor[::2].numpy())
# print("Reversed:", rank_1_tensor[::-1].numpy())

# Get elements from tensor with rank 2
rank_2_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
# Print the second row & the second column === 5
print(rank_2_tensor[1, 1].numpy())
print(rank_2_tensor[1, 1].numpy())

# Get elements from tensor with rank 3
# Example of tensor with rank of 3 with  shape of (1 - batch, 2 - height, 3 - width)
rank_3_tensor_example_1 = tf.constant([[[1, 2, 3], [4, 5, 6]]])
# Example of tensor with rank of 3 with  shape of (2 - batch, 3 - height, 1 - width)
rank_3_tensor_example_2 = tf.constant([[[1], [2], [3]], [[4], [5], [6]]])
# Example of tensor with rank of 3 with  shape of (2 - batch, 2 - height, 2 - width)
rank_3_tensor_example_3 = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# print(rank_3_tensor_example_1)
# print(rank_3_tensor_example_2)
# print(rank_3_tensor_example_3)


rank_3_tensor_3batch_2height_5features = tf.constant([
    [[1, 1, 1, 0, 0], [2, 2, 2, 0, 0]],
    [[3, 3, 3, 0, 0], [4, 4, 4, 0, 0]],
    [[5, 5, 5, 0, 0], [6, 6, 6, 0, 0]]])


# Reshape the tensor to shape of (2, 15)
# because we define 2 and the -1 take all and split with 2
ee = tf.reshape(rank_3_tensor_3batch_2height_5features, [2, -1])
# print(ee)

# It's return only the 4 in the rank 3
# it's mean that it go inside the batch
# and then inside the height,
# and then it's go-to length number 4 until the end, and it's return the same rank with specific numbers
# print(rank_3_tensor_3batch_2height_5features[:, :, 4:])

# In this example it's return only the length 4
#  that why it's become to tensor with 2 rank
# print(rank_3_tensor_3batch_2height_5features[:, :, 4])

# it's Reshape this tensor to be 3, 10
# because the -1 give all rank
# and the 3 it's the width
# print(tf.reshape(rank_3_tensor_3batch_2height_5features, [3, -1]))

# -------------------------- Manipulating Shapes ----------------------
x = tf.constant([[1], [2], [3]])
# print(x)

# it's reshape from (3,) to (1,3)
reshaped_tensor = tf.reshape(x, [1, 3])

# Change the shape from (1, 3) to (3,)
# the -1 return the all tensor inside 1 rank
reshaped_tensor2 = tf.reshape(x, [-1])
# print(reshaped_tensor2.shape)
# print(reshaped_tensor)
# print(reshaped_tensor2)

# -------------------------- Broadcasting ----------------------
broadcasting1 = tf.constant([1, 2, 3, 4])
broadcasting2 = tf.constant(2)
broadcasting3 = tf.constant([2, 2, 2])

# A sample example of multi tensor
print(tf.multiply(broadcasting1, 2))

# Make a reshape to the tensor to be (4, 1)
reshape_broadcasting1 = tf.reshape(broadcasting1, [4, 1])
# Create a tensor with sequences numbers from 1 to 5 without limit it's mean without 5
range_num = tf.range(1, 5)

print(reshape_broadcasting1, "\n")
print(range_num, '\n')

# In this print we change the rank to (4, 4)
# that's because it's multi each number with all the range that we define earlier
print(tf.multiply(reshape_broadcasting1, range_num))

print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3, 3]))

