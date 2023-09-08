import tensorflow as tf


# Tensors concat
def __getShapeOfTensor__():
    # in tf.constant i'm build a constant tensor
    __shapeTensor__ = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    __inputShape__ = tf.keras.Input(shape = [10])

    # here we print the shape of the tensor above
    print(__shapeTensor__.shape)

    # here we print the shape of input
    # be attention that we got none that because we have Unknown batch size
    print(__inputShape__.shape)


__getShapeOfTensor__()

