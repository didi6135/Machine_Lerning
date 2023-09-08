import tensorflow as tf

# Define some input data
x = tf.constant(3.0)

# Record operations inside a GradientTape context
with tf.GradientTape() as tape:
    tape.watch(x)  # Watch the variable x
    y = x * 10

# Compute the gradient of y with respect to x
dy_dx = tape.gradient(y, x)

print(f"dy/dx at x = 3.0 is {dy_dx.numpy()}")

# f(x) = 2x**2 - 2
# f'(10)
# f(x) = 2 . 1 - 2