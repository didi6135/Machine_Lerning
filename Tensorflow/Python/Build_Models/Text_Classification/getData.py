import shutil
import numpy as np
import tensorflow as tf
import os
import string
import re


# def get_data():
#     url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#
#     getDataset = tf.keras.utils.get_file("aclImdb_v1", url,
#                                          untar=True, cache_dir='.',
#                                          cache_subdir='')
#
#     dataset_dir = os.path.join(os.path.dirname(getDataset), 'aclImdb')
#
#     os.listdir(dataset_dir)
#
#     train_dir = os.path.join(dataset_dir, "train")
#     os.listdir(train_dir)
#
#     #  Remove unused folder from dataset
#     remove_dir = os.path.join(train_dir, 'unsup')
#     shutil.rmtree(remove_dir)
#
#     sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
#     with open(sample_file) as f:
#         print(f.read())
#
#     split_dataset()


# def split_dataset():
batch_size = 32
seed = 42

# create dataset for train
train_dataset = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
)

# Create dataset for validation
validation_dataset = tf.keras.utils.text_dataset_from_directory(
       'aclImdb/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
)

# Create dataset for test
test_dataset = tf.keras.utils.text_dataset_from_directory(
       'aclImdb/test',
        batch_size=batch_size
)


def fiexd_text(input_data):
    lowercase = tf.strings.lower(input_data)
    remove_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(remove_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.TextVectorization(
         standardize=fiexd_text,
         max_tokens=max_features,
         output_mode='int',
         output_sequence_length=sequence_length)

train_text = train_dataset.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(train_dataset))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", train_dataset.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


train_ds = train_dataset.map(vectorize_text)
validation_ds = validation_dataset.map(vectorize_text)
test_ds = test_dataset.map(vectorize_text)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features + 1, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.summary()

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  tf.keras.layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(test_ds)
print(accuracy)

examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)








