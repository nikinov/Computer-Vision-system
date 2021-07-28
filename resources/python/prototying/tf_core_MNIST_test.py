import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt


# Properties
batch_size = 128
n_classes = 10

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# Normalize and reshape images to [28,28,1]
train_images  = np.expand_dims(train_images.astype(np.float32) / 255.0, axis=3)
test_images = np.expand_dims(test_images.astype(np.float32) / 255.0, axis=3)

train_labels = to_categorical(train_labels)

print(train_labels.shape)


dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
iterator = dataset.repeat().batch(batch_size).make_initializable_iterator()
data_batch = iterator.get_next()


sess = tf.Session()
"""
sess.run(iterator.initializer)


# Get the first batch of images and display first image
batch_images, batch_labels = sess.run(data_batch)
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(batch_images)[0], cmap='gray')

# Get a second batch of images and display first image
batch_images, _ = sess.run(data_batch)
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(batch_images)[0], cmap='gray')
plt.show()
"""

"""
# New session
sess = tf.Session()
sess.run(iterator.initializer)

# Get the first batch of images and display first image
batch_images, batch_labels = sess.run(data_batch)
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(batch_images)[0], cmap='gray')

# Close and restart session
sess.close()
sess = tf.Session()
sess.run(iterator.initializer)

# Get a second batch of images and display first image
batch_images, _ = sess.run(data_batch)
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(batch_images)[0], cmap='gray')
plt.show()
"""

weights = {
    # Convolution Layers
    'c1': tf.get_variable('W1', shape=(3, 3, 1, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'c2': tf.get_variable('W2', shape=(3, 3, 16, 16), initializer=tf.contrib.layers.xavier_initializer()),
    'c3': tf.get_variable('W3', shape=(3, 3, 16, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'c4': tf.get_variable('W4', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),

    # Dense Layers
    'd1': tf.get_variable('W5', shape=(7 * 7 * 32, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    # Convolution Layers
    'c1': tf.get_variable('B1', shape=(16), initializer=tf.zeros_initializer()),
    'c2': tf.get_variable('B2', shape=(16), initializer=tf.zeros_initializer()),
    'c3': tf.get_variable('B3', shape=(32), initializer=tf.zeros_initializer()),
    'c4': tf.get_variable('B4', shape=(32), initializer=tf.zeros_initializer()),

    # Dense Layers
    'd1': tf.get_variable('B5', shape=(128), initializer=tf.zeros_initializer()),
    'out': tf.get_variable('B6', shape=(n_classes), initializer=tf.zeros_initializer()),
}


#Define 2D convolutional function
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv_net(data, weights, biases, training=False):
    # Convolution layers
    conv1 = conv2d(data, weights['c1'], biases['c1']) # [28,28,16]
    conv2 = conv2d(conv1, weights['c2'], biases['c2']) # [28,28,16]
    pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # [14,14,16]

    conv3 = conv2d(pool1, weights['c3'], biases['c3']) # [14,14,32]
    conv4 = conv2d(conv3, weights['c4'], biases['c4']) # [14,14,32]
    pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # [7,7,32]

    # Flatten
    flat = tf.reshape(pool2, [-1, weights['d1'].get_shape().as_list()[0]])
    # [7*7*32] = [1568]

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flat, weights['d1']), biases['d1']) # [128]
    fc1 = tf.nn.relu(fc1) # [128]

    # Dropout
    if training:
        fc1 = tf.nn.dropout(fc1, rate=0.2)

    # Output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out']) # [10]
    return out

Xtrain = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
logits = conv_net(Xtrain, weights, biases)


ytrain = tf.placeholder(tf.float32, shape=(None, n_classes))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ytrain))


optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

test_predictions = tf.nn.softmax(conv_net(test_images, weights, biases))
acc,acc_op = tf.metrics.accuracy(predictions=tf.argmax(test_predictions,1), labels=test_labels)

batch_images, batch_labels = sess.run(data_batch)
feed_dict = {Xtrain: batch_images, ytrain: batch_labels}
sess.run(train_op, feed_dict=feed_dict)

nepochs = 5

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(iterator.initializer)

for epoch in range(nepochs):
    for step in tqdm(range(int(len(train_images) / batch_size))):
        # Batched data
        batch_images, batch_labels = sess.run(data_batch)

        # Train model
        feed_dict = {Xtrain: batch_images, ytrain: batch_labels}
        sess.run(train_op, feed_dict=feed_dict)

    # Test model
    accuracy = sess.run(acc_op)

    print('\nEpoch {} Accuracy: {}'.format(epoch + 1, accuracy))











