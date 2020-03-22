import tensorflow as tf
import keras.backend as K
def decode(serialized_example):
    """decode the serialized example"""   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'train_input1': tf.FixedLenFeature([], tf.string),
                                           'train_input2': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
    train_input_10 = tf.decode_raw(features['train_input1'], tf.float32)
    train_input_10 = tf.reshape(train_input_10, [4, 32, 32])
    #train_input_10 = tf.cast(train_input_10,tf.float32)
    train_input_20 = tf.decode_raw(features['train_input2'], tf.float32)
    train_input_20 = tf.reshape(train_input_20, [6, 32,32])
    #train_input_20=tf.cast(train_input_20, tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    label = tf.reshape(label, [6, 32, 32])
    #label=tf.cast(label, tf.float32)    
    return train_input_10, train_input_20, label

dataset = tf.data.TFRecordDataset("../DSen2/data/train_test.tfrecords")
#dataset = dataset.repeat(n_repeats) # for train
#dset = dset.shard(hvd.size(), hvd.rank())
dataset = dataset.map(decode) # decode and normalize    
dataset = dataset.shuffle(1000) # shuffle
dataset = dataset.batch(1,drop_remainder=True)
iterator = dataset.make_initializable_iterator()
train_input_10, train_input_20, label = iterator.get_next()
#train_input_10, train_input_20, label = iterator[0], iterator[1], iterator[2]
#####################################
#print (label.shape)
#print (iterator[0].shape)
batch_count = 1000
sess = tf.Session()
#for _ in range(0,1):
sess.run(iterator.initializer)
value = sess.run(train_input_10)
print (value)
value = sess.run(train_input_20)
print (value)
value = sess.run(label)
print (value)


