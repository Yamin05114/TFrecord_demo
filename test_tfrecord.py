import Layers
import tensorflow as tf
import matplotlib.pyplot as plt

# data flow:
Layers.images2tfrecord(['./images/1.jpg'], [1], './tfrecord', 'test_tfrecord')

image_batch, label_batch = Layers.tfrecord_decode(['./tfrecord/test_tfrecord.tfrecords'], batch_size=1)


# session:
with tf.Session() as sess:  # sess will call the CUDA operations
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(20000):

        try:
            while not coord.should_stop() and i < 1:
                image, label = sess.run([image_batch, label_batch])  # this sentence is wrong?
                plt.figure()
                plt.imshow(image[0])
                plt.show()
                print('to here now')
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
    coord.join(threads)






