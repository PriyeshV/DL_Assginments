import tensorflow as tf
import matplotlib.pyplot as plt
import cifar10_data_queue as data_queues

tr_images, tr_labels = data_queues.inputs("train", "dataset", 10)


with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.initialize_all_variables())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print "from the train set:"
    for i in range(20):
        images = sess.run(tr_images)
        print(images.shape)
        plt.imshow(images[0])
        plt.show()

    coord.request_stop()
    coord.join(threads)
    sess.close()