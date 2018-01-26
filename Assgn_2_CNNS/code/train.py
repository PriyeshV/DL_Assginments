import argparse
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
#import network as architecture
import network_bn as architecture
import cifar10_data_queue as cifar_queues
import time, math, sys, os
import numpy as np
from random import randint


@ops.RegisterGradient("GuidedRelu")
def _GuidedRelu(op, grad):
    global tmp
    tmp = tf.select(0.<grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))
    return tmp

class CifarModel(object):

    def __init__(self,config):
        self.config = config
        self.patience = self.config.patience

        self.create_queues()
        self.add_placeholders()
        # self.graph = tf.get_default_graph()

        self.arch = self.add_network()

        with tf.variable_scope("predict") as scope:
            self.tr_output = self.arch.predict(self.tr_images, self.keep_prob, self.training)
            #self.tr_output = self.arch.predict(self.tr_eval_images, self.keep_prob, self.training)
            scope.reuse_variables()
            self.val_output = self.arch.predict(self.val_images, self.keep_prob, self.training)
            scope.reuse_variables()
            self.te_output = self.arch.predict(self.te_images, self.keep_prob, self.training)

        with tf.variable_scope("n_corr_predictions") as scope:
            self.tr_n_corr_pred = self.arch.get_n_correct_predictions(self.tr_output, self.tr_labels)
            #self.tr_n_corr_pred = self.arch.get_n_correct_predictions(self.tr_output, self.tr_eval_labels)
            scope.reuse_variables()
            self.val_n_corr_pred = self.arch.get_n_correct_predictions(self.val_output, self.val_labels)
            scope.reuse_variables()
            self.te_n_corr_pred = self.arch.get_n_correct_predictions(self.te_output, self.te_labels)

        self.loss = self.arch.loss(self.tr_output, self.tr_labels)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train = self.arch.training(self.loss, tf.train.AdamOptimizer(self.config.lr))

        self.check_op = tf.add_check_numerics_ops()
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        self.summary = tf.merge_all_summaries()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step + 1)
        self.init = tf.initialize_all_variables()
        #self.init = tf.global_variables_initializer()

    def create_queues(self):
        self.tr_images, self.tr_labels = cifar_queues.distorted_inputs(self.config.data_dir, self.config.batch_size)
        self.tr_eval_images, self.tr_eval_labels = cifar_queues.inputs("train", self.config.data_dir, self.config.batch_size)
        self.val_images, self.val_labels = cifar_queues.inputs('validation', self.config.data_dir, self.config.batch_size)
        self.te_images, self.te_labels = cifar_queues.inputs('test', self.config.data_dir, self.config.batch_size)

    def add_placeholders(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool, name='training')

    def add_network(self):
        return architecture.Network(self.config)

    def add_metrics(self):
        """assign and add summary to a metric tensor"""
        for i, metric in enumerate(self.config.metrics):
            tf.scalar_summary(metric, self.metrics[i])

    def add_summaries(self, sess):
        self.summary_writer_train = tf.train.SummaryWriter(self.config.logs_dir + "train", sess.graph)
        self.summary_writer_val = tf.train.SummaryWriter(self.config.logs_dir + "val", sess.graph)
        self.summary_writer_test = tf.train.SummaryWriter(self.config.logs_dir + "test", sess.graph)

    def write_summary(self, sess, summary_writer, metric_values, step, feed_dict):
        summary = self.merged_summary
        #feed_dict[self.loss]=loss
        feed_dict[self.metrics] = metric_values
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    def run_epoch(self, sess, phase='test', summary_writer=None, verbose = 10):

        total_loss = []
        total_correct = []

        feed_dict = {}
        if phase == "validation":
            train_op = tf.no_op()
            keep_prob = 1
            feed_dict[self.training] = False
            n_samples = cifar_queues.N_VAL
            total_steps = n_samples/ self.config.batch_size
            corr_pred = self.val_n_corr_pred
        elif phase == "train":
            train_op = self.train
            feed_dict[self.training] = True
            keep_prob = self.config.dropout
            n_samples = cifar_queues.N_TRAIN
            total_steps = n_samples/self.config.batch_size
            corr_pred = self.tr_n_corr_pred
        else:
            train_op = tf.no_op()
            keep_prob = 1
            feed_dict[self.training] = False
            n_samples = cifar_queues.N_TEST
            total_steps = n_samples/ self.config.batch_size
            corr_pred = self.te_n_corr_pred
        feed_dict[self.keep_prob] = keep_prob

        for step in range(total_steps):
            _, loss_value, correct_pred = sess.run([train_op, self.loss, corr_pred], feed_dict=feed_dict)

            #op_value = sess.run([self.tmp],feed_dict=feed_dict)
            #import matplotlib.pyplot as plt
            #plt.imshow(op_value[0][0])
            #plt.show()

            total_loss.append(loss_value)
            total_correct.append(correct_pred)
            #tstep +=1
            #print([step,correct_pred])
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : Loss = {} acc = {}'.format(step, total_steps, np.mean(total_loss), np.sum(total_correct)/((step+1.)*self.config.batch_size)))
                sys.stdout.flush()
        accuracy = np.sum(total_correct)/float(n_samples)
        sys.stdout.write('\r')

        return accuracy, np.mean(total_loss)

    def fit(self, sess):
        max_epochs = self.config.max_epochs
        save_epochs_after = 1
        patience = self.config.patience
        improvement_threshold = 0.90

        validation_loss = 0.20
        done_looping = False
        step = 1
        best_step = -1
        learning_rate = self.config.lr
        val_epoch_freq = 1
        losses = []
        accuracy = []

        while (step <= self.config.max_epochs) and (not done_looping):
            sess.run([self.step_incr_op])
            epoch = self.arch.global_step.eval(session=sess)

            start_time = time.time()
            tr_acc, tr_loss = self.run_epoch(sess, phase="train", summary_writer=self.summary_writer_train)
            duration = time.time() - start_time

            if (epoch % val_epoch_freq == 0):
                val_acc, val_loss = self.run_epoch(sess, phase="validation", summary_writer=self.summary_writer_val)

                te_acc, te_loss = self.run_epoch(sess, phase="test", summary_writer=self.summary_writer_test)

                print('Epoch %d: tr_loss = %.2f, val_loss = %.2f, te_loss = %.2f || tr_acc = %.2f, val_acc = %.2f te_acc  = %.2f (%.3f sec)'
                      % (epoch, tr_loss, val_loss, te_loss, tr_acc, val_acc, te_acc, duration))

                if val_loss < validation_loss:
                    validation_loss = val_loss
                    checkpoint_file = os.path.join(self.config.ckpt_dir, 'checkpoint')
                    self.saver.save(sess, checkpoint_file, global_step=epoch)
                    best_step = epoch
                    if (val_loss < validation_loss * improvement_threshold):
                        self.patience = self.config.patience

                # elif val_loss > validation_loss * improvement_threshold:
                else:
                    if self.patience < 1:
                        if self.config.lr <= 0.00000001:
                            print('Stopping by patience method')
                            done_looping = True
                            break
                        else:
                            self.config.lr /= 5
                            self.patience = self.config.patience
                            print 'Learning rate dropped to %.8f' % (self.config.lr)
                    else:
                        self.patience -= 1
            else:
                # Print status to stdout.
                print('Epoch %d: loss = %.2f acc = %.2f (%.3f sec)' % (epoch, tr_loss, tr_acc, duration))
            losses.append([tr_loss, val_loss, te_loss])
            accuracy.append([tr_acc, val_acc, te_acc])
            step += 1

        return losses, accuracy, best_step

def get_argumentparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, help="Debug mode")
    parser.add_argument("--lr", default=0.001, help="initial learning rate for gradient descent based algorithms")
    parser.add_argument("--batch_size", default=100,
                        help="the batch size to be used - valid values are 1 and multiples of 5")
    parser.add_argument("--init", default=2, help="he or xavier")
    parser.add_argument("--batch_norm", default=True, help="True or False")
    parser.add_argument("--anneal", default=False,
                        help="halve the learning rate if at any epoch the validation loss decreases and then restart the epoch")
    parser.add_argument("--save_dir", default='Ckpt_dir/', help="the directory in which the pickled model should be saved")
    parser.add_argument("--data_dir", default='dataset/', help="path to the cifar data in pickeled format")
    parser.add_argument("--logs_dir", default='Log_dir/', help="path to the Log dir")
    parser.add_argument("--ckpt_dir", default='Ckpt_dir/', help="path to the Log dir")

    parser.add_argument("--max_epochs", default=300, help="Maximum epochs")
    parser.add_argument("--patience", default=5, help="Patience")
    parser.add_argument("--dropout", default=0.5, help="Dropout Keep Probability")
    parser.add_argument("--retrain", default=True, help="1: Retrain from save_dir")
    return parser

def init_Model(config):
    tf.reset_default_graph()
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        with tf.variable_scope('CNN', reuse=None) as scope:
            model = CifarModel(config)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        print('inside retrain')
        load_ckpt_dir = config.ckpt_dir
        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            load_ckpt_dir = ''
        print('--------- Loading variables from checkpoint if available')
    else:
        # Delete folder if available
        load_ckpt_dir = ''
        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
        print('--------- Training from scratch')
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tfconfig)
    return model, sess

def train():
    parser = get_argumentparser()
    config = parser.parse_args()
    model, sess = init_Model(config)
    writer = tf.train.SummaryWriter(config.logs_dir+'tf_graphs', sess.graph)
    with sess:
        model.add_summaries(sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        losses, accuracy, best_step = model.fit(sess)

        coord.request_stop()
        coord.join(threads)

        np.save(config.logs_dir+'losses.npy',losses)
        np.save(config.logs_dir+'accuracy.npy',accuracy)

def visualize_filters():
    parser = get_argumentparser()
    config = parser.parse_args()
    model, sess = init_Model(config)
    with sess:
        w_conv_1 = tf.get_default_graph().get_tensor_by_name("CNN/wc1:0")
        w_conv_1 = sess.run([w_conv_1])[0]

    images = w_conv_1
    min = np.min(images)
    max = np.max(images)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(8, 8)
    fig.subplots_adjust(hspace=1, wspace=0.8)
    for i, ax in enumerate(axes.flat):
        image = images[:,:,:, i]
        #image = (image - np.min(images)) / (np.max(images) - np.min(images))
        image = (image - min) / (max - min)
        ax.imshow(image)
        xlabel = str(i+1)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Visualization of conv-1 layer filters')
    #plt.show()
    plt.savefig(config.logs_dir+'conv1_imgs_n',format='png')

def get_interesting_neurons():
    parser = get_argumentparser()
    config = parser.parse_args()
    model, sess = init_Model(config)

    with sess:
        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        conv3 = tf.get_default_graph().get_tensor_by_name("CNN/predict/conv3:0")

        feed_dict = {}
        feed_dict[model.training] = False
        feed_dict[model.keep_prob] = 1
        n_samples = cifar_queues.N_TRAIN
        total_steps = n_samples / config.batch_size

        class_neurons = np.zeros((10,8,8,256))

        for step in range(total_steps):
           outputs, labels = sess.run([conv3, model.tr_labels], feed_dict=feed_dict)
           label_ids = np.where(labels)[1]

           for i in range(10):
                pos = np.where(label_ids == i)[0]
                val = class_neurons[i, ...] + np.sum(outputs[pos, ...], axis=0)
                class_neurons[i, ...] = val

        neurons = []
        for i in range(10):
            neurons.append(np.argmax(class_neurons[i,...]))
        print neurons, len(neurons)
        #exit()

        n_int_images = 10
        interesting_images = np.zeros((10,n_int_images,32,32,3))
        image_confidence = np.zeros((10, n_int_images))
        neurons_cnt = np.zeros(10)
        for step in range(total_steps):
            outputs, labels, images = sess.run([conv3, model.tr_labels, model.tr_images], feed_dict=feed_dict)
            label_ids = np.where(labels)[1]

            for sample in range(config.batch_size):
                id = np.argmax(outputs[sample, ...])
                conf = np.max(outputs[sample, ...])
                label_id = np.where(labels[sample, :])[0]
                # print neurons, 'Hi \n'
                if id in neurons:
                    neurons_cnt[np.where(neurons == id)[0]] += 1

                    if (conf > image_confidence[label_id, :]).any():
                        pos = np.argmin(image_confidence[label_id, :])
                        image_confidence[label_id, pos] = conf
                        interesting_images[label_id,pos, ...] = images[sample, ...]
        coord.request_stop()
        coord.join(threads)

    print neurons_cnt
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(10, n_int_images)
    #fig.subplots_adjust(hspace=0.05, wspace=0.8)
    print np.min(images), np.max(images)

    for i, ax in enumerate(axes.flat):
        pos = np.unravel_index(neurons[i/n_int_images], (8,8,256))
        image = interesting_images[i/n_int_images , i % n_int_images , ...]
        image = (image - np.min(images))/(np.max(images)- np.min(images))

        x = (((2*pos[1]+2)*2)-4)
        y = (((2*pos[0]+2)*2)-4)

        boundary = patches.Rectangle((x,y-7),18,18,linewidth=1,edgecolor='r',facecolor='none')

        ax.imshow(image)
        ax.add_patch(boundary)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.suptitle('Interesting neurons')
        #plt.show()
        plt.savefig('neurons16N.png')

def train_guided_backprop():
    parser = get_argumentparser()
    config = parser.parse_args()
    config.ckpt_dir = config.save_dir
    model, sess = init_Model(config)
    #writer = tf.train.SummaryWriter(config.logs_dir+'tf_graphs', sess.graph)
    n_images = 10
    with sess:
        g = tf.get_default_graph()

        conv3 = model.arch.conv3op
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #conv3 = tf.get_default_graph().get_tensor_by_name("CNN/predict/conv3:0")

        feed_dict = {}
        feed_dict[model.training] = False
        feed_dict[model.keep_prob] = 1
        conv3_ops = sess.run([conv3], feed_dict=feed_dict)[0]
        print np.shape(conv3_ops)

        pos = np.ones((n_images,4)).astype(int)
        val = np.ones(n_images)
        image_ops = conv3_ops[0]

        coord.request_stop()
        coord.join(threads)


    model, sess = init_Model(config)
    ip_image = np.zeros((100, 32, 32, 3))
    with sess:
        g = tf.get_default_graph()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        feed_dict = {}
        feed_dict[model.training] = False
        feed_dict[model.keep_prob] = 1
        images = sess.run([model.arch.x], feed_dict=feed_dict)[0]
        print images[0].shape


        max = np.zeros((100))
        min = np.zeros((100))
        for i in range(100):
            ip_image[i,...] = images[i, ...]
            max[i] = np.max(ip_image[i,...])
            min[i] = np.min(ip_image[i,...])
            ip_image[i,...] = (ip_image[i,...] - min[i]) / (max[i] - min[i])

        coord.request_stop()
        coord.join(threads)

    for i in range(100):
        del model, coord, threads
        tmp = np.zeros((100, 8, 8, 256))
        pos = np.argmax(conv3_ops[i, ...])
        pos = np.unravel_index(pos, (8, 8, 256))
        val = np.max(conv3_ops[i, ...])
        tmp[i, pos[0], pos[1], pos[2]] = val

        model, sess = init_Model(config)
        with sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            conv3 = model.arch.conv3op
            feed_dict = {}
            feed_dict[model.training] = False
            feed_dict[model.keep_prob] = 1
            feed_dict[conv3] = tmp

            im_output = tf.gradients(conv3, model.arch.x)

            with g.gradient_override_map({'Relu': '_GuidedRelu'}):

                gen_image = sess.run([im_output], feed_dict=feed_dict)
                gen_image = gen_image[0][0]
                image = gen_image[i , ...]
                image = (image - min[i]) / (max[i] - min[i])
                #image = (image - np.min(image)) / (np.max(image) - np.min(image))

            print i, "hi"
            import matplotlib.pyplot as plt
            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(ip_image[i])
            a.set_title('Before')
            plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(image)
            imgplot.set_clim(0.0, 0.7)
            a.set_title('After')
            plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
            plt.savefig('Guided_BackProp_images/figure_neuron'+str(i))
            # plt.show()
            plt.close()

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        #os.system("create_train_val.py")
        # train()
        #visualize_filters()
        #get_interesting_neurons()
        train_guided_backprop()