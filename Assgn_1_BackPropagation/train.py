import argparse
import collections
import cPickle
import pickle
import gzip
import copy
import os
from Network import Network
import numpy as np
from dataset import DataSet

update_w = []
update_b = []
m_w = []
m_b = []
v_w = []
v_b = []

beta_1_t = 1
beta_2_t = 1

def read_dataset(folder_name, debug=False):
    f = gzip.open(folder_name, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    n_samples = train_set[0].shape[0]

    if debug:
        n_samples = 10000

    datasets_template = collections.namedtuple('Datasets_template', ['train', 'validation', 'test'])
    Datasets = datasets_template(train=DataSet(train_set[0][:n_samples, :], train_set[1]),
                                 validation=DataSet(valid_set[0], valid_set[1]), test=DataSet(test_set[0], test_set[1]))
    return Datasets


def get_argumentparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, help="initial learning rate for gradient descent based algorithms")
    parser.add_argument("--momentum", default=0, help="momentum to be used by momentum based algorithms")
    parser.add_argument("--num_hidden", default=3,
                        help="number of hidden layers - this does not include the 784 dimensional input layer and the 10 dimensional output layer")
    parser.add_argument("--sizes", default='128,96,32', help="a comma separated list for the size of each hidden layer")
    parser.add_argument("--activation", default='sigmoid', help="the choice of activation function - valid values are tanh/sigmoid")
    parser.add_argument("--loss", default='ce', help="possible choices are squared error[sq] or cross entropy loss[ce]")
    parser.add_argument("--opt", default='gd', help="the optimization algorithm to be used: gd, momentum, nag, adam")
    parser.add_argument("--batch_size", default=20,
                        help="the batch size to be used - valid values are 1 and multiples of 5")
    parser.add_argument("--anneal", default=False,
                        help="halve the learning rate if at any epoch the validation loss decreases and then restart the epoch")
    parser.add_argument("--save_dir", help="the directory in which the pickled model should be saved")
    parser.add_argument("--expt_dir", help="the directory in which the log files will be saved")
    parser.add_argument("--mnist", default='../mnist.pkl.gz', help="path to the mnist data in pickeled format")
    parser.add_argument("--max_epoch", default=150, help="Maximum epochs")
    parser.add_argument("--debug", default=False, help="Debug mode")
    return parser


def predict(x, y, network, all_layers=False):

    activations = []
    layer = 0
    activations.append(x)

    for layer in xrange(1, network.n_layers - 1):
        activations.append(network.layers[layer].feed_fwd(activations[layer - 1]))
    predictions = activations[layer]

    # Compute cost
    cost = network.layers[network.n_layers - 1](predictions, y)

    if not all_layers:
        return predictions, np.mean(cost)
    else:
        return activations, np.mean(cost)

def compute_gradients(y, activations, network):

    predictions = activations[network.n_layers - 2]
    layer = network.n_layers - 1
    gradients = {}

    #Gradient w.r.t Loss layer
    gradients[layer] = network.layers[layer].delta(predictions, y).T

    #Gradient w.r.t Output layer
    above = gradients[layer]
    layer -= 1
    gradients[layer] = network.layers[layer].activation.delta(activations[layer - 1].T, activations[layer].T, above)

    #Gradient w.r.t Hidden layers
    for layer in range(network.n_layers - 3, 0, -1):
        above = np.matmul(network.layers[layer + 1].weights, gradients[layer + 1])
        gradients[layer] = network.layers[layer].activation.delta(activations[layer - 1].T, activations[layer].T, above)
    return gradients

def update_parameters(activations, gradients, network, lr, args, x, y):

    batch_size = int(args.batch_size)
    momentum = float(args.momentum)
    if args.opt == 'sgd':
        momentum = 0

    if args.opt == 'nag':
        t_update_w = copy.deepcopy(update_w)
        t_update_b = copy.deepcopy(update_b)
        t_network = copy.deepcopy(network)

        for layer in xrange(1, network.n_layers - 1):
            #compute gradients w.r.t weight and bias

            t_update_w[layer - 1] = float(args.momentum)*update_w[layer - 1] + lr*(np.matmul(np.transpose(activations[layer-1]), gradients[layer].T)/batch_size)
            t_update_b[layer - 1] = float(args.momentum)*update_b[layer - 1] + lr*np.mean(np.transpose(gradients[layer].T), axis=1)

            t_network.layers[layer].weights -= t_update_w[layer - 1]
            t_network.layers[layer].bias -= t_update_b[layer - 1]

            t_activations, _ = predict(x, y, t_network, all_layers=True)

            # Compute Gradients by Backpropogation
            gradients = compute_gradients(y, t_activations, t_network)
        del t_network, t_activations, t_update_b, t_update_w


    for layer in xrange(1, network.n_layers - 1):

        if args.opt == 'adam':

            beta_1 = 0.9
            beta_2 = 0.999
            eps = 1e-8

            global beta_1_t, beta_2_t
            beta_1_t *= beta_1
            beta_2_t *= beta_2

            grad_w = (np.matmul(np.transpose(activations[layer-1]), gradients[layer].T)/batch_size)
            grad_b = np.mean(np.transpose(gradients[layer].T), axis=1)

            m_w[layer-1] = beta_1*m_w[layer-1] + (1-beta_1)*grad_w
            m_b[layer-1] = beta_1*m_b[layer-1] + (1-beta_1)*grad_b

            v_w[layer - 1] = beta_2*v_w[layer - 1] + (1 - beta_2)*grad_w*grad_w
            v_b[layer - 1] = beta_2*v_b[layer - 1] + (1 - beta_2)*grad_b*grad_b

            lr_t = lr*np.sqrt(1-beta_2_t)/(1-beta_1_t)

            #print([network.layers[layer].bias.shape, v_b[layer - 1].shape, m_b[layer-1].shape, np.sqrt(v_b[layer-1] + eps).shape])
            network.layers[layer].weights -= (lr_t * m_w[layer - 1]) / (np.sqrt(v_w[layer-1] + eps))
            network.layers[layer].bias -= (lr_t * m_b[layer-1]) / (np.sqrt(v_b[layer-1] + eps))

            continue


        update_w[layer - 1] = momentum*update_w[layer-1] + lr*(np.matmul(np.transpose(activations[layer-1]), gradients[layer].T)/batch_size)
        update_b[layer - 1] = momentum*update_b[layer-1] + lr*np.mean(np.transpose(gradients[layer].T), axis=1)

        network.layers[layer].weights -= update_w[layer-1]
        network.layers[layer].bias -= update_b[layer-1]

    return update_w, update_b


def run_epoch(epoch_id, network, args, lr, batch_size=None):

    step = 0
    for x, y in Datasets.train.next_batch(batch_size):
        step += 1

        # Feed forward
        activations, cost = predict(x, y, network, all_layers=True)
        predictions = activations[-1]

        # Compute gradients
        gradients = compute_gradients(y, activations, network)

        # Update weight
        update_parameters(activations, gradients, network, lr, args, x, y)

        if step % 100 == 0:
            predictions, tr_loss = predict(Datasets.train.x, Datasets.train.y, network, all_layers=False)
            predictions = np.argmax(predictions, axis=1)
            tr_acc = round(np.count_nonzero(predictions == Datasets.train.y) / float(Datasets.train.n_samples) * 100, 2)
            tr_err = round(np.count_nonzero(predictions != Datasets.train.y) / float(Datasets.train.n_samples) * 100, 2)
            tr_loss_f.write('Epoch '+str(epoch_id)+', Step '+str(step)+', Loss: '+str(tr_loss)+', lr: '+str(lr)+'\n')
            tr_err_f.write('Epoch ' + str(epoch_id) + ', Step ' + str(step) + ', Error: ' + str(tr_err) + ', lr: ' + str(lr) + '\n')

            predictions, val_loss = predict(Datasets.validation.x, Datasets.validation.y, network, all_layers=False)
            predictions = np.argmax(predictions, axis=1)
            np.savetxt(os.path.join(args.expt_dir,'valid_predictions.txt'), predictions, fmt='%d')
            val_acc = round(np.count_nonzero(predictions == Datasets.validation.y) / float(Datasets.validation.n_samples) * 100, 2)
            val_err = round(np.count_nonzero(predictions != Datasets.validation.y) / float(Datasets.validation.n_samples) * 100, 2)
            val_loss_f.write('Epoch '+str(epoch_id)+', Step '+str(step)+', Loss: '+str(val_loss)+', lr: '+str(lr)+'\n')
            val_err_f.write('Epoch ' + str(epoch_id) + ', Step ' + str(step) + ', Error: ' + str(val_err) + ', lr: ' + str(lr) + '\n')

            predictions, te_loss = predict(Datasets.test.x, Datasets.test.y, network, all_layers=False)
            predictions = np.argmax(predictions, axis=1)
            np.savetxt(os.path.join(args.expt_dir, 'test_predictions.txt'), predictions, fmt='%d')
            te_acc = round(np.count_nonzero(predictions == Datasets.test.y) / float(Datasets.test.n_samples) * 100, 2)
            te_err = round(np.count_nonzero(predictions != Datasets.test.y) / float(Datasets.test.n_samples) * 100, 2)
            te_loss_f.write('Epoch '+str(epoch_id)+', Step '+str(step)+', Loss: '+str(te_loss)+', lr: '+str(lr)+'\n')
            te_err_f.write('Epoch ' + str(epoch_id) + ', Step ' + str(step) + ', Error: ' + str(te_err) + ', lr: ' + str(lr) + '\n')
            #print(step)

    return tr_loss, val_loss, te_loss, tr_acc, val_acc, te_acc


def init_updates(network,opt):

    for layer in xrange(1, network.n_layers - 1):
        update_w.append(np.zeros(network.layers[layer].weights.shape))
        update_b.append(np.zeros(network.layers[layer].bias.shape))

        if opt == 'adam':
            m_w.append(np.zeros(network.layers[layer].weights.shape))
            m_b.append(np.zeros(network.layers[layer].bias.shape))

            v_w.append(np.zeros(network.layers[layer].weights.shape))
            v_b.append(np.zeros(network.layers[layer].bias.shape))


def main():
    # Parser parameters
    parser = get_argumentparser()
    args = parser.parse_args()

    #Create Model_Save and Log Directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.expt_dir):
        os.makedirs(args.expt_dir)

    # Create loss log files
    global tr_loss_f, te_loss_f, val_loss_f
    tr_loss_f = open(os.path.join(args.expt_dir, 'log_loss_train.txt'), 'w')
    val_loss_f = open(os.path.join(args.expt_dir, 'log_loss_valid.txt'), 'w')
    te_loss_f = open(os.path.join(args.expt_dir, 'log_loss_test.txt'), 'w')

    # Create error log files
    global tr_err_f, te_err_f, val_err_f
    tr_err_f = open(os.path.join(args.expt_dir, 'log_err_train.txt'), 'w')
    val_err_f = open(os.path.join(args.expt_dir, 'log_err_valid.txt'), 'w')
    te_err_f = open(os.path.join(args.expt_dir, 'log_err_test.txt'), 'w')

    # Get Dataset
    global Datasets
    Datasets = read_dataset(args.mnist, args.debug)

    # Create the feed forward network
    network = Network(args, Datasets.train.n_features, Datasets.train.n_labels)

    # Initialize previous updates for optimization
    init_updates(network, args.opt)
    past_dump = Batch_Dump()

    # Training
    rate = float(args.lr)
    val_best = 10
    patience = 1

    #with open(os.path.join(args.save_dir, 'model.pkl'), 'rb') as model:
    #    network = pickle.load(model)

    for epoch in xrange(1,args.max_epoch):

        tr_loss, val_loss, te_loss, tr_acc, val_acc, te_acc = run_epoch(epoch, network, args, lr=rate, batch_size=int(args.batch_size))
        #print([epoch, tr_loss, val_loss, te_loss, val_best, tr_acc, val_acc, te_acc, rate])

        #Annealing
        if (val_best - val_loss) > 0.0000001:
            val_best = val_loss
            old_network = copy.deepcopy(network)
        else:
            if args.anneal == 'True' or args.anneal == 'true':
                rate /= 2
                network = copy.deepcopy(old_network)
                #Stop annealing by patience
                patience += 1
                if patience > 10:
                    break
                print 'Learning rate dropped to '+str(rate)
                #restore_weights from previous epoch
            else:
                patience += 1
                if patience > 2:
                    break

    with open(os.path.join(args.save_dir, 'model.pkl'), 'w') as model:
        pickle.dump(network, model)

    tr_loss_f.close()
    te_loss_f.close()
    val_loss_f.close()
    tr_err_f.close()
    te_err_f.close()
    val_err_f.close()

class Batch_Dump:
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.loss = {}

if __name__ == '__main__':
    main()
