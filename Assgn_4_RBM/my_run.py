from my_rbm import RBM, CDTrainer
import numpy as np
import scipy.io.matlab as mio

#Data is /127.
mfile = mio.loadmat('tr_data.mat')
data = mfile['images']

rbm = RBM(784, 500)
#trainer = CDTrainer(rbm, lr=1e2, mr=0, wdecay=0)
trainer = CDTrainer(rbm)
trainer.train(data, n_epochs=50, cdsteps=1, batch_size=100)
np.save('rbm_model',rbm)
del mfile
print('Training Over')

print('Extracting features')
mfile = mio.loadmat('te_data.mat')
data = mfile['images']
test_hidden = rbm.get_hidden(data, sample=False)
np.save('my_test_hidden.npy', test_hidden)

print('Running Tsne')
from tsne import tsne
test_hidden = np.load('my_test_hidden.npy')
Y = tsne(test_hidden, 2, 50, 20.0, 200)
np.save('my_Y_orig', Y)

print('Plotting Data')
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as mio

Y = np.load('my_Y_orig.npy')
labels = mio.loadmat('te_data.mat')['labels']
colors = plt.cm.rainbow(np.linspace(0, 1, 10))

for i,c in enumerate(colors):
	pos = np.where(labels==i)
	plt.scatter(Y[pos[:-1], 0], Y[pos[:-1], 1], 10, c=c, label=i)
plt.legend(numpoints=1, bbox_to_anchor=(1.1, 1), ncol=1)
plt.show()
plt.savefig('my_tsne.png')


