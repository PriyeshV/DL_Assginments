#!/bin/sh

#  run.sh
#  PA_1
#
#  Created by Priyesh Vijayan on 09/02/17.
#  Copyright © 2017 Priyesh Vijayan. All rights reserved.

python train.py --lr 0.001 --momentum 0.5 --num_hidden 2 --sizes 625,625 --activation leaky_relu --loss ce --opt adam --batch_size 100 --anneal true --save_dir models/ --expt_dir logs/ --mnist /Users/priyesh/Desktop/Courses/mnist.pkl.gz
