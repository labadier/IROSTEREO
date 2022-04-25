#!/bin/bash

# Train irony

python main.py -mode encoder -phase train -lr 1e-5 -decay 1e-5 -tmode offline -interm_layer 64 -epoches 12 -bs 64 -l en -mtl stl -tp data/irony_train.csv -dp data/irony_test.csv




# Train encoder multitasking hate-stereotipe irony

python main.py -mode encoder -phase train -lr 1e-5 -decay 1e-5 -tmode offline -interm_layer 64 -epoches 12 -bs 64 -l en -mtl mtl -tp data/augmented_train.csv -dp data/augmented_test.csv



###
# python main.py -mode encoder -phase train -lr 1e-5 -decay 1e-5 -tmode offline -interm_layer 64 -epoches 1 -bs 8 -l en -tp data/hater/train

# python main.py -mode encoder -phase encode -tmode offline -interm_layer 64 -bs 16 -l en -tp data/hater/train -wp logs
# python main.py -mode encoder -phase encode -tmode offline -interm_layer 64 -bs 16 -l en -tp data/hater/dev -wp logs
# python main.py -l en  -mode gcn -tp data/hater/train -epoches 15 -lr 1e-4 -decay 0 -phase train -bs 16 -interm_layer 64



# ### Graph

# #train
# python main.py -l en  -mode gcn -tp data/hater/train -epoches 15 -lr 1e-1 -decay 0 -phase train -bs 32 -interm_layer 16

# #predict
# python main.py -l en  -mode gcn -dp data/hater/dev -splits 5 -phase test -bs 2 -interm_layer 16 -output outputs

# #encode
# python main.py -l en  -mode gcn -dp data/hater/train -phase encode -bs 200 -interm_layer 16 -output outputs

# #eval
# python main.py -mode eval -dp data/hater/dev -l en
