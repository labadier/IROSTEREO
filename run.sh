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
# python main.py -l en  -mode gcn -tp data/hater/train -epoches 15 -lr 1e-3 -decay 2e-5 -phase train -bs 16 -interm_layer 32 -dp data/hater/dev/
python main.py -l en  -mode gcn -tp data/pan22-author-profiling-training-2022-03-29/ -epoches 28 -lr 2e-3 -decay 2e-5 -phase train -bs 8 -interm_layer 32


# #predict
python main.py -l en  -mode gcn -dp data/hater/dev -splits 1 -phase test -bs 64 -interm_layer 32 -output outputs

# #encode
python main.py -l en  -mode gcn -dp data/hater/train -phase encode -bs 200 -interm_layer 32 -output outputs

# #eval
python main.py -mode eval -dp data/hater/dev -l en
