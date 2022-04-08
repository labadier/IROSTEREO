#!/bin/bash

python main.py -mode encoder -phase train -lr 1e-5 -decay 1e-5 -tmode offline -interm_layer 64 -epoches 1 -bs 8 -l en -dp data/hater/train

python main.py -mode encoder -phase encode -tmode offline -interm_layer 64 -bs 16 -l en -dp data/hater/train -wp logs
python main.py -mode encoder -phase encode -tmode offline -interm_layer 64 -bs 16 -l en -dp data/hater/dev -wp logs
# python main.py -l EN  -mode cgn -dp data/pan22-author-profiling-training-2022-03-29 -epoches 60 -lr 1e-4 -decay 0 -phase train -bs 16 -interm_layer 64


# python main.py -mode encoder -phase train -lr 1e-5 -decay 1e-5 -tmode offline -interm_layer 64 -epoches 1 -bs 8 -l en -dp data/pan22-author-profiling-training-2022-03-29
# python main.py -mode encoder -phase encode -tmode offline -interm_layer 64 -bs 16 -l en -dp data/pan22-author-profiling-training-2022-03-29 -wp logs

# python main.py -l EN  -mode cgn -dp data/pan22-author-profiling-training-2022-03-29 -epoches 60 -lr 1e-4 -decay 0 -phase train -bs 16 -interm_layer 64
