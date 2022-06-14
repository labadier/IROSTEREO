
import os

for i in [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99]:
  os.system(f"python main.py -l en  -mode gcn -tp data/pan22-author-profiling-training-2022-03-29/ \
            -epoches 28 -lr 2e-3 -decay 2e-5 -phase train -bs 8 -interm_layer 32 -beta 0.{i}")
