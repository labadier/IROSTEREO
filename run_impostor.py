
import os

for i in (10, 51, 5):
  os.system(f"python main.py -l en -mode impostor -tp data/pan22-author-profiling-training-2022-03-29 \
    -dp data/pan22-author-profiling-test-2022-04-22-without_truth -output outputs -port 0.{i}")
