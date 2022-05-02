#!/bin/bash

dp=/home/nitro/Desktop/PAN/things/pan20-author-profiling-training-2020-02-23/pan20-author-profiling-training-2020-02-23
op=/home/nitro/Desktop/hola
phase=train

bs=200

while getopts ":i:o:p:" option;
do
 case $option in
 i)
   dp=$OPTARG
   echo received data path as $dp
   ;;
 o)
   op=$OPTARG
   echo received output dir as $op
 ;; 
 p)
   phase=$OPTARG
   echo received phase dir as $phase
 ;;
 *)
   echo invalid command
   ;;
 esac
done

# encode for irony
python main.py -mode encoder -phase encode -tmode online -interm_layer 64 -bs $bs -l en -tp $dp -wp logs -mtl stl -model cardiffnlp/twitter-roberta-base-irony

mkdir -p logs/irony
mv logs/$phase\_penc_en.pt logs/irony/

# encode for hate
python main.py -mode encoder -phase encode -tmode online -interm_layer 64 -bs $bs -l en -tp $dp -wp logs -mtl stl -model Hate-speech-CNERG/bert-base-uncased-hatexplain

mkdir -p logs/hate
mv logs/$phase\_penc_en.pt logs/hate/

# encode for raw
python main.py -mode encoder -phase encode -tmode online -interm_layer 64 -bs $bs -l en -tp $dp -mtl stl -model bert-base-uncased 

mkdir -p logs/raw
mv logs/$phase\_penc_en.pt logs/raw/


#predict graph
python main.py -l en  -mode gcn -dp $dp -splits 1 -phase test -bs 64 -interm_layer 32 -output $op
