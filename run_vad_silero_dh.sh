#!/usr/bin/env bash

#cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node06"

source ~/myenv/bin/activate

input_wav=/home4/huyuchen/raw_data/third_dihard_challenge_eval/data/flac
input_rttm=/home4/huyuchen/raw_data/third_dihard_challenge_eval/data/rttm
output_rttm=FYP/vad/code_and_model/egs/silero_vad/output_wav/dh

#$cmd ./log/run_vad_silero_dh.log \
python vad_silero_dh.py --input_wav ${input_wav} --input_rttm ${input_rttm} --output_rttm ${output_rttm}

