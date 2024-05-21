#!/usr/bin/env bash

#cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node06"

source ~/myenv/bin/activate

input_wav=/home4/huyuchen/raw_data/Alimeeting/Test_Ali_far/audio_dir
input_rttm=/home4/huyuchen/raw_data/Alimeeting/Test_Ali_far/rttm_groundtruth
output_rttm=/home3/huyuchen/pytorch_workplace/vad/egs/silero_vad/output_wav/ali

#$cmd ./log/run_vad_silero_ali.log \
python vad_silero_ali.py --input_wav ${input_wav} --input_rttm ${input_rttm} --output_rttm ${output_rttm}

