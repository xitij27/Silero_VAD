# Silero-VAD

We follow the official code of [Silero-VAD](https://github.com/snakers4/silero-vad) to evaluate the performance on [Alitmeeting](https://www.openslr.org/119/) and [Dihard-3](https://catalog.ldc.upenn.edu/LDC2022S12) datasets.

## Alimeeting dataset

Please open the script `run_vad_silero_ali.sh`, specify your conda environment name and following data paths:

- `input_wav`: directory that contains audio files in `.wav` format;
- `input_rttm`: directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `output_rttm`: output directory for generated `.rttm` files;

Then, run it with following command:
```shell
bash run_vad_silero_ali.sh
```

The evaluation results would be printed in the experiment log.

## Dihard-3 dataset

Please open the script `run_vad_silero_dh.sh`, specify your conda environment name and following data paths:

- `input_wav`: directory that contains audio files in `.flac` format;
- `input_rttm`: directory that contains time-stamp label files in `.rttm` format, with same file name as the corresponding audio files;
- `output_rttm`: output directory for generated `.rttm` files;

Then, run it with following command:
```shell
bash run_vad_silero_dh.sh
```

The evaluation results would be printed in the experiment log.


