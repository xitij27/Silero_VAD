# Silero VAD evaluation

This repo runs the pretrained [Silero VAD](https://github.com/snakers4/silero-vad) on [AliMeeting](https://www.openslr.org/119/) (far-field) and [DIHARD III](https://catalog.ldc.upenn.edu/LDC2022S12). It writes the detected speech segments as RTTM files and scores them against reference RTTMs. There is no training code.

Part of the final year project [Robust Voice Activity Detection using DNN Approaches](https://github.com/xitij27/Robust-Voice-Activity-Detection-using-DNN-Approaches).

## Files

| File | Purpose |
|------|---------|
| `vad_silero_ali.py` | AliMeeting. Reads `*.wav` and loads Silero from `torch.hub` (`snakers4/silero-vad`) |
| `vad_silero_dh.py` | DIHARD III. Reads `*.flac` and loads the local `silero_vad.jit` |
| `run_vad_silero_ali.sh`, `run_vad_silero_dh.sh` | Set the data paths and call the matching Python script |
| `silero_vad.jit` | TorchScript Silero model, used only by `vad_silero_dh.py` |
| `utils_vad.py` | Copy of Silero's helper functions (`get_speech_timestamps`, `VADIterator`, …), used only by `vad_silero_dh.py` |

## Requirements

The repo has no requirements file. The scripts import `torch`, `torchaudio`, `librosa`, `soundfile`, `pydub`, `pyannote.core`, `scikit-learn`, `tqdm`, `numpy` and `IPython`. `IPython` is imported at the top of both scripts but never used, so it still has to be installed.

`vad_silero_ali.py` downloads Silero from GitHub through `torch.hub` on first use, so it needs internet access. That version isn't pinned, so AliMeeting results can differ from runs with the bundled `silero_vad.jit`.

## Setup

1. **Edit the run script** (`run_vad_silero_ali.sh` or `run_vad_silero_dh.sh`):
   - `source ~/myenv/bin/activate` activates a virtualenv. Change it to activate your own environment.
   - `input_wav`: the folder of audio. The AliMeeting script reads `.wav` files. The DIHARD script reads `.flac` files despite the variable name.
   - `input_rttm`: the folder of reference RTTMs, named `<audio basename>.rttm`.
   - `output_rttm`: where the predicted RTTMs are written. **Create this folder first**, because the script doesn't create it.
2. **DIHARD only:** `vad_silero_dh.py` loads the model from a hard-coded absolute path:
   ```python
   model = torch.jit.load("/home/users/ntu/kshitij0/FYP/vad/code_and_model/egs/silero_vad/silero_vad.jit")
   ```
   Change it to point at `silero_vad.jit` in this repo.

## Run

From the repository root:

```bash
bash run_vad_silero_ali.sh
bash run_vad_silero_dh.sh
```

Per-file results and a summary are printed to stdout. The scripts don't write a log file, because the Slurm logging line in each run script is commented out. The DIHARD script also prints the VAD threshold.

## How it works

1. Each recording is read in windows. Silero's `get_speech_timestamps` runs on each window with its default settings (threshold 0.5), and the segments it finds are shifted by the window's start time.
2. All segments for a file are written to `<output_rttm>/<name>.rttm`.
3. The reference RTTM and the predicted RTTM are both turned into one speech/non-speech value per audio sample, using the first channel at the file's own sample rate, and compared.

|                 | `vad_silero_ali.py`                          | `vad_silero_dh.py`        |
|-----------------|----------------------------------------------|---------------------------|
| Model source    | `torch.hub` (`snakers4/silero-vad`, latest)  | local `silero_vad.jit`    |
| Window / hop    | 60 s / 55 s                                  | 5 s / 5 s                 |
| Input files     | `*.wav`                                      | `*.flac`                  |

The window sizes are the `vad_chunk` and `vad_hop` defaults of the `silero_vad` class in each script. The reported processing time covers detection and RTTM writing, not scoring.

## How the metrics are computed

- **Accuracy:** the fraction of samples where prediction and reference agree.
- **False alarm:** samples predicted as speech that the reference marks as non-speech, as a fraction of *all* samples.
- **Missed detection:** samples predicted as non-speech that the reference marks as speech, as a fraction of *all* samples.
- **ROC-AUC:** `sklearn.metrics.roc_auc_score` on the binary decisions, not on speech probabilities.
- **Averaging:** each file is scored separately and the scores are averaged, so every file counts equally.
- **`RTF`:** printed as *total audio duration ÷ processing time*, which is how many times faster than real time the model runs. This is the inverse of the usual RTF definition. "Speech duration" in the output means total audio duration, assuming 16 kHz.

## Attribution

The model, `silero_vad.jit` and `utils_vad.py` come from [snakers4/silero-vad](https://github.com/snakers4/silero-vad).
