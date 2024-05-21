from typing import Dict, List, Union
import numpy as np
from torch import hub
from pydub import AudioSegment
from pyannote.core import Annotation, Segment, Timeline
import soundfile as sf
import os
import sys
import torch
import librosa
import torchaudio
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)
from IPython.display import Audio
from pprint import pprint
import soundfile as sf
import numpy as np
import time

# proj_dir = os.path.join(os.getcwd().split("OneDiarization")[0], "OneDiarization")
# sys.path.append(proj_dir)

# from utils.feature import get_slices, to_array

def get_slices(num_samps: int, winlen: int = 1536, return_last: bool = True, match_samps: bool = True):
	"""
    Parameters:
    - num_samps: an integer representing the total number of samples in the signal to be sliced.
    - winlen: an integer representing the length of each slice. The default value is 1536.
    - return_last: a boolean flag indicating whether or not to return the last slice if it is shorter than the specified window length. The default value is True.
    - match_samps: a boolean flag indicating whether or not to return a slice that matches the exact window length or shorter for the last slice. The default value is True.
    Returns:
    - A list of slice indices, where each slice has the same length as specified by the window length parameter, except
    for the last slice, which may be shorter if there are not enough samples remaining to make a full-length slice.
    """

	slice_indices = []
	num_slices = num_samps // winlen

	for i in range(num_slices):
		slice_indices.append(slice(i * winlen, (i + 1) * winlen))

	if num_slices * winlen < num_samps:
		if return_last:
			if match_samps:
				slice_indices.append(slice(num_slices * winlen, num_samps))
			else:
				slice_indices.append(slice(num_slices * winlen, min((num_slices + 1) * winlen, num_samps)))
	return slice_indices


def to_array(window: AudioSegment, to_torch: bool = True):
	"""
    Parameters:
    - window: an instance of AudioSegment representing an audio signal window.
    - to_torch: a boolean flag indicating whether or not to convert the output array to a PyTorch tensor. The default value
    is True.
    Returns:
    - A NumPy array or PyTorch tensor representing the audio signal window. The output array or tensor will have a shape of
    (num_channels, num_samples), where num_channels represents the number of audio channels in the signal and num_samples
    represents the number of samples in the signal.
    """

	# Convert AudioSegment to NumPy array
	samples = np.array(window.get_array_of_samples())

	# Reshape NumPy array to (num_channels, num_samples)
	num_channels = window.channels
	num_samples = len(samples) // num_channels
	samples = samples.reshape(num_channels, num_samples)

	if to_torch:
		# Convert NumPy array to PyTorch tensor
		tensor = torch.from_numpy(samples.astype(np.float32))
		return tensor
	else:
		return samples


class silero_vad():

	def __init__(self,
				 winlen: float = 0.0976875,
				 msecs: int = 1000,
				 repo_or_dir: str = 'snakers4/silero-vad',
				 model_name: str = 'silero_vad',
				 force_reload: bool = False,
				 return_last: bool = True,
				 match_samps: bool = True,
				 start_key: str = 'start',
				 stop_key: str = 'end',
				 to_torch: bool = True,
				 return_seconds: bool = True,
				 verbose: bool = True,
				 vad_chunk: int = 60,
				 vad_hop: int = 55,
				 sampling_rate: int = 16000, ):

		'''
		Inputs:
			winlen: Window length of VAD (in seconds, 1536 samples for 16kHz)
			msecs: Conversion to or from milliseconds

		Output:
			Generator of slices of speech frames

		'''

		# Initialise parameters and model
		self.winlen = winlen
		self.repo_or_dir = repo_or_dir
		self.model_name = model_name
		self.force_reload = force_reload
		self.msecs = msecs
		self.return_last = return_last
		self.match_samps = match_samps
		self.start_key = start_key
		self.stop_key = stop_key
		self.to_torch = to_torch
		self.return_seconds = return_seconds
		self.verbose = verbose

		self.vad_chunk = vad_chunk
		self.vad_hop = vad_hop
		self.sampling_rate = sampling_rate

		self.load_model()

	def load_model(self):

		model, utils = hub.load(repo_or_dir=self.repo_or_dir,
								model=self.model_name,
								force_reload=self.force_reload,
								verbose=self.verbose)

		(get_speech_timestamps,
		 save_audio,
		 read_audio,
		 VADIterator,
		 collect_chunks) = utils

		self.vad = VADIterator(model)
		self.model = model
		self.get_speech_timestamps = get_speech_timestamps
		self.read_audio = read_audio

	def get_voice_frames_offline(self, file_path, sr=16000):
		wav = self.read_audio(file_path, sampling_rate=sr)
		return self.get_speech_timestamps(wav, self.model, sampling_rate=sr, return_seconds=True)

	def get_voice_segments(self, audio, sr=16000):
		if not isinstance(audio, torch.Tensor):
			audio = torch.Tensor(audio)

		segments = []
		for _slice in self.get_speech_timestamps(audio, self.model, sampling_rate=sr, return_seconds=True):
			segments.append(Segment(_slice['start'], _slice['end']))

		return segments

	def inference_single_file(self, audio_path, out_rttm_path, ignore_existing_rttm=True):
		hyp_annot_filename = os.path.join(out_rttm_path, os.path.basename(audio_path).split(".")[-2] + ".rttm")
		if not ignore_existing_rttm and os.path.exists(hyp_annot_filename):
			print("RTTM file detected: {}".format(hyp_annot_filename))
			return

		chunk_size = self.vad_chunk
		hop_size = self.vad_hop

		duration_seconds = librosa.get_duration(filename=audio_path)
		sampling_rate = librosa.get_samplerate(audio_path)

		hyp_annot = Annotation()
		for time_st in tqdm(np.arange(0, duration_seconds, hop_size)):
			try:
				audio_chunk, _ = torchaudio.load(audio_path, frame_offset=int(sampling_rate * time_st),
												 num_frames=int(sampling_rate * chunk_size))
			except:
				print("Cannot read file: {}".format(audio_path))
				return

			audio_chunk = audio_chunk[0]
			if sampling_rate != self.sampling_rate:
				audio_chunk = torchaudio.functional.resample(audio_chunk, orig_freq=sampling_rate,
															 new_freq=self.sampling_rate)

			segments = self.get_voice_segments(audio_chunk)
			for segment in segments:
				hyp_annot[Segment(segment.start + time_st, segment.end + time_st)] = 0

		with open(hyp_annot_filename, "w") as f_rttm:
			hyp_annot.write_rttm(f_rttm)

	def get_windows(self, audio: AudioSegment):

		num_samps = int(audio.duration_seconds * self.msecs)
		winlen = int(self.winlen * self.msecs)

		# Return slices (indices) of audio
		slices = get_slices(num_samps=num_samps,
							winlen=winlen,
							return_last=self.return_last,
							match_samps=self.match_samps)

		return slices

	def is_speech_frame(self, window: AudioSegment):

		array = to_array(window, to_torch=self.to_torch)
		speech_dict = self.vad(array, return_seconds=self.return_seconds)

		if speech_dict:
			if self.start_key in speech_dict.keys():
				self.start = speech_dict[self.start_key]
				self.start *= self.msecs
				return [int(self.start), None]
			else:
				self.stop = speech_dict[self.stop_key]
				self.stop *= self.msecs
				return [None, int(self.stop)]

	# max_vad_len_s is used to split a long segment into short segments
	def get_voice_frames(self, audio: AudioSegment, max_vad_len_s=-1):

		start, stop = None, None
		for _slice in self.get_windows(audio):
			# For each slice/window of audio
			window = audio[_slice]
			ret_slice = self.is_speech_frame(window)
			if ret_slice:
				if ret_slice[0] is not None:
					start = ret_slice[0]
				if ret_slice[1] is not None:
					stop = ret_slice[1]
				if start and stop:
					yield slice(*[int(start), int(stop)])
					start, stop = None, None
			elif max_vad_len_s > 0:
				if start and (_slice.stop - start) > max_vad_len_s * self.msecs:
					yield slice(*[int(start), int(_slice.stop)])
					start = _slice.stop

		# If a vad segment has started, it should last until the end
		if start:
			yield slice(*[int(start), _slice.stop])

		self.vad.reset_states()


def inference_vad(audio_file_path: str, output_rttm_file: str, audio_save_path: str = '', min_vad_ms: int = 300,
				  collar_ms: int = 350):
	audio = AudioSegment.from_file(audio_file_path)
	vad = silero_vad(return_last=False)

	if audio_save_path == '':
		audio_to_save = None
	else:
		audio_to_save = AudioSegment.empty()

	rttm_annotation = Annotation()
	last_end_time = 0
	for _slice in vad.get_voice_frames(audio):
		start_s = _slice.start / 1000
		end_s = _slice.stop / 1000
		if audio_to_save is not None:
			if last_end_time < _slice.start:
				audio_to_save += AudioSegment.silent(_slice.start - last_end_time, frame_rate=audio.frame_rate)
			audio_to_save += audio[_slice]
			last_end_time = _slice.stop

		rttm_annotation[Segment(start_s, end_s)] = 0

	if audio_to_save is not None:
		num_sample = len(audio.get_array_of_samples())
		audio_len_ms = int(num_sample / audio.frame_rate * 1000)
		if audio_len_ms > last_end_time:
			audio_to_save += AudioSegment.silent(audio_len_ms - last_end_time, frame_rate=audio.frame_rate)

	rttm_annotation = rttm_annotation.support(collar=float(collar_ms / 1000.0))

	new_annot = Annotation()
	for segment, track, label in rttm_annotation.itertracks(yield_label=True):
		if segment.duration >= float(min_vad_ms / 1000.0):
			new_annot[segment] = label

	with open(output_rttm_file, 'w') as f_rttm:
		new_annot.write_rttm(f_rttm)

	if audio_to_save is not None:
		audio_to_save.export(audio_save_path, format='wav')


def main():
	import time
	start_time = time.time()

	###############################################################

	audiofile = 'wav/mix_0000001.wav'
	out_wavfile = 'wav/mix_0000001_silero.wav'
	audio = AudioSegment.from_file(audiofile)
	vad = silero_vad(return_last=False)

	print('Duration BEFORE vad: {}s'.format(audio.duration_seconds))

	combined = AudioSegment.empty()
	for _slice in vad.get_voice_frames(audio):
		combined += audio[_slice]

	###############################################################

	run_time = time.time() - start_time
	print('Duration AFTER vad: {}s'.format(combined.duration_seconds))
	print('Took {:.2f}s to do vad'.format(run_time))

	combined.export(out_wavfile, format='wav')
	print('Trimmed audio saved to: {}'.format(out_wavfile))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_wav',
		type=str,
	)
	parser.add_argument(
		'--input_rttm',
		type=str,
	)
	parser.add_argument(
		'--output_rttm',
		type=str,
	)
	args = parser.parse_args()

	total_acc, total_fa, total_missing, total_roc_auc, num_samples = 0, 0, 0, 0, 0
	vad = silero_vad(return_last=False)
	total_t1, total_t2 = 0, 0
	for source in glob(os.path.join(args.input_wav, "*.wav")):
		t0 = time.time()
		vad.inference_single_file(source, args.output_rttm)
		t = time.time() - t0
		total_t1 += t

		### 1. wav/flac format
		audio, sr = torchaudio.load(source)

		### 2. flac format
		# audio, sr = sf.read(source)
		# audio = torch.from_numpy(audio).float()

		if len(audio.shape) > 1:
			audio = audio[0, :]

		utt_id = os.path.basename(source).split('.')[0]
		gt_mask, pred_mask = torch.zeros_like(audio), torch.zeros_like(audio)
		total_t2 += gt_mask.shape[-1]/16000

		f_gt = open(f'{args.input_rttm}/{utt_id}.rttm', 'r')
		for line in f_gt.readlines():
			tokens = line.strip().split()
			start, end = int(float(tokens[3]) * sr), int((float(tokens[3]) + float(tokens[4])) * sr)
			if end > gt_mask.shape[0]:
				end = gt_mask.shape[0]
			gt_mask[start - 1: end] = 1
		f_gt.close()

		f_pred = open(f'{args.output_rttm}/{utt_id}.rttm', 'r')
		for line in f_pred.readlines():
			tokens = line.strip().split()
			start, end = int(float(tokens[3]) * sr), int((float(tokens[3]) + float(tokens[4])) * sr)
			if end > pred_mask.shape[0]:
				end = pred_mask.shape[0]
			pred_mask[start - 1: end] = 1
		f_pred.close()

		acc = (pred_mask == gt_mask).sum() / gt_mask.shape[0]
		fa = ((pred_mask == 1) & (gt_mask == 0)).sum() / gt_mask.shape[0]
		missing = ((pred_mask == 0) & (gt_mask == 1)).sum() / gt_mask.shape[0]
		roc_auc = roc_auc_score(gt_mask.numpy(), pred_mask.numpy())
		print(f'{utt_id}: Acc = {acc:.3f}, Missing detection = {missing:.3f}, False alarm = {fa:.3f}, ROC-AUC score = {roc_auc:.3f}')
		print(f'processing time: {t:.3f}, speech duration: {gt_mask.shape[-1] / 16000:.3f}')

		total_acc += acc
		total_fa += fa
		total_missing += missing
		total_roc_auc += roc_auc
		num_samples += 1

	total_acc /= num_samples
	total_fa /= num_samples
	total_missing /= num_samples
	total_roc_auc /= num_samples
	print('======================== Results ========================')
	print(f'Acc = {total_acc:.3f}, Missing detection = {total_missing:.3f}, False alarm = {total_fa:.3f}, ROC-AUC score = {total_roc_auc:.3f}')
	print(f'Processing time = {total_t1:.3f}, speech duration = {total_t2:.3f}, RTF = {total_t2/total_t1:.3f}')

