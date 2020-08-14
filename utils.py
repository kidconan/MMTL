import librosa
from pydub import AudioSegment

import numpy as np
import torch
import math
from torch.autograd import Variable

import threading
from collections import OrderedDict

from read_conf import read_set_list_single, read_set_tuple
import pickle
import os
from copy import deepcopy


#////////////////////////////////////////////////////////////////////////////////// #
#                                                                                   #
#                   Data processing for audio and text                              #
#                                                                                   #
#////////////////////////////////////////////////////////////////////////////////// #

class MyThread(threading.Thread):
	"""docstring for MyThread"""

	def __init__(self, func, args=()):
		super(MyThread, self).__init__()
		self.func = func
		self.args = args

	def run(self):
		self.result = self.func(*self.args)

	def get_result(self):
		try:
			return self.result
		except Exception:
			return None


def segment_file(file, seg_dur, exp_path, file_type='wav'):
	wav = AudioSegment.from_wav(file)
	wav_name = (file.rsplit('/', maxsplit=1)[1]).split('.')[0]
	wav_len = wav.duration_seconds
	seg_list = []
	if wav_len <= seg_dur:
		new_wav_name = exp_path + '/' + wav_name + '_' + str(0) + '.' + file_type
		wav_seg = wav[:]
		wav_seg.export(new_wav_name, format=file_type)
		seg_list.append(new_wav_name)
	else:
		ceil_amount = math.ceil(wav_len/seg_dur)
		avg_len = wav_len/ceil_amount
		for i in range(ceil_amount):
			skip = i * avg_len * 1000
			skep = (i+1) * avg_len * 1000
			skep = min(skep, wav_len*1000)
			# print(skep-skip)
			wav_seg = wav[skip:skep]
			new_wav_name = exp_path + '/' + wav_name + '_' + str(i) + '.' + file_type
			wav_seg.export(new_wav_name, format=file_type)
			seg_list.append(new_wav_name)
	return seg_list


def get_librosa_feature(input_file, feature='fbank', dim=40, delta=False, delta_delta=False, window_size=25, stride=10,
						save_feature=None):
	y, sr = librosa.load(input_file, sr=None)
	ws = int(sr * 0.001 * window_size)
	st = int(sr * 0.001 * stride)
	if feature == 'fbank':  # log-scaled
		feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
											  n_fft=ws, hop_length=st)
		# feat = librosa.power_to_db(feat)
	elif feature == 'mfcc':
		feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26, n_fft=ws, hop_length=st)
		feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

	else:
		raise ValueError('Unsupported Acoustic Feature: ' + feature)

	feat = [feat]
	if delta:
		feat.append(librosa.feature.delta(feat[0]))

	if delta_delta:
		feat.append(librosa.feature.delta(feat[0], order=2))
	feat = np.concatenate(feat, axis=0)

	if save_feature is not None:
		tmp = np.swapaxes(feat, 0, 1).astype('float32')
		np.save(save_feature, tmp)
		return len(tmp)
	else:
		return np.swapaxes(feat, 0, 1).astype('float32')


def get_spectrogram_data(input_file, window_shift=10, winows_size=40, dft_length=1600, freq_min=0, freq_max=4000,
						 hz_resultion=10):

	data, rate = librosa.load(input_file, sr=None)
	window = int(rate*winows_size/1000)
	overlap = int(rate*window_shift/1000)
	nftt = dft_length
	spectrogram = librosa.stft(y=data, n_fft=nftt, hop_length=overlap, win_length=window, window='hamm')
	skip = int(freq_min/hz_resultion)
	skep = int(freq_max/hz_resultion)
	spectrogram = spectrogram[skip:skep, :]
	log_pow_spe = librosa.power_to_db((np.abs(spectrogram))**2)

	F, T = log_pow_spe.shape
	if T < 300:
		log_pow_spe = np.hstack((log_pow_spe, np.zeros((F, 300-T))))
	elif T > 300:
		log_pow_spe = log_pow_spe[:, :300]

	return log_pow_spe


def vad_project_fun(value, range_list, range_amount):
	pos = 0
	while pos < range_amount:
		lower, upper = range_list[pos]
		if lower == upper:
			if value == lower:
				return int(pos)
			else:
				pos += 1
		elif lower <= value < upper:
			return int(pos)
		else:
			pos += 1
	if pos == range_amount:
		print(pos, value)
		raise ValueError


#////////////////////////////////////////////////////////////////////////////////// #
#                                                                                   #
#                   Feature padding and target padding                              #
#                                                                                   #
#////////////////////////////////////////////////////////////////////////////////// #

def pack_exm_int(example_input):
	"""
	example_input: torch.cuda_tensor
	"""

	lengths = torch.sum(torch.sum(example_input, dim=-1) != 0, dim=-1)
	_, idx_sort = torch.sort(lengths, dim=0, descending=True)
	_, idx_unsort = torch.sort(idx_sort, dim=0)
	length = lengths[idx_sort]

	return example_input.index_select(0, idx_sort), length, idx_sort


def feature_padding(fea_list, ref_num=1, sold_length=0):
	# fea_list: a list of [time_len, audio_feature_dim]
	# ref_num : 2**encoder.pyramidal_layers
	if sold_length == 0:
		max_length = max([(wav.shape)[0] for wav in fea_list])
	else:
		max_length = sold_length
	# max_length = 3496
	if ref_num > 1:
		max_length = (max_length % ref_num) + max_length
	wav_amount = len(fea_list)
	fea_dim = (fea_list[0].shape)[1]
	fea_tensor = torch.zeros(wav_amount, max_length, fea_dim)
	for i in range(wav_amount):
		skip = (fea_list[i].shape)[0]
		fea_tensor[i, :skip, :] = torch.from_numpy(fea_list[i])
	return fea_tensor


def target_padding(tar_list, ref_num=1, sold_length=0):
	# tar_list: a list of transcption, which is list as well
	# ref_num : 2**encoder.pyramidal_layers
	if sold_length == 0:
		max_length = max(list(map(len, tar_list)))
	else:
		max_length = sold_length
	if ref_num > 1:
		max_length = (max_length % ref_num) + max_length
	wav_amount = len(tar_list)
	target_list = [[0 for j in range(max_length)] for i in range(wav_amount)]
	for i in range(wav_amount):
		skip = len(tar_list[i])
		target_list[i][:skip] = tar_list[i][:]

	return torch.tensor(target_list)


#////////////////////////////////////////////////////////////////////////////////// #
#                                                                                   #
#                                 Model Training                                    #
#                                                                                   #
#////////////////////////////////////////////////////////////////////////////////// #


def get_info_args(penlty, smooth, smooth_value, ckt_path, max_freq, cnn2rnn, rnn2dnn):
	penlty_bool = penlty
	if penlty_bool not in ['T', 'F']:
		print('the parse parameter of <penlty> shoudld be T or F')
		raise ValueError

	if smooth == 'T':
		name_smooth = 'smoothT'
		smoothing = smooth_value
		confidence = 1 - smoothing
	else:
		name_smooth = ''

	if penlty_bool == 'T':
		env_name = 'MTLReptileFCT1' + '-' + str(max_freq) + '-' + cnn2rnn + '-' + rnn2dnn + '-' + penlty_bool + '-' + \
				name_smooth
		ckp_path = ckt_path + '/' + 'MTLReptileFCT1' + '-' + str(max_freq) + '-' + cnn2rnn + '_' + rnn2dnn + '-' + \
				penlty_bool + '-' + name_smooth + '/'
		penlty_bool = True
	else:
		env_name = 'MTLReptileFCT1' + '-' + str(max_freq) + '-' + cnn2rnn + '-' + rnn2dnn + '-' + name_smooth
		ckp_path = ckt_path + '/' + 'MTLReptileFCT1' + '-' + str(max_freq) + '-' + cnn2rnn + '_' + rnn2dnn + '-' + \
				name_smooth + '/'
		penlty_bool = False
	return smoothing, confidence, name_smooth, env_name, ckp_path, penlty_bool


def get_crnn_paras(cfg, max_freq, cnn2rnn):
	cnn_filter = read_set_list_single(cfg, sec_name='cnn', sec_key='cnn_filter')
	cnn_inchan = [1] + (cnn_filter[:-1])
	cnn_outchan = cnn_filter[:]
	cnn_kernel = read_set_tuple(cfg, sec_name='cnn', sec_key='cnn_kernel')
	cnn_stride = read_set_tuple(cfg, sec_name='cnn', sec_key='cnn_stride')
	cnn_padding = read_set_tuple(cfg, sec_name='cnn', sec_key='cnn_padding')
	cnn_setting = list(zip(cnn_inchan, cnn_outchan, cnn_kernel, cnn_stride, cnn_padding))

	pooling_kernel = read_set_tuple(cfg, sec_name='pooling', sec_key='pooling_kernel')
	pooling_stride = read_set_tuple(cfg, sec_name='pooling', sec_key='pooling_stride')
	pooling_setting = list(zip(pooling_kernel, pooling_stride))

	pool_amount = len(pooling_stride)
	base = max_freq
	for i in range(pool_amount):
		base = math.floor((base - pooling_kernel[i][0]) / pooling_stride[i][0] + 1)
	if cnn2rnn == 'concat':
		rnn_in_dim = cnn_filter[-1] * base
	elif (cnn2rnn == 'sum') or (cnn2rnn == 'avg') or (cnn2rnn == 'max'):
		rnn_in_dim = base
	else:
		raise ValueError

	rnn_layer = int(cfg.get('rnn', 'rnn_layers'))
	rnn_hid = int(cfg.get('rnn', 'rnn_hid'))
	rnn_bi = cfg.get('rnn', 'rnn_bi')
	if rnn_bi:
		rnn_bi = True
	else:
		rnn_bi = False

	rnn_setting = [(rnn_in_dim, rnn_hid, rnn_layer, rnn_bi)]
	dnn_setting = int(cfg.get('dnn', 'dnn_hid1'))
	drop_rate = float(cfg.get('dropout', 'p'))

	return cnn_setting, rnn_setting, pooling_setting, dnn_setting, drop_rate


def get_corrsponding_task(str_num):
	task_index = []
	for char in str_num:
		if char in '123':
			task_index.append(int(char))
		else:
			print("the element of str_num should be 1 or 2 or 3")
			raise ValueError
	return task_index


def get_WA(confusion_mat):
	# which is the classification accuarcy of all utterances
	return confusion_mat.diagonal().sum() / confusion_mat.sum()


def get_UA(confusion_mat):
	# which average the accuarcy of each individual emotion class
	return (confusion_mat.diagonal() / confusion_mat.sum(axis=0)).mean()


def get_device_idx(net):
	return next(net.parameters()).get_device()


def point_to_grad(main_net, auxi_net, cuda_bool, deviec_idx, base=1):
	for p, target_p in zip(main_net.parameters(), auxi_net.parameters()):
		if p.grad is None:
			if cuda_bool:
				p.grad = Variable(torch.zeros(p.size())).cuda(deviec_idx)
			else:
				p.grad = Variable(torch.zeros(p.size()))
		if target_p.grad is None:
			if cuda_bool:
				target_p.grad = Variable(torch.zeros(target_p.size())).cuda(deviec_idx)
			else:
				target_p.grad = Variable(torch.zeros(target_p.size()))
		else:
			p.grad.data.add_(deepcopy(target_p.grad.data) / base)


def average_grad(main_net, task_num):
	for p in main_net.parameters():
		p.grad.data.div_(task_num)


def model_zero_grad(net, cuda_bool, deviec_idx):
	for p in net.parameters():
		if cuda_bool:
			p.grad = Variable(torch.zeros(p.size())).cuda(deviec_idx)
		else:
			p.grad = Variable(torch.zeros(p.size()))


#////////////////////////////////////////////////////////////////////////////////// #
#                                                                                   #
#                                 MAML setting                                      #
#                                                                                   #
#////////////////////////////////////////////////////////////////////////////////// #


def update_dict(dict1, dict2, way='add'):
	if set(dict1.keys()) != set(dict2.keys()):
		print('the key of dict1 and dict2 are different')
		raise ValueError
	else:
		new_dict = OrderedDict()
		new_dict.update(dict1)
		for key in dict2.keys():
			if way == 'add':
				new_dict[key] += dict2[key]
			else:
				new_dict[key] = list(np.array(new_dict[key]) + np.array(dict2[key]))

		new_dict = OrderedDict(sorted(new_dict.items(), key=lambda obj: obj[0]))

		return new_dict


def construct_model(cfg, max_freq, out_dim, in_chan=1, cnn2rnn='', rnn2dnn=''):

	cnn_filter = read_set_list_single(cfg, sec_name='cnn', sec_key='cnn_filter')
	cnn_inchan = [in_chan] + (cnn_filter[:-1])
	cnn_outchan = cnn_filter[:]
	cnn_kernel = read_set_tuple(cfg, sec_name='cnn', sec_key='cnn_kernel')
	cnn_stride = read_set_tuple(cfg, sec_name='cnn', sec_key='cnn_stride')
	cnn_padding = read_set_tuple(cfg, sec_name='cnn', sec_key='cnn_padding')
	cnn_setting = list(zip(cnn_inchan, cnn_outchan, cnn_kernel, cnn_stride, cnn_padding))

	pooling_kernel = read_set_tuple(cfg, sec_name='pooling', sec_key='pooling_kernel')
	pooling_stride = read_set_tuple(cfg, sec_name='pooling', sec_key='pooling_stride')
	pooling_setting = list(zip(pooling_kernel, pooling_stride))

	pool_amount = len(pooling_stride)
	base = max_freq
	for i in range(pool_amount):
		base = math.floor((base - pooling_kernel[i][0]) / pooling_stride[i][0] + 1)
	if cnn2rnn == 'concat':
		rnn_in_dim = cnn_filter[-1] * base
	elif (cnn2rnn == 'sum') or (cnn2rnn == 'avg') or (cnn2rnn == 'max'):
		rnn_in_dim = base
	else:
		raise ValueError

	rnn_layer = int(cfg.get('rnn', 'rnn_layers'))
	rnn_hid = int(cfg.get('rnn', 'rnn_hid'))
	rnn_bi = cfg.get('rnn', 'rnn_bi')
	if rnn_bi == 'T':
		rnn_bi = True
	else:
		rnn_bi = False

	# in_dim, hid_dim, layers, bool_bi = rnn_setting[i]
	rnn_setting = [(rnn_in_dim, rnn_hid, rnn_layer, rnn_bi)]

	tmp = int(cfg.get('dnn', 'dnn_hid1'))
	base_dim = rnn_setting[-1][1] * (1+rnn_setting[-1][-1])
	dnn_setting = [(base_dim, tmp), (tmp, out_dim)]
	drop_rate = float(cfg.get('dropout', 'p'))

	transf_setting = (out_dim, out_dim)

	config = []
	# for i in range(len(cnn_setting)):
	config.append(('conv2d', cnn_setting[0]))
	config.append(('relu', ''))
	config.append(('max_pool2d', pooling_setting[0]))

	config.append(('conv2d', cnn_setting[1]))
	config.append(('relu', ''))
	config.append(('max_pool2d', pooling_setting[1]))

	config.append(('conv2d', cnn_setting[2]))
	config.append(('relu', ''))
	config.append(('max_pool2d', pooling_setting[2]))

	config.append(('cnn2rnn', cnn2rnn))
	config.append(('permute', [0, 2, 1]))
	config.append(('LSTM', rnn_setting[0]))
	config.append(('rnn2dnn', rnn2dnn))
	config.append(('FC', dnn_setting[0]))
	config.append(('relu', ''))
	config.append(('dropout', drop_rate))
	config.append(('FC', dnn_setting[1]))

	support_config = deepcopy(config)
	query_config = deepcopy(config)

	support_config.append(('softmax', ''))

	query_config.append(('FC', transf_setting))
	query_config.append(('softmax', ''))

	return support_config, query_config


def construct_meta_info(cfg, extra_config):

	update_lr = float(cfg.get('meta_info', 'update_lr'))
	meta_lr = float(cfg.get('meta_info', 'meta_lr'))
	# used in inner loop in program, so not saved in config
	meta_test_lr = float(cfg.get('meta_info', 'meta_test_lr'))

	update_step = int(cfg.get('meta_info', 'update_step'))
	# used in inner loop in program, so not saved in config
	test_update_step = int(cfg.get('meta_info', 'test_update_step'))

	config = [update_lr, meta_lr, update_step]
	config.extend(extra_config)
	config_name = ['update_lr', 'meta_lr', 'update_step', 'task_num']

	meta_config = []
	for name, value in zip(config_name, config):
		meta_config.append((name, value))

	return meta_config, meta_test_lr, test_update_step


def gather_lab(base_path='/home/gkb/asr_model/IEMOCAP/meta_length800_db'):
	for i in range(5):
		data_mat = []
		data_emo = []
		data_v = []
		data_a = []
		data_d = []

		cir_path = base_path + '/' + 'leave_' + str(i + 1)
		for j in range(6):
			with open(cir_path + '/' + 'train_' + str(j + 1) + '_mat.pkl', 'rb') as f:
				data = pickle.load(f)
				data_mat.extend(data)

			with open(cir_path + '/' + 'train_' + str(j + 1) + '_emo.pkl', 'rb') as f:
				data = pickle.load(f)
				data_emo.extend(data)

			with open(cir_path + '/' + 'train_' + str(j + 1) + '_v.pkl', 'rb') as f:
				data = pickle.load(f)
				data_v.extend(data)

			with open(cir_path + '/' + 'train_' + str(j + 1) + '_a.pkl', 'rb') as f:
				data = pickle.load(f)
				data_a.extend(data)

			with open(cir_path + '/' + 'train_' + str(j + 1) + '_d.pkl', 'rb') as f:
				data = pickle.load(f)
				data_d.extend(data)

		save_path = cir_path + '/' + 'reptile'
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		# Restore the feature mat
		with open(save_path + '/' + 'meta_test_mat.pkl', 'wb') as f:
			pickle.dump(data_mat, f)

		# Restore the emo mat
		with open(save_path + '/' + 'meta_test_emo.pkl', 'wb') as f:
			pickle.dump(data_emo, f)

		# Restore the v mat
		with open(save_path + '/' + 'meta_test_v.pkl', 'wb') as f:
			pickle.dump(data_v)

		# Restore the a mat
		with open(save_path + '/' + 'meta_test_a.pkl', 'wb') as f:
			pickle.dump(data_a)

		# Restore the d mat
		with open(save_path + '/' + 'meta_test_d.pkl', 'wb') as f:
			pickle.dump(data_d)

		# Restore the penlty dictionary of emo
		with open(cir_path + '/' + 'train_emo_portion.pkl', 'rb') as f:
			temp = pickle.load(f)

		with open(save_path + '/' + 'meta_test_emo_portion.pkl', 'wb') as f:
			pickle.dump(temp, f)

		# Restore the penlty dictionary of v
		with open(cir_path + '/' + 'train_v_portion.pkl', 'rb') as f:
			temp = pickle.load(f)

		with open(save_path + '/' + 'meta_test_v_portion.pkl', 'wb') as f:
			pickle.dump(temp, f)

		# Restore the penlty dictionary of a
		with open(cir_path + '/' + 'train_a_portion.pkl', 'rb') as f:
			temp = pickle.load(f)

		with open(save_path + '/' + 'meta_test_a_portion.pkl', 'wb') as f:
			pickle.dump(temp, f)

		# Restore the penlty dictionary of d
		with open(cir_path + '/' + 'train_d_portion.pkl', 'rb') as f:
			temp = pickle.load(f)

		with open(save_path + '/' + 'meta_test_d_portion.pkl', 'wb') as f:
			pickle.dump(temp, f)