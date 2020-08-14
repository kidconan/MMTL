import os
import codecs
import numpy as np
import random
import math
import pickle
import argparse
from collections import OrderedDict
from utils import MyThread, segment_file, get_spectrogram_data, get_librosa_feature, vad_project_fun, update_dict
from read_conf import read_spectrogram_conf
parser = argparse.ArgumentParser()
parser.add_argument('--exppath', type=str, help='indicate the path to svae the segment the audio',
					default='/home/gkb/asr_model/IEMOCAP/segment_files2/')
parser.add_argument('--savdir', type=str, help='indicate the file name to save the lab_mat',
					default='/home/gkb/asr_model/IEMOCAP/meta_multi_task_db2/')
parser.add_argument('--conf', type=str, help='indicate the feature conf to use',
					default='/home/gkb/asr_model/IEMOCAP//config/spectrogram.conf')
parser.add_argument('--spect', type=str, default='2', help='1: DFT of 1600/10 hz resolution; '
														   '2: DFT of 800/20 hz resolution ')
parser.add_argument('--vad', type=str, default='T', help='if True, valence, arouse and domain are used for the model')

paras = parser.parse_args()

exp_path = paras.exppath
sav_dir = paras.savdir
conf = paras.conf
spect_type = 'spectrogram' + paras.spect
vad = paras.vad
if vad == 'T':
	vad = True
elif vad == 'F':
	vad = False
else:
	print('vad should be T or F')
	raise ValueError

print(spect_type)
random.seed(1234)

data_path_prefix = '/home/gkb/asr_model/IEMOCAP/'
lab_path_suffix = '/dialog/EmoEvaluation/'
wav_path_suffix = '/sentences/wav/'

lab_m_list = ['neu', 'ang', 'hap', 'sad']
lab_2_index = {lab_m_list[i]: i for i in range(len(lab_m_list[:4]))}

def txt_filter(file, other_rule=['script']):
	'''
	define the rule of extracting the needed audio
	:param file:
	:param other_rule:
	:return:
	'''
	if '.txt' in file:
		for rule in other_rule:
			if rule in file:
				return None
		return file
	else:
		return None

def get_vad(vad_string, proj_range=[(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)]):
	'''
	project the valence, arousal, dominance value to discrete value
	:param vad_string:
	:param proj_range:
	:return:
	'''
	valence = (vad_string[0]).replace('[', ''); valence = valence.replace(',', ''); valence = float(valence)
	arouse = (vad_string[1]).replace(',', ''); arouse = float(arouse)
	domain = (vad_string[2]).replace(']', ''); domain = domain.replace(',', ''); domain = float(domain)
	valence = vad_project_fun(valence, proj_range, 3)
	arouse = vad_project_fun(arouse, proj_range, 3)
	domain = vad_project_fun(domain, proj_range, 3)
	return [valence, arouse, domain]

def get_emo_data(session_name, bool_seg=False, bool_vad=False, feature_extra='spectrogram', seg_dur=0, exp_path='',
				 conf=None, cfg_section=None):

	'''
	:param session_name:
	:param bool_seg: bool, if segment the audio file
	:param feature_extra: str, the type of feature, the value of which  should be 'spectrogram' or 'mel_filter'
	:param seg_dur: int, represents the length of each segment in second, and it works only when the bool_seg=True
	:param exp_path: str, the path to save the segment of the audio, and it works only when the bool_seg=True

	:return: data_dict: dict, the key is session_name + '_' + gender, the value is [feature mat and emotion kind]
	'''

	window_shift, windows_size, dft_length, freq_min, freq_max, hz_resultion = read_spectrogram_conf(conf, cfg_section)
	dir_path = data_path_prefix + session_name + lab_path_suffix
	file_list = os.listdir(dir_path)
	file_list = list(filter(txt_filter, file_list))
	data_dict = dict()
	for file in file_list:
		mask = False
		with codecs.open(dir_path + file, 'r') as f:
			for line in f.readlines():
				data = line.strip()
				if data == '':
					mask = True
				else:
					if mask:
						info_data = data.split()
						# e.g  info_data = [5.2300 - 8.7100]	Ses02M_script01_1_F000	xxx	[3.0000, 3.0000, 3.0000]
						if bool_vad:
							vad = info_data[5:]
							vad = get_vad(vad)
						emotion = info_data[4]
						if emotion in lab_m_list:
							file_name = info_data[3]
							# e.g  file_name = Ses02M_script01_1_F000
							dir_name, speaker = file_name.rsplit('_', maxsplit=1)
							speaker = speaker[0]
							session_speaker = session_name + '_' + speaker
							if data_dict.get(session_speaker) is None:
								data_dict[session_speaker] = dict()
							# if emotion == 'exc':
							# 	emotion = 'hap'
							wav_path = data_path_prefix + session_name + wav_path_suffix + dir_name + '/' + file_name\
									   + '.wav'
							if bool_seg:
								new_wav_list = segment_file(wav_path, seg_dur, exp_path, 'wav')
							else:
								new_wav_list = []
								new_wav_list.append(wav_path)
							# wav_mat = get_librosa_feature(wav_path)
							wav_info = []
							for wav in new_wav_list:
								if feature_extra == 'spectrogram':
									wav_mat = get_spectrogram_data(input_file=wav, window_shift=window_shift,
																   winows_size=windows_size,  dft_length=dft_length,
																   freq_min=freq_min, freq_max=freq_max,
																   hz_resultion=hz_resultion)
								elif feature_extra == 'mel_filter':
									wav_mat = get_librosa_feature(wav)
								else:
									print('The value of feature_extra should be %s or %s' % ('spectrogram',
																							'mel_filter')
										  )
									raise ValueError
								wav_info.append([wav_mat, emotion])
								if bool_vad:
									wav_info[-1].extend(vad)
							data_dict[session_speaker][file_name] = wav_info
						mask = False
					else:
						continue
	return data_dict

def stort_mat_lab(X, Y, info_idx, block_num, avg_amount, total_amount, file_path=None, info=None, bool_vad=False):
	# used for store training data
	for i in range(block_num):
		skip = i * avg_amount
		skep = min((i + 1) * avg_amount, total_amount)
		idx_list = info_idx[skip:skep]
		data_mat = [ X[idx] for idx in idx_list ]
		if bool_vad:
			data_emo = []
			data_valence = []
			data_arouse = []
			data_domain = []
			for idx in idx_list:
				emo, valence, arouse, domain = Y[idx]
				data_emo.append(lab_2_index.get(emo))
				data_valence.append(valence)
				data_arouse.append(arouse)
				data_domain.append(domain)
			with open(file_path + '/' + info + '_' + str(i + 1) + '_emo.pkl', 'wb') as f:
				pickle.dump(data_emo, f)
			with open(file_path + '/' + info + '_' + str(i + 1) + '_v.pkl', 'wb') as f:
				pickle.dump(data_valence, f)
			with open(file_path + '/' + info + '_' + str(i + 1) + '_a.pkl', 'wb') as f:
				pickle.dump(data_arouse, f)
			with open(file_path + '/' + info + '_' + str(i + 1) + '_d.pkl', 'wb') as f:
				pickle.dump(data_domain, f)
		else:
			data_lab = [ lab_2_index.get(Y[idx]) for idx in idx_list ]
			# data = [ [ X[idx], Y[idx] ] for idx in idx_list ]
			with open(file_path + '/' + info + '_' + str(i + 1) + '_lab.pkl', 'wb') as f:
				pickle.dump(data_lab, f)
		with open(file_path + '/' + info + '_' + str(i + 1) + '_mat.pkl', 'wb') as f:
			pickle.dump(data_mat, f)

def store(file_path, train_key, test_key, data_dict, bias=0, bool_vad=False):

	total_E_set = {i: 0 for i in range(4)}
	total_V_set = {i: 0 for i in range(4)}
	total_A_set = {i: 0 for i in range(4)}
	total_D_set = {i: 0 for i in range(4)}
	train_amount = 0
	for session_speaker in train_key:
		file_mat_lab_dict = data_dict.get(session_speaker)
		for wav_file in file_mat_lab_dict:
			mat_lab_list = file_mat_lab_dict.get(wav_file)
			if bool_vad:
				for _, label, valence, arouse, domain in mat_lab_list:
					emo = lab_2_index.get(label)
					total_E_set[emo] += 1
					total_V_set[valence] += 1
					total_A_set[arouse] += 1
					total_D_set[domain] += 1
					train_amount += 1
			else:
				print("Nothing")
	total_E_set = {i: train_amount / total_E_set[i] for i in range(4)}
	total_V_set = {i: train_amount / total_V_set[i] for i in range(4)}
	total_A_set = {i: train_amount / total_A_set[i] for i in range(4)}
	total_D_set = {i: train_amount / total_D_set[i] for i in range(4)}

	# =====================================Construct meta-train dataset=====================================
	k = 1
	for session_speaker in train_key:
		train_mat = []
		train_Y = []
		file_mat_lab_dict = data_dict.get(session_speaker)

		for wav_file in file_mat_lab_dict:
			mat_lab_list = file_mat_lab_dict.get(wav_file)
			if bool_vad:
				for mat, label, valence, arouse, domain in mat_lab_list:
					mat = mat - bias
					mat = (mat - mat.mean())/mat.std()
					train_mat.append(mat)
					emo = lab_2_index.get(label)
					train_Y.append([emo, valence, arouse, domain])
			else:
				for mat, label in mat_lab_list:
					mat = mat - bias
					mat = (mat - mat.mean()) / mat.std()
					train_mat.append(mat)
					train_Y.append(label)
		train_amount = len(train_Y)
		train_idx = list(range(train_amount))
		random.shuffle(train_idx)

		features = []
		labels = []
		for idx in train_idx:
			features.append(train_mat[idx])
			labels.append(train_Y[idx])

		file_name = file_path + '/' + 'train_task' + str(k) + '_mat.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(features, f)
		file_name = file_path + '/' + 'train_task' + str(k) + '_lab.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(labels, f)

		file_name = file_path + '/' + 'train_task' + str(k) + '_Vport.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(total_V_set, f)

		file_name = file_path + '/' + 'train_task' + str(k) + '_Aport.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(total_A_set, f)

		file_name = file_path + '/' + 'train_task' + str(k) + '_Dport.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(total_D_set, f)

		file_name = file_path + '/' + 'train_task' + str(k) + '_Eport.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(total_E_set, f)

		k += 1
	print("After processing the trainset, k=%d" % k)

	# =====================================Construct meta-test dataset=====================================
	k = 1
	test_dict = dict()
	for session_speaker in test_key:
		test_mat = []
		test_Y = []
		E_set = {i: 0 for i in range(4)}
		V_set = {i: 0 for i in range(4)}
		A_set = {i: 0 for i in range(4)}
		D_set = {i: 0 for i in range(4)}
		file_mat_lab_dict = data_dict.get(session_speaker)
		for wav_file in file_mat_lab_dict:
			test_dict[wav_file] = []
			segment_mat = []
			mat_lab_list = file_mat_lab_dict.get(wav_file)
			if bool_vad:
				for mat, label, valence, arouse, domain in mat_lab_list:
					mat = mat - bias
					mat = (mat - mat.mean())/mat.std()
					test_mat.append(mat)
					emo = lab_2_index.get(label)
					test_Y.append([emo, valence, arouse, domain])
					E_set[emo] += 1
					V_set[valence] += 1
					A_set[arouse] += 1
					D_set[domain] += 1
					segment_mat.append(mat)
			else:
				for mat, label in mat_lab_list:
					mat = mat - bias
					mat = (mat - mat.mean()) / mat.std()
					test_mat.append(mat)
					test_Y.append(label)
					segment_mat.append(mat)
			test_dict[wav_file] = [segment_mat, emo]
		test_amount = len(test_Y)
		test_idx = list(range(test_amount))
		random.shuffle(test_idx)
		print(session_speaker)
		print("Emotion:", E_set)
		print("Valence:", V_set)
		print("Arousal:", A_set)
		print("Dominance:", D_set)

		features = []
		labels = []
		for idx in test_idx:
			features.append(test_mat[idx])
			labels.append(test_Y[idx])

		file_name = file_path + '/' + 'test_set.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(test_dict, f)

		file_name = file_path + '/' + 'test_task' + str(k) + '_mat.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(features, f)
		file_name = file_path + '/' + 'test_task' + str(k) + '_lab.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(labels, f)

		amount = test_amount

		file_name = file_path + '/' + 'test_task' + str(k) + '_Vport.pkl'
		portion_dict = {i: amount / V_set.get(i) for i in range(4)}
		with open(file_name, 'wb') as f:
			pickle.dump(portion_dict, f)

		file_name = file_path + '/' + 'test_task' + str(k) + '_Aport.pkl'
		portion_dict = {i: amount / A_set.get(i) for i in range(4)}
		with open(file_name, 'wb') as f:
			pickle.dump(portion_dict, f)

		file_name = file_path + '/' + 'test_task' + str(k) + '_Dport.pkl'
		portion_dict = {i: amount / D_set.get(i) for i in range(4)}
		with open(file_name, 'wb') as f:
			pickle.dump(portion_dict, f)

		file_name = file_path + '/' + 'test_task' + str(k) + '_Eport.pkl'
		portion_dict = {i: amount / E_set.get(i) for i in range(4)}
		with open(file_name, 'wb') as f:
			pickle.dump(portion_dict, f)

		k += 1
	print("After processing the testset, k=%d" % k)


def main(cross_num=5, exp_path='', sav_dir='', conf='', cfg_sec='', bool_vad=False):

	if not os.path.exists(exp_path):
		os.mkdir(exp_path)

	for i in range(1, 6):
		session_name = 'Session' + str(i)
		session_path = exp_path + '/' + session_name
		if not os.path.exists(session_path):
			os.mkdir(session_path)

	if not os.path.exists(sav_dir):
		os.mkdir(sav_dir)

	thread_list = []
	lab_list = []
	for i in range(5):
		t = MyThread(get_emo_data, args=('Session' + str(i + 1), True, bool_vad, 'spectrogram', 3,
										 exp_path + 'Session' + str(i+1) + '/', conf, cfg_sec))
		thread_list.append(t)

	for t in thread_list:
		t.start()

	for t in thread_list:
		t.join()
		lab_list.append(t.get_result())

	lab_dict = dict()
	for item in lab_list:
		lab_dict.update(item)


	min_val = np.Inf

	freq_bag = set()
	time_bag = set()

	for sess_spk in lab_dict.keys():
		# sess_spk aims to specific session of F/M,
		# wav_file_dict: {wav_file_name: [[mat1,lab1], [mat2, lab2] ...], wav_file_name: [[mat1,lab1], [mat2,
		# lab2] ...] , ..}
		wav_file_info_dict = lab_dict.get(sess_spk)
		print(sess_spk, len(list(wav_file_info_dict.keys())))
		for wav_file in wav_file_info_dict:
			info_list = wav_file_info_dict.get(wav_file)
			for mat, label, valence, arouse, domain in info_list:
				min_val = min(min_val, mat.min())
				# max_length = max(max_length, mat.shape[1])
				# min_length = min(min_length, mat.shape[1])
				time_bag.add(mat.shape[1])
				freq_bag.add(mat.shape[0])
	#
	print('Time bag:\n\t', time_bag)
	print('Freq bag:\n\t', freq_bag)
	print(max(time_bag), min(time_bag))
	bias = min_val-1
	print(bias)

	sess_spk = set(lab_dict.keys())
	if cross_num == 5:
		for i in range(1, 6):
			# construct the dataset for five-fold cross validation
			cross_val = 'leave_' + str(i)
			file_path = sav_dir + '/' + cross_val
			dev_key = {'Session' + str(i) + '_F'}
			test_key = {'Session' + str(i) + '_M'}
			train_key = sess_spk - test_key - dev_key
			if not os.path.exists(file_path):
				os.mkdir(file_path)
			print(test_key)
			store(file_path=file_path, train_key=train_key, test_key=test_key, data_dict=lab_dict, bias=bias,
				  bool_vad=bool_vad)
			print()
	else:
		sess_spk_list = list(sess_spk)
		# construct the dataset for ten-fold cross validation
		for i in range(1, 11):
			idx = sess_spk_list[i-1]
			cross_val = 'leave_' + str(i)
			file_path = data_path_prefix + '10cross_set/' + cross_val
		test_key = {idx}
		train_key = sess_spk - test_key
		if not os.path.exists(file_path):
			os.mkdir(file_path)
		store(file_path=file_path, train_key=train_key, test_key=test_key, data_dict=lab_dict)
