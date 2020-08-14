from read_conf import read_set_tuple, read_set_list_single
import torch
import torch.nn as nn
import numpy as np
import random
import configparser
import pickle
import math
from copy import deepcopy
from sklearn.metrics.classification import confusion_matrix
import sys
import argparse
from torch.autograd import Variable
from torch.nn import functional as F
import time
from CRNN import CRNN
from utils import get_info_args, get_crnn_paras, get_WA, get_UA, get_device_idx, point_to_grad, average_grad, model_zero_grad

parser = argparse.ArgumentParser()
parser.add_argument('--main-path', type=str, help='indicate prefix of the cross-validation path ',
					default='/data0/gkb_dataset/meta_multi_task/')
parser.add_argument('--dataset', type=str, help='indicate the name of dataset ',
					default='/data0/gkb_dataset/meta_multi_task/')
parser.add_argument('--ckt-path', type=str, help='indicate the name of dataset ',
					default='/data0/gkb_dataset/meta_multi_task/')
parser.add_argument('--conf', type=str, help='the config file of the network', default='20hz_nn.conf')
parser.add_argument('--cnn2rnn', type=str, help='indicate the way to put cnn output to rnn ', default='concat')
parser.add_argument('--rnn2dnn', type=str, help='indicate the way to put rnn output to dnn ', default='Avg')
parser.add_argument('--penlty', type=str, help='indicate if penlty different class ', default='T')
parser.add_argument('--task-num', type=int, help='indicate the number of task on meta-training', default=8)
parser.add_argument('--freq', type=int, help='indicate the maximum value in frequency domain ', default=200)
parser.add_argument('--out-dim', type=int, help='indicate the number of output label', default=4)

parser.add_argument('--smooth', type=str, help='indicate if smooth used', default='F')
parser.add_argument('--smooth-value', type=float, help='indicate smooth value', default=0.1)

parser.add_argument('--inner-loop', type=int, help='indicate the number of inner loop in meta-train', default=4)
parser.add_argument('--train-batch', type=int, help='indicate the value of outer batch size in meta-train', default=64)
parser.add_argument('--test-batch', type=int, help='indicate the value of outer batch size in meta-test', default=128)
parser.add_argument('--train-loop', type=int, help='indicate the value of outer iteration in '
												   'meta-train', default=300)
parser.add_argument('--test-loop', type=int, help='indicate the value of outer iteration in '
												  'meta-test', default=300)
parser.add_argument('--train-eval', type=int, help='indicate the epoch num to turn to meta-test in meta-train',
					default=20)
parser.add_argument('--test-eval', type=int, help='indicate the epoch num to turn to eval the main task in meta-task',
					default=5)

parser.add_argument('--gpu', type=int, help='indicate the gpu to use in demo ', default=3)
parser.add_argument('--start-cross', type=int, help='indicate which cross validation to start ',
					default=1)
parser.add_argument('--end-cross', type=int, help='indicate which cross validation to end ', default=6)

parser.add_argument('--sub-task', type=int, help='indicate the number of sub task of each task', default=3)
parser.add_argument('--inner-lr', type=float, help='indicate the inner learning rate', default=0.1)
parser.add_argument('--outer-lr', type=float, help='indicate the outer learning rate', default=0.001)
parser.add_argument('--fine-tune', type=float, help='indicate the learning rate for fine-tuning', default=0.0005)

parser.add_argument('--start-base', type=int, help='indicate the the base number to start meta-test, so the meta test'
												   'will start at train_eval * start_base', default=1)
parser.add_argument('--base-length', type=int, help='indicate the the max base number to start meta-test, so the meta-'
												  'test will start at train_eval * start_base', default=15)

paras = parser.parse_args()

main_path = paras.main_path
dataset = paras.dataset
conf = main_path + '/config/' + paras.conf
crossprefix = dataset + '/'

task_num = paras.task_num
max_freq = paras.freq
out_dim = paras.out_dim

cnn2rnn = paras.cnn2rnn
rnn2dnn = paras.rnn2dnn
if cnn2rnn not in ['concat', 'sum', 'avg', 'max']:
	raise NotImplementedError
if rnn2dnn not in ['Avg', 'Sum', 'Max', 'L-concat', 'FB-concat']:
	raise NotImplementedError

smoothing, confidence, name_smooth, env_name, ckp_path, penlty_bool = get_info_args(paras.penlty, paras.smooth, paras.smooth_value, paras.ckt_path, max_freq, cnn2rnn, rnn2dnn)

gpu = paras.gpu
gpus = [gpu]
torch.cuda.set_device(gpu)
dev_test_gpus = "cuda:" + str(gpu)

GRAD_CLIP = 2
max_esplon = 1e-6
cross_idx_list = list(range(paras.start_cross, paras.end_cross))
lab_m_list = ['neu', 'ang', 'hap', 'sad']
task_lab = ['meta_test_v.pkl', 'meta_test_a.pkl', 'meta_test_d.pkl']
task_pen = ['meta_test_v_portion.pkl', 'meta_test_a_portion.pkl', 'meta_test_d_portion.pkl']


def get_pen_dict(penlty_dict, lab_tensor, batch_num):
	if paras.smooth == 'T':
		if penlty_bool == True:
			penlty = [list(map(penlty_dict.get, range(4))) for _ in range(batch_num)]
		else:
			penlty = [list([1.0 for _ in range(4)]) for _ in range(batch_num)]
	else:
		if penlty_bool == True:
			penlty = list(map(penlty_dict.get, lab_tensor))
		else:
			penlty = list([1.0 for _ in range(batch_num)])
	return penlty


def get_task_batch(task_n_dict):
	'''
	get the batch size of each task, and each task shares the train_step
	:param task_n_dict: {task_1: N_1, task_2: N_2, ... task_n: N_n}
	:return:
	'''
	task_n_list = [task_n_dict.get(i) for i in range(paras.task_num)]
	task_p_list = np.array(task_n_list) / max(task_n_list)
	task_batch_dict = list(np.floor(paras.train_batch * task_p_list))
	train_steps = math.floor(max(task_n_list) / paras.train_batch)
	return train_steps, task_batch_dict


def comput_loss(criterion, model, fea_tensor, lab_tensor, penlty, mode='query'):
	
	if paras.smooth == 'T':
		out = model(fea_tensor, mode)
		device = out.get_device()
		true_dist = torch.zeros(out.size()).cuda(device=device)
		true_dist.scatter_(1, lab_tensor.data.unsqueeze(1), 1)
		true_dist = confidence * true_dist + smoothing / paras.out_dim
		true_dist.requires_grad = False
		loss = (((-torch.log(out) * true_dist) * penlty).sum(dim=1)).mean()
	# loss = (criterion(torch.log(out), true_dist) * penlty).mean()
	else:
		loss = (criterion(model(fea_tensor, mode), lab_tensor) * penlty).mean()

	return loss


def meta_test(model, optim_state, cross_idx, cross_path, learning_rate=0.001, outer_epoch=0,
			  ckp_pth=None, penlty_bool=True):
	
	
	#////////////////////////////////////////////////////////////////////////////////// #
	#                                                                                   #
	#            Indicate the loss function and initialize the optimizer                #
	#                                                                                   #
	#////////////////////////////////////////////////////////////////////////////////// #
	auxi_model = deepcopy(model)
	print('============================Meta-test Training start============================')

	criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)
	model_para_n = len(list(model.parameters()))
	for idx, para_p in enumerate(model.parameters()):
		if idx < (model_para_n-2):
			para_p.requires_grad = True
		else:
			para_p.requires_grad = False
	main_optimizer = torch.optim.Adam(list(model.parameters())[:-2], lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
									  weight_decay=0)
	auxi_optimizer = torch.optim.Adam(list(auxi_model.parameters())[:-2], lr=learning_rate, betas=(0.9, 0.999),
									 eps=1e-08, weight_decay=0)

	old_state = deepcopy(optim_state)
	main_optimizer.load_state_dict(deepcopy(old_state))
	auxi_optimizer.load_state_dict(deepcopy(old_state))
	device_idx = get_device_idx(model)

	#////////////////////////////////////////////////////////////////////////////////// #
	#                                                                                   #
	#                Load dataset and compute weight for each class                     #
	#                                                                                   #
	#////////////////////////////////////////////////////////////////////////////////// #
	fea_list = []
	lab_list = []
	pen_list = [[] for _ in range(1)]
	task_n_dict = {k: 0 for k in range(1)}
	for k in range(1):
		file_name = cross_path + '/' + 'test_task' + str(k + 1) + '_mat.pkl'
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
			fea_list.append(deepcopy(data))
			task_n_dict[k] = len(data)
			del data

		file_name = cross_path + '/' + 'test_task' + str(k + 1) + '_lab.pkl'
		with open(file_name, 'rb') as f:
			lab_list.append(pickle.load(f))

		file_name = cross_path + '/' + 'test_task' + str(k + 1) + '_Vport.pkl'
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
			max_t = max(data.values())
			data = {k: data[k] / max_t for k in data.keys()}
			(pen_list[k]).append(data)

		file_name = cross_path + '/' + 'test_task' + str(k + 1) + '_Aport.pkl'
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
			max_t = max(data.values())
			data = {k: data[k] / max_t for k in data.keys()}
			(pen_list[k]).append(data)

		file_name = cross_path + '/' + 'test_task' + str(k + 1) + '_Dport.pkl'
		with open(file_name, 'rb') as f:
			data = pickle.load(f)
			max_t = max(data.values())
			data = {k: data[k] / max_t for k in data.keys()}
			(pen_list[k]).append(data)

	train_batch_epo = math.floor(len(lab_list[0])/paras.test_batch)
	task_batch_dict = {0: paras.test_batch}
	task_n_dict = {0: len(lab_list[0])}
	task_idx_order = [0]

	txt_name = ckp_pth + '/' + 'Leave_' + cross_idx + '_Outer_' + str(outer_epoch) + '_meta_test.txt'
	with open(txt_name, 'w') as f:

		#////////////////////////////////////////////////////////////////////////////////// #
		#                                                                                   #
		#                              Multi-train stage only                               #
		#                                                                                   #
		#////////////////////////////////////////////////////////////////////////////////// #
		
		for epoch in range(paras.test_loop):

			real_loss = 0.0

			model.train()
			auxi_model.train()

			for j in range(train_batch_epo):

				main_optimizer.zero_grad()

				old_var = model.state_dict()

				for k in range(1):

					task_idx = task_idx_order[k]  # select the correspongding task index
					batch = task_batch_dict[task_idx]  # get the corresponding task batch

					# get the start/end index of the training task
					skip = int(j * batch)
					if j == (train_batch_epo - 1):
						skep = task_n_dict[task_idx]
					else:
						skep = (j + 1) * batch
					skep = int(skep)
					batch_num = skep - skip

					# select k-th task information
					pen_task = pen_list[task_idx]
					fea = fea_list[task_idx][skip:skep]
					lab = lab_list[task_idx][skip:skep]
					lab = np.array(lab)
					lab_V, lab_A, lab_D = lab[:, 1], lab[:, 2], lab[:, 3]
					lab_V = list(lab_V); lab_A = list(lab_A); lab_D = list(lab_D)
					pen_V = get_pen_dict(pen_task[0], lab_V, batch_num)
					pen_A = get_pen_dict(pen_task[1], lab_A, batch_num)
					pen_D = get_pen_dict(pen_task[2], lab_D, batch_num)

					train_set = zip((fea, fea, fea), (lab_V, lab_A, lab_D), (pen_V, pen_A, pen_D))
					for fea_mat, lab_mat, pen in train_set:

						auxi_model.load_state_dict(deepcopy(old_var))
						auxi_optimizer.zero_grad()
						auxi_optimizer.load_state_dict(deepcopy(old_state))
						auxi_optimizer.zero_grad()

						if not isinstance(fea_mat, list):
							lab_mat = [lab_mat]
							fea_mat = [fea_mat]
						fea_tensor = torch.tensor(fea_mat).type(torch.FloatTensor).cuda(gpus[0])
						lab_tensor = torch.tensor(lab_mat).cuda(gpus[0])
						penlty = torch.tensor(pen).cuda(gpus[0])

						loss = comput_loss(criterion=criterion, model=auxi_model, fea_tensor=fea_tensor,
										   lab_tensor=lab_tensor, penlty=penlty, mode='support')
						loss.backward()
						real_loss += loss.item()
						point_to_grad(model, auxi_model, cuda_bool=True, deviec_idx=device_idx, base=paras.sub_task)

				#////////////////////////////////////////////////////////////////////////////////// #
				#                                                                                   #
				#                                        Backward                                   #
				#                                                                                   #
				#////////////////////////////////////////////////////////////////////////////////// #
				auxi_optimizer.zero_grad()
				main_optimizer.step()
				main_optimizer.zero_grad()
				old_state = main_optimizer.state_dict()

			
			#////////////////////////////////////////////////////////////////////////////////// #
			#                                                                                   #
			#                                Testing or Evaluting                               #
			#                                                                                   #
			#////////////////////////////////////////////////////////////////////////////////// #
			real_loss = real_loss / train_batch_epo / paras.sub_task
			if ((epoch + 1) % paras.test_eval) == 0:

				UA, WA = test(deepcopy(model), cross_idx=cross_idx, cross_path=cross_path, inner_epoch=epoch + 1,
							  outer_epoch=outer_epoch, ckp_pth=ckp_pth, device=dev_test_gpus)
				f.writelines('Testing Outer epoch: ' + str(outer_epoch) + ' Inner epoch: ' + str(epoch + 1) + '\n')
				f.writelines('Testing UA: ' + str(float(UA)) + '\n')
				f.writelines('Testing WA: ' + str(float(WA)) + '\n')

				f.writelines('\n')

	print('============================Training Finish============================')


def test(model, cross_idx, cross_path, inner_epoch=0, outer_epoch=0, ckp_pth=None, device=None):
	# print()
	model.eval()
	print('============================Meta-test Testing Start============================')

	all_conf_mat = np.zeros((len(lab_m_list), len(lab_m_list)))

	with torch.no_grad():

		with open(cross_path + 'test_set.pkl', 'rb') as f:
			data = pickle.load(f)

		ses_names = list(data.keys())
		for wav_file in ses_names:
			mat, true_y = data.get(wav_file)
			true_y = np.array([true_y])
			mat = torch.tensor(mat).type(torch.FloatTensor).to(device)
			pred_y = model(mat, mode='query').sum(dim=0)
			pred_y = ((pred_y.topk(1))[1]).data.cpu().flatten().tolist()
			confusion_mat = confusion_matrix(true_y, pred_y, labels=[0, 1, 2, 3])
			all_conf_mat += confusion_mat

	all_conf_mat = all_conf_mat.T
	UA_metric = get_UA(all_conf_mat)
	WA_metric = get_WA(all_conf_mat)
	npy_name = ckp_pth + '/' + 'Leave_' + cross_idx + '_Outer_' + str(outer_epoch) + '_Inner_' + str(inner_epoch) + \
			   '_test.npy'
	np.save(npy_name, all_conf_mat)

	print('============================Meta-test Testing Finish============================')
	print()

	return UA_metric, WA_metric


def main(env_name, ckt_path, cross_idx_list, crossprefix, net_conf, cnn2rnn, rnn2dnn, outer_range_list,
		 penlty_bool=True):
	import visdom
	import os

	if not os.path.exists(ckt_path):
		os.mkdir(ckt_path)

	cfg = configparser.ConfigParser()
	cfg.read(net_conf)
	# in_chan, out_chan, kernel_size, stride
	cnn_setting, rnn_setting, pooling_setting, dnn_setting, drop_rate = get_crnn_paras(cfg, max_freq, cnn2rnn)

	for j in cross_idx_list:

		print("=================================================Run the leave %d speaker programming"
			  "=================================================" % j)
		print(time.localtime(time.time()))

		cross_idx = str(j)
		cross_path = crossprefix + 'leave_' + cross_idx + '/'

		for outer_epoch in outer_range_list:

			para_file = ckt_path + 'Leave_' + cross_idx + '_Outer_' + str(outer_epoch) + '_para.tar'
			print(para_file)
			if not os.path.exists(para_file):
				print("para_file should exist!")
				raise FileNotFoundError
			print("Reload the para_file: %s" % para_file)
			para_file = torch.load(para_file)

			seed = 2234
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)
			if torch.cuda.is_available():
				torch.cuda.manual_seed_all(seed)
			torch.backends.cudnn.deterministic = True

			asr_model = CRNN(out_dim=4, cnn_setting=cnn_setting, rnn_setting=rnn_setting,
							 pooling_setting=pooling_setting, dnn_setting=dnn_setting, dropout_rate=drop_rate,
							 cnn2rnn=cnn2rnn, rnn2dnn=rnn2dnn)

			asr_model.cuda(gpus[0])
			asr_model.load_state_dict(para_file['model_state_dict'])
			main_optimizer_state = para_file['main_optimizer_share']

			meta_test(asr_model, optim_state=main_optimizer_state, cross_idx=cross_idx, cross_path=cross_path,
					  learning_rate=0.001, outer_epoch=outer_epoch, ckp_pth=ckt_path, penlty_bool=penlty_bool)
			del asr_model
			del main_optimizer_state

		print("=================================================Finish the leave %d speaker programming"
			  "=================================================" % j)
		print()


if __name__ == "__main__":
	out_epoch_start = paras.start_base
	out_epoch_end = paras.start_base + paras.base_length
	out_epoch_range = [paras.train_eval * i for i in range(out_epoch_start, out_epoch_end)]
	main(env_name=env_name, ckt_path=ckp_path, cross_idx_list=cross_idx_list, crossprefix=crossprefix, net_conf=conf, cnn2rnn=cnn2rnn, rnn2dnn=rnn2dnn, outer_range_list=out_epoch_range, penlty_bool=penlty_bool)