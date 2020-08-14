import configparser

def read_spectrogram_conf(conf_name, section_name):
	cf = configparser.ConfigParser()
	cf.read(conf_name)
	win_shift = int(cf.get(section_name, 'window_shift'))
	win_size = int(cf.get(section_name, 'window_size'))
	dft_length  = int(cf.get(section_name, 'dft_length'))
	freq_min  = int(cf.get(section_name, 'freq_min'))
	freq_max = int(cf.get(section_name, 'freq_max'))
	hz_resultion = int(cf.get(section_name, 'hz_resultion'))
	return win_shift, win_size, dft_length, freq_min, freq_max, hz_resultion


def read_set_tuple(cfg, sec_name, sec_key, split_s='/', split_t=','):
	item = cfg.get(sec_name, sec_key)
	item = item.split(split_s)
	result = []
	for each in item:
		tmp = each.split(split_t)
		result.append(tuple(map(int, tmp)))
	return result


def read_set_list_single(cfg, sec_name, sec_key, split_s=',', bool_single=False):
	if bool_single:
		return int(cfg.get(sec_name, sec_key))
	else:
		item = cfg.get(sec_name, sec_key)
		item = item.split(split_s)
		result = []
		for each in item:
			result.append(int(each))
		return result