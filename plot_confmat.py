import numpy as np
from plot_cm import plot_confusion_matrix

save_path = '/data0/gkb_dataset/meta_multi_task_db_ext/result_jpg/'

file_dict = {
	'MTLReptileFCT1-200-concat_Avg-T-': ['/data0/gkb_dataset/meta_multi_task_db_ext/new_idea_task_result/', '-WA-1st-mat.npy', 'MTLReptile.pdf'],
	'Non-multi2-200-concat_Avg-T-': ['/data0/gkb_dataset/meta_multi_task_db_ext/new_idea_task_result/', '-WA-1st-mat.npy', 'Multitask.pdf'],
	'200-concat_Avg-T-': ['/data0/gkb_dataset/old_idea/new_single_task_result/', '-F1-1st-mat.npy', 'baseline.pdf']
}

file_name = file_dict.keys()

for each in file_name:
	prefix, suffix, photo_name = file_dict.get(each)
	mat_name = prefix + each + suffix
	mat = np.load(mat_name)
	mat = mat.T
	plot_confusion_matrix(cm=mat,
						  target_names=['neu', 'ang', 'hap', 'sad'],
						  title='Confusion matrix',
						  cmap=None,
						  normalize=True,
						  save_path=save_path,
						  save_name=photo_name)