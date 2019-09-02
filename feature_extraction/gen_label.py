import argparse
import numpy as np
import os
label_lst = ['Alarm_bell_ringing','Blender','Cat','Dishes','Dog','Electric_shaver_toothbrush','Frying','Running_water','Speech','Vacuum_cleaner']
LEN = 500
def gen_label(input_file, label_dir):
	label_index = {}
	for i, u in enumerate(label_lst):
		label_index[u] = i
	with open(input_file) as f:
		fs = f.readlines()
	fs = [f.rstrip() for f in fs]
	for f in fs:
		tmp = f.split('\t')
		fname = tmp[0]
		label = np.zeros([len(label_lst)])
		if len(tmp) > 1:
			labels = tmp[1].split(',')
			for u in labels:
				label[label_index[u]] = 1
		path = os.path.join(label_dir, str.replace(fname, '.wav',''))
		print('%s done'%path)
		np.save(path, label)

def gen_label_for_detail_csv(input_file, label_dir):
	label_index = {}
	for i, u in enumerate(label_lst):
		label_index[u] = i
	with open(input_file) as f:
		fs = f.readlines()
	fs = [f.rstrip() for f in fs]
	result = {}
	for f in fs:
		tmp = f.split('\t')
		fname = str.replace(tmp[0], '.wav','')
		if fname not in result:
			result[fname] = np.zeros([len(label_lst)])
		if len(tmp) > 1:
			label = result[fname]
			label[label_index[tmp[-1]]] = 1
			result[fname] = label
	for re in result:
		path = os.path.join(label_dir, re)
		print('%s done'%path)
		np.save(path, result[re])

def gen_detail_label_for_detail_csv(input_file, label_dir):
	label_index = {}
	for i, u in enumerate(label_lst):
		label_index[u] = i
	with open(input_file) as f:
		fs = f.readlines()
	fs = [f.rstrip() for f in fs]
	result = {}
	for f in fs:
		tmp = f.split('\t')
		fname = str.replace(tmp[0], '.wav','')
		if fname not in result:
			result[fname] = np.zeros([LEN, len(label_lst)])
		if len(tmp) > 1:
			st = int(float(tmp[1]) * LEN / 10)
			ed = int(float(tmp[2]) * LEN / 10)
			label = result[fname]
			label[st:ed, label_index[tmp[-1]]] += 1
			result[fname] = label
	for re in result:
		path = os.path.join(label_dir, re)
		a = result[re]
		a[a > 1] = 1
		print('%s done'%path)
		np.save(path, a)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('-c','--wav_csv', dest = 'wav_csv', 
                help = 'ground truth file of weak labels or detailed labels')

	parser.add_argument('-l','--label_dir', dest = 'label_dir', 
		help = 'dir for output labels')
	parser.add_argument('-d','--is_detailed', dest = 'is_detailed', 
		help = 'labels in ground truth file are detailed labels or not')
	parser.add_argument('-v','--convert_in_detail', 
		dest = 'convert_in_detail', help = 'generate detailed labels')
	f_args = parser.parse_args()
	wav_csv = f_args.wav_csv
	label_dir = f_args.label_dir
	assert os.path.exists(wav_csv)
	assert os.path.exists(label_dir)
	if f_args.is_detailed == 'True' or f_args.is_detailed == 'true':
		is_detailed = True
	else:
		is_detailed = False
	
	if f_args.convert_in_detail == 'True' or f_args.convert_in_detail == 'true':
		convert_in_detail = True
	else:
		convert_in_detail = False
	if convert_in_detail == True:
		gen_detail_label_for_detail_csv(wav_csv, label_dir)
	elif is_detailed:
		gen_label_for_detail_csv(wav_csv, label_dir)
	else:
		gen_label(wav_csv, label_dir)
	
