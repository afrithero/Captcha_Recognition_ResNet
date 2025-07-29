import cv2
import scipy.ndimage
import numpy as np
import Augmentor
import data_config
import os
from os.path import join

def remove_noise(img):
	img = ~img # white characters, black background
	img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 1) # weaken circle noise and line noise
	img = ~img # black letters, white background
	img = scipy.ndimage.median_filter(img, (5, 1)) # remove line noise
	img = scipy.ndimage.median_filter(img, (1, 3)) # weaken circle noise
	img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 1) # dilate image to initial stage
	img = scipy.ndimage.median_filter(img, (3, 3)) # remove remaning noise 
	return img

def generate_augmented_data(path, count):
	p = Augmentor.Pipeline(path)
	p.zoom(probability=1, min_factor=1.05, max_factor=1.05)
	p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
	p.sample(count*2)
	return p

if __name__ == '__main__':
	# 手動產生 augmented data 然後搬移檔案
	task1_path = join(data_config.ROOT_PATH, "dataset", "train", "task1")
	task1_count = len(os.listdir(task1_path))
	p = generate_augmented_data(task1_path,task1_count)
	task2_path = join(data_config.ROOT_PATH, "dataset", "train", "task2")
	task2_count = len(os.listdir(task2_path))
	p = generate_augmented_data(task2_path,task2_count)
	task3_path = join(data_config.ROOT_PATH, "dataset", "train", "task3")
	task3_count = len(os.listdir(task3_path))
	p = generate_augmented_data(task3_path,task3_count)

	task1_aug_files = os.listdir(join(task1_path, 'output'))
	task2_aug_files = os.listdir(join(task2_path, 'output'))
	task3_aug_files = os.listdir(join(task3_path, 'output'))
	
	for file in task1_aug_files:
		temp = file.split('_')
		new_name = 'aug_' + temp[2]
		source = join(task1_path,'output',file)
		dest = join(task1_path, new_name)
		os.rename(source, dest)

	for file in task2_aug_files:
		temp = file.split('_')
		new_name = 'aug_' + temp[2]
		source = join(task2_path,'output',file)
		dest = join(task2_path, new_name)
		os.rename(source,dest)

	for file in task3_aug_files:
		temp = file.split('_')
		new_name = 'aug_' + temp[2]
		source = join(task3_path,'output',file)
		dest = join(task3_path, new_name)
		os.rename(source,dest)

	task1_files = os.listdir(task1_path)
	task2_files = os.listdir(task2_path)
	task3_files = os.listdir(task3_path)

	os.rmdir(join(task1_path,'output'))
	os.rmdir(join(task2_path,'output'))
	os.rmdir(join(task3_path,'output'))

	print('now processing task1')
	for file_name in task1_files[1:]:
		if file_name == 'output':
			continue
		img = cv2.imread(join(task1_path, file_name),  cv2.IMREAD_GRAYSCALE)
		img = remove_noise(img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		cv2.imwrite(join(task1_path, f'clean_{file_name}'), img)

	print('now processing task2')
	for file_name in task2_files[1:]:
		if file_name == 'output':
			continue
		img = cv2.imread(join(task2_path, file_name),  cv2.IMREAD_GRAYSCALE)
		img = remove_noise(img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		cv2.imwrite(join(task2_path, f'clean_{file_name}'), img)

	print('now processing task3')
	for file_name in task3_files[1:]:
		if file_name == 'output':
			continue
		img = cv2.imread(join(task3_path, file_name),  cv2.IMREAD_GRAYSCALE)
		img = remove_noise(img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		cv2.imwrite(join(task3_path, f'clean_{file_name}'), img)