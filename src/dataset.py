import torch
import cv2
import numpy as np
import pandas as pd
import data_encoding
import data_config
import data_preprocess
import glob
from torch.utils.data import Dataset, DataLoader
from os.path import join
from sklearn.model_selection import train_test_split


class CaptchaDataset(Dataset):
		def __init__(self, root_path, df, is_predict=False):
			self.root_path = root_path
			self.data = df["filename"].tolist()
			raw_labels = df["label"].tolist()
			self.is_predict = is_predict
			if is_predict:
				self.labels = np.array(raw_labels)
			else:
				self.labels = np.array([data_encoding.encode(label) for label in raw_labels])
			
		def __getitem__(self, index):
			file_name, label = self.data[index], self.labels[index]
			
			if self.is_predict:
				img = cv2.imread(join(self.root_path, "dataset", "test", file_name),  cv2.IMREAD_GRAYSCALE)
			else:
				img = cv2.imread(join(self.root_path, "dataset", "train", file_name),  cv2.IMREAD_GRAYSCALE)

			img = img.reshape(img.shape[0],img.shape[1], -1)
			img = (img - 128) / 128
			img = np.transpose(img,(2,0,1)) #因為 Conv2D channel 要在第一個 dim，所以做轉換
			return torch.tensor(img,dtype=torch.float), torch.tensor(label, dtype=torch.float), file_name
			
		def __len__(self):
			return len(self.data)

		
def split_train_val(df, random_seed=42):
	np.random.seed(random_seed)
	np.random.seed(42)
	df_task1 = df[df["filename"].str.startswith("task1")]
	df_task1_train, df_task1_val = train_test_split(df_task1, test_size=0.2)

	df_task2 = df[df["filename"].str.startswith("task2")]
	df_task2_train, df_task2_val = train_test_split(df_task2, test_size=0.2)

	df_task3 = df[df["filename"].str.startswith("task3")]
	df_task3_train, df_task3_val = train_test_split(df_task3, test_size=0.2)

	df_train = pd.concat([df_task1_train, df_task2_train, df_task3_train], axis = 0)
	df_val = pd.concat([df_task1_val, df_task2_val, df_task3_val], axis = 0)

	return df_train, df_val

def parse_annotation(root_path):
	df = pd.read_csv(join(root_path, "dataset", "train", "annotations.csv"))
	task1_aug = glob.glob(join(root_path,'dataset','train','task1','aug*'))
	task2_aug = glob.glob(join(root_path,'dataset','train','task2','aug*'))
	task3_aug = glob.glob(join(root_path,'dataset','train','task3','aug*'))

	task1_clean = glob.glob(join(root_path,'dataset','train','task1','clean*'))
	task2_clean = glob.glob(join(root_path,'dataset','train','task2','clean*'))
	task3_clean = glob.glob(join(root_path,'dataset','train','task3','clean*'))
	
	task1_temp = []
	task2_temp = []
	task3_temp = []

	for file in task1_aug:
		a = file.split('/')
		b = join(a[-2],a[-1])
		task1_temp.append(b)

	for file in task1_clean:
		a = file.split('/')
		b = join(a[-2],a[-1])
		task1_temp.append(b)

	for file in task2_aug:
		a = file.split('/')
		b = join(a[-2],a[-1])
		task2_temp.append(b)

	for file in task2_clean:
		a = file.split('/')
		b = join(a[-2],a[-1])
		task2_temp.append(b)

	for file in task3_aug:
		a = file.split('/')
		b = join(a[-2],a[-1])
		task3_temp.append(b)

	for file in task3_clean:
		a = file.split('/')
		b = join(a[-2],a[-1])
		task3_temp.append(b)
	
	all_temp = task1_temp + task2_temp + task3_temp

	df_temp = pd.DataFrame({
		'filename' : all_temp,
		'label' : ''
	})

	for i in df_temp['filename']:
		keywords = i.split('_')
		df_temp.loc[df_temp["filename"] == i, ["label"]] = df.loc[df["filename"].str.contains(keywords[-1])]['label'].item()

	df_merged = pd.concat([df,df_temp], axis = 0)

	return df_merged

def parse_submission(root_path):
	def assign_label_len(x):
		if x.startswith('task1'):
				return 1
		elif x.startswith('task2'):
				return 2
		else:
				return 4
	df = pd.read_csv(join(root_path, 'dataset', 'sample_submission.csv'))
	df['label'] = df['filename'].apply(assign_label_len)
	
	return df


if __name__ == "__main__":
	df = parse_annotation(data_config.ROOT_PATH)
	df_train, df_val = split_train_val(df, random_seed=42)
	train_ds = CaptchaDataset(data_config.ROOT_PATH, df_train)
	train_dl = DataLoader(train_ds, batch_size=100, num_workers=4, drop_last=True, shuffle=True)
	
	df = parse_submission(data_config.ROOT_PATH)
	test_ds = CaptchaDataset(data_config.ROOT_PATH, df, is_predict=True)
	test_dl = DataLoader(test_ds, batch_size=100, num_workers=4, drop_last=True, shuffle=True)
