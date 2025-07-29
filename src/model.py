import torch.nn as nn
import torch
import data_config
import model_config
import data_encoding
import torch.nn.functional as F
from dataset import *
from tqdm import tqdm


class ResidualBlock(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(ResidualBlock, self).__init__()
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel, track_running_stats = True),
			nn.ReLU(),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(outchannel, track_running_stats = True))
		self.shortcut = nn.Sequential()

		# 如果 input 和 output 維度不一樣，不能直接相加，所以要再做一次卷積
		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias = False),
				nn.BatchNorm2d(outchannel, track_running_stats = True),
				nn.ReLU())

	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		return out

# ResNet
class ResNet(nn.Module): 
	def __init__(self, ResidualBlock, num_classes=62):
		super(ResNet, self).__init__()
		self.inchannel = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64, track_running_stats=True),
			nn.ReLU())
		# ResidualBlock basic
		# res34 3 4 6 3
		self.layer1 = self.make_layer(ResidualBlock, 64, 3, 1)
		self.layer2 = self.make_layer(ResidualBlock, 128, 4, 2)
		self.layer3 = self.make_layer(ResidualBlock, 256, 6, 2)
		self.layer4 = self.make_layer(ResidualBlock, 512, 3, 2)
		self.drop = nn.Dropout(0.5)
		self.rfc = nn.Sequential(nn.Linear(512, data_config.MAX_CAPTCHA*data_config.ALL_CHAR_SET_LEN))
		
	def make_layer(self, block, channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)  # strides = [1,1], to determine the stride for each layer in one residual block
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel, channels, stride))
			self.inchannel = channels
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x) # 100, 64, 96, 96
		x = self.layer1(x) # 100, 128, 48, 48
		x = self.layer2(x) # 100, 256, 24, 24
		x = self.layer3(x) # 100, 512, 12, 12
		x = self.layer4(x) # 100, 512, 1, 1
		x = nn.AdaptiveAvgPool2d(1)(x) # 100, 512
		x = x.view(-1, 512)
		x = self.drop(x)
		x = self.rfc(x) # 100, 248
		return x

# CNN Model (2 conv layer)
class CaptchaCNN(nn.Module):
	def __init__(self):
		super(CaptchaCNN, self).__init__()
		self.layer1 = nn.Sequential(
									nn.Conv2d(1, 32, kernel_size=3, padding=1),
									nn.BatchNorm2d(32),
									nn.Dropout(0.5),  # drop 50% of the neuron
									nn.ReLU(),
									nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(
									nn.Conv2d(32, 64, kernel_size=3, padding=1),
									nn.BatchNorm2d(64),
									nn.Dropout(0.5),
									nn.ReLU(),
									nn.MaxPool2d(2))
		self.layer3 = nn.Sequential(
			            nn.Conv2d(64, 64, kernel_size=3, padding=1),
									nn.BatchNorm2d(64),
									nn.Dropout(0.5),
									nn.ReLU(),
									nn.MaxPool2d(2))
		self.fc = nn.Sequential(
							nn.Linear((data_config.IMAGE_WIDTH//8)*(data_config.IMAGE_HEIGHT//8)*64, 1024),
							nn.Dropout(0.5),
							nn.ReLU())
		self.rfc = nn.Sequential(
							nn.Linear(1024, data_config.MAX_CAPTCHA*data_config.ALL_CHAR_SET_LEN))

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		out = self.rfc(out)
		return out

def fit(model, data_loader, optimizer, criterion, current_epoch):
	model.train()
	batches = tqdm(data_loader, total=len(data_loader))
	total_loss = 0
	for images, labels, _ in batches:
		with torch.autograd.set_detect_anomaly(True):
			images = images.to(model_config.DEVICE)
			labels = labels.to(model_config.DEVICE)
			preds = model(images)
			# print(preds)
			loss = criterion(preds, labels)
			batches.set_description(f'Epoch [{current_epoch+1}/{model_config.EPOCHS}]')
			batches.set_postfix(loss = loss.item())
			total_loss = total_loss + loss.item()
			# 初始化梯度
			optimizer.zero_grad()
			# 更新權重
			loss.backward()
			optimizer.step()
	print(f'Current Epoch:{current_epoch+1},Total Loss:{total_loss}')

def eval(model, data_loader):
		model.eval()
		model.load_state_dict(torch.load('./model_ResNet_all_data.pkl'))
		batches = tqdm(data_loader, total=len(data_loader), leave=False)
		correct_counts = 0

		for image, label, file_name in batches:
			image = image.to(model_config.DEVICE)
			label = label.to(model_config.DEVICE)
			pred = model(image)
			true_label = data_encoding.decode(label.data.cpu().numpy()[0])
			if len(true_label) == 1:
				text_0 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 0:data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				pred_decoded = f'{text_0}'
			elif len(true_label) == 2:
				text_0 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 0:data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				text_1 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, data_config.ALL_CHAR_SET_LEN:2*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				pred_decoded = f'{text_0}{text_1}'
			elif len(true_label) == 4:
				text_0 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 0:data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				text_1 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, data_config.ALL_CHAR_SET_LEN:2*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				text_2 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 2*data_config.ALL_CHAR_SET_LEN:3*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				text_3 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 3*data_config.ALL_CHAR_SET_LEN:4*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
				pred_decoded = f'{text_0}{text_1}{text_2}{text_3}'
			if pred_decoded == true_label:
				correct_counts += 1
			batches.set_description(f'[True:{true_label}/Pred:{pred_decoded}]')
			batches.set_postfix(file = file_name)
			tqdm.write(f'[file:{file_name}][True:{true_label}/Pred:{pred_decoded}]')
		
		print(f'Total:{len(data_loader)},Correct:{correct_counts},Accuracy:{correct_counts/len(data_loader)}')


def predict(model, data_loader, df):
	model.eval()
	model.load_state_dict(torch.load('./model_ResNet_all_data_34_epoch_50.pkl'))
	batches = tqdm(data_loader, total=len(data_loader), leave=False)
	preds = []

	for image, label, file_name in batches:
		image = image.to(model_config.DEVICE)
		pred = model(image)
		if label.item() == 1.0:
			text_0 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 0:data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			pred_decoded = f'{text_0}'
		elif label.item() == 2.0:
			text_0 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 0:data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			text_1 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, data_config.ALL_CHAR_SET_LEN:2*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			pred_decoded = f'{text_0}{text_1}'
		elif label.item() == 4.0:
			text_0 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 0:data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			text_1 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, data_config.ALL_CHAR_SET_LEN:2*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			text_2 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 2*data_config.ALL_CHAR_SET_LEN:3*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			text_3 = data_config.INDEX_TO_CAPTCHA_DICT[np.argmax(pred[0, 3*data_config.ALL_CHAR_SET_LEN:4*data_config.ALL_CHAR_SET_LEN].data.cpu().numpy())]
			pred_decoded = f'{text_0}{text_1}{text_2}{text_3}'
		
		preds.append(pred_decoded)
		batches.set_description(f'[label_length:{label.item()}/Pred:{pred_decoded}]')
		batches.set_postfix(file = file_name)
		tqdm.write(f'[file:{file_name}][label_length:{label.item()}/Pred:{pred_decoded}]')

	df['label'] = preds
	
	df.to_csv('./submission_ResNet_all_data_34_epoch_50.csv',index=False)


if __name__ == '__main__':
	df = parse_annotation(data_config.ROOT_PATH)
	model = ResNet(ResidualBlock)
	# model.to(model_config.DEVICE)
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model, dim=0)
	# train
	# df_train, df_val = split_train_val(df, random_seed=42)
	# train_ds = CaptchaDataset(data_config.ROOT_PATH, df_train)
	train_ds = CaptchaDataset(data_config.ROOT_PATH, df)
	train_dl = DataLoader(train_ds, batch_size=model_config.BATCH_SIZE, num_workers=4, drop_last=True, shuffle=True)
	model.to(model_config.DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=model_config.LR)
	criterion=nn.MultiLabelSoftMarginLoss()
	
	for epoch in range(model_config.EPOCHS):
		fit(model, data_loader=train_dl, optimizer=optimizer, criterion=criterion, current_epoch=epoch)
		if (epoch+1) % 50 == 0:
			torch.save(model.state_dict(), f"./model_ResNet_all_data_34_epoch_{epoch+1}.pkl") 

	print("save last model")

	# validate
	# eval_ds = CaptchaDataset(data_config.ROOT_PATH, df_val)
	# eval_dl = DataLoader(eval_ds, batch_size=1, num_workers=4, drop_last=True, shuffle=True)
	# eval(model, eval_dl)

	# predict
	# df_test = parse_submission(data_config.ROOT_PATH)
	# test_ds = CaptchaDataset(data_config.ROOT_PATH, df_test, is_predict=True)
	# test_dl = DataLoader(test_ds, batch_size=1, num_workers=4, drop_last=True, shuffle=False)	
	# predict(model, test_dl, df_test)