# src/trainer.py
import os
import torch
from tqdm import tqdm
import data_config, model_config
from dataset import parse_annotation, split_train_val, CaptchaDataset
from torch.utils.data import DataLoader
from model import ResNet, ResidualBlock
import data_encoding
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch

class Trainer:
    def __init__(self, model, device, batch_size, epochs, 
                save_dir, lr, val_batch_size=1, val_shuffle=False, val_drop_last=False):

        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.lr = lr
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.model = model(ResidualBlock).to(device)       

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
         
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()

        # split data and parse annotation
        df = parse_annotation(data_config.ROOT_PATH)
        df_train, df_val = split_train_val(df, random_seed=42)

        self.train_loader = DataLoader(
            CaptchaDataset(data_config.ROOT_PATH, df_train),
            batch_size=self.batch_size, shuffle=True,
            num_workers=4, drop_last=True)
        
        self.val_loader = DataLoader(
            CaptchaDataset(data_config.ROOT_PATH, df_val),
            batch_size=val_batch_size, shuffle=val_shuffle,
            num_workers=4, drop_last=val_drop_last)

    def train(self):
        with mlflow.start_run(run_name="train") as run:
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("learning_rate", self.lr)

            for epoch in range(self.epochs):
                total_loss = self._train_one_epoch(epoch)
                mlflow.log_metric("train_loss", total_loss, step=epoch)

                if (epoch + 1) % model_config.SAVE_FREQ == 0:
                    checkpoint_path = os.path.join(
                        self.save_dir,
                        f"resnet_epoch_{epoch+1}.pkl")

                    print(f'saving to {checkpoint_path}')
                    torch.save(self.model.state_dict(), checkpoint_path)
                    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

            mlflow.pytorch.log_model(self.model, artifact_path="resnet_model")
            result = mlflow.register_model(
                f"runs:/{run.info.run_id}/resnet_model",
                "CaptchaResNet"
            )              
                        

    def _train_one_epoch(self, epoch):
        self.model.train()
        batches = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{self.epochs}")
        total_loss = 0.0
        
        for images, labels, _ in batches:
            with torch.autograd.set_detect_anomaly(True):
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                batches.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)
    
    def _load_model(self, checkpoint_path=None, model_uri=None):
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.to(self.device)
        elif model_uri:
            self.model = mlflow.pytorch.load_model(model_uri).to(self.device)
        else:
            raise ValueError("Please provide either checkpoint_path or model_uri.")

    def evaluate(self, checkpoint_path=None, model_uri=None):
        with mlflow.start_run(run_name="evaluate"):
            self._load_model(checkpoint_path, model_uri)
            self.model.eval()

            correct = 0
            total   = 0
            batches = tqdm(self.val_loader, desc="Evaluate", leave=False)

            with torch.no_grad():
                for images, labels, fnames in batches:
                    images = images.to(self.device)
                    preds  = self.model(images).cpu().numpy()
                    true   = data_encoding.decode(labels.numpy()[0])

                    # slicing + argmax to rebuild pred_str
                    ALL = data_config.ALL_CHAR_SET_LEN
                    chars = []
                    
                    for i in range(len(true)):
                        start = i*ALL
                        end = (i+1)*ALL
                        idx = np.argmax(preds[0, start:end])
                        chars.append(data_config.INDEX_TO_CAPTCHA_DICT[idx])
                    
                    pred_str = ''.join(chars)

                    if pred_str == true:
                        correct += 1
                    total += 1

                    batches.set_description(f"[T:{true}|P:{pred_str}]")
                    tqdm.write(f"{fnames[0]} → True:{true}, Pred:{pred_str}")

            acc = correct / total if total > 0 else 0
            mlflow.log_metric("eval_accuracy", acc)
        
        return acc

    def _parse_submission(self, sample_submission_path):
        def assign_label_len(x):
            if x.startswith('task1'):
                return 1
            elif x.startswith('task2'):
                return 2
            else:
                return 4
        df = pd.read_csv(sample_submission_path)
        df['label'] = df['filename'].apply(assign_label_len)
        return df

    def predict(self, sample_submission_path, submission_path, checkpoint_path=None):
        with mlflow.start_run(run_name="predict"):
            self._load_model(checkpoint_path, model_uri)

            self.model.eval()

            df_test = self._parse_submission(sample_submission_path)
            test_ds = CaptchaDataset(data_config.ROOT_PATH, df_test, is_predict=True)
            test_dl = DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                drop_last=False
            )

            preds = []
            batches = tqdm(test_dl, total=len(test_dl), desc="Predict", leave=False)
            with torch.no_grad():
                for images, labels, fnames in batches:
                    images = images.to(self.device)
                    pred_probs = self.model(images).cpu().numpy()
                    length = int(labels.numpy()[0])

                    ALL = data_config.ALL_CHAR_SET_LEN
                    chars = []
                    for i in range(length):
                        start, end = i*ALL, (i+1)*ALL
                        idx = np.argmax(pred_probs[0, start:end])
                        chars.append(data_config.INDEX_TO_CAPTCHA_DICT[idx])
                    pred_str = ''.join(chars)
                    preds.append(pred_str)

                    batches.set_description(f"[len:{length}|pred:{pred_str}]")
                    batches.set_postfix(file=fnames)

            df_test['label'] = preds
            df_test.to_csv(submission_path, index=False)
            mlflow.log_artifact(submission_path)
            print(f"Submission saved to {submission_path}")