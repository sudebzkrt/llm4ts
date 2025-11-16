# data_custom.py  — 80/10/10 split + train-only scaler + sade multivariate
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_CustomStock(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='stock_data.csv',
                 target='Close', scale=True, timeenc=0, freq='h',
                 percent=100, return_single_feature=False):

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features      # 'M' => multi-variate input, single target
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.return_single_feature = return_single_feature

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.C = self.data_x.shape[1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        full_path = os.path.join(self.root_path, 'dataset', self.data_path)
        df_raw = pd.read_csv(full_path)

        # sütun isimlerini normalize et
        df_raw.columns = [x.strip().lower() for x in df_raw.columns]
        target_col = self.target.lower()

        # zaman sütunu bul
        time_col = None
        for c in ['timestamp', 'date', 'time']:
            if c in df_raw.columns:
                time_col = c
                break
        if time_col is None:
            raise ValueError("CSV’de tarih/timestamp/time sütunu bulunamadı.")

        if target_col not in df_raw.columns:
            raise ValueError(f"Hedef sütun '{self.target}' bulunamadı. Mevcut: {df_raw.columns.tolist()}")

        # çoklu girdi, tek hedef olacak şekilde yeniden sırala
        cols = list(df_raw.columns)
        cols.remove(time_col)
        cols.remove(target_col)
        df_raw = df_raw[[time_col, target_col] + cols]
        df_raw[time_col] = pd.to_datetime(df_raw[time_col])

        # -------- 80 / 10 / 10 bölme  --------
        num_train = int(len(df_raw) * 0.8)
        num_vali  = int(len(df_raw) * 0.1)
        num_test  = len(df_raw) - num_train - num_vali

        border1s = [0,
                    num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train,
                    num_train + num_vali,
                    len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        # -------- Veri seçimi --------
        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]     # target + diğer feature’lar
        elif self.features == 'S':
            df_data = df_raw[[target_col]]
        else:
            raise ValueError("features parametresi 'M', 'MS' veya 'S' olmalı.")

        # -------- Ölçekleme (sadece train’e fit) --------
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # -------- Zaman özellikleri --------
        df_stamp = df_raw[[time_col]][border1:border2]
        df_stamp[time_col] = pd.to_datetime(df_stamp[time_col].values)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[time_col].dt.month
            df_stamp['day'] = df_stamp[time_col].dt.day
            df_stamp['weekday'] = df_stamp[time_col].dt.weekday
            df_stamp['hour'] = df_stamp[time_col].dt.hour
            data_stamp = df_stamp.drop(columns=[time_col]).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp[time_col].values), freq=self.freq).T

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, idx):
        if not self.return_single_feature:
            s = idx
            e = s + self.seq_len
            r_begin = e - self.label_len
            r_end   = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s:e]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s:e]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        else:
            original = idx // self.C
            ch = idx % self.C
            s = original
            e = s + self.seq_len
            r_begin = e - self.label_len
            r_end   = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s:e, ch:ch+1]
            seq_y = self.data_y[r_begin:r_end, ch:ch+1]
            seq_x_mark = self.data_stamp[s:e]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        base_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return base_len * self.C if self.return_single_feature else base_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
