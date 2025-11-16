# exp/exp_long_term_forecasting.py (SONUÇLARI KAYDEDEN NİHAİ HALİ)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, print_params
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import shutil

warnings.filterwarnings("ignore")

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if getattr(self, "model", None) is not None:
            raise ValueError("Model already exists!")
        train_data, train_loader = self._get_data(flag="train")
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
        self.args.C_t = batch_x_mark.shape[2]
        model = self.model_dict[self.args.model].Model(self.args).float()
        print_params(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = getattr(optim, self.args.dft_optim)(
            self.model.parameters(),
            lr=self.args.dft_learning_rate,
            weight_decay=self.args.dft_weight_decay,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, use_tqdm=False):
        print(f">>>>> start training (long-term forecasting: {self.args.pred_len}) : {self.args.setting}>>>>>")
        
        if self.args.enable_supervised_finetuning:
            sft_setting_name = "sft_" + self.args.setting
            checkpoint_path = f"./checkpoints/{sft_setting_name}/checkpoint.pth"
            print(f"🔍 Loading checkpoint from: {checkpoint_path}")
            if not os.path.exists(checkpoint_path):
                 raise FileNotFoundError(f"❌ SFT Checkpoint not found in: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            keys_related_to_output_layer = [k for k in checkpoint.keys() if "output_layer" in k]
            for key in keys_related_to_output_layer: del checkpoint[key]
            self.model.load_state_dict(checkpoint, strict=False)
            print("✅ Successfully loaded the model trained with SFT.")

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(path): os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        best_test_loss, best_test_mae, best_epoch = (np.inf, np.inf, 0)
        
        for epoch in range(self.args.dft_train_epochs):
            self.model.train()
            if self.args.ft_mode == "lp_ft" and epoch == self.args.dft_train_epochs // 2:
                self.model.linear_probe_to_fine_tuning()
                print_params(self.model)

                print_params(self.model)

            # === HYBRID GPU + CPU FIX (MPS LoRA Freeze Patch) ===
            # Eğer MPS (Apple GPU) kullanıyorsak ve model fine-tuning aşamasına geçtiyse,
            # LoRA FT aşamasını CPU'da çalıştır (MPS freeze bug'ını engeller).
            if self.args.device == "mps":
                print("⚠️ MPS tespit edildi: Fine-tuning CPU'da yapılacak (freeze çözümü).")
                self.device = "cpu"
                self.model.to("cpu")
            # =====================================================
            
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs, batch_y = outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()

            train_loss = np.average(train_loss)
            vali_loss, vali_mae = self.get_metrics(vali_loader)
            test_loss, test_mae = self.get_metrics(test_loader)
            
            if test_loss < best_test_loss:
                best_test_loss, best_test_mae, best_epoch = test_loss, test_mae, epoch + 1

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args.dft_learning_rate, self.args.dft_lradj)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))
        
        # ===============================================================================================
        #                             <<< SONUÇLARI KAYDETME KODU >>>
        # ===============================================================================================
        print("\n>>> EĞİTİM BİTTİ, FİNAL TEST SONUÇLARI KAYDEDİLİYOR...")
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs, batch_y = outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:]
                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
        
        # Sonuçları 'checkpoints' klasöründeki ilgili alt klasöre kaydediyoruz.
        np.save(os.path.join(path, 'pred.npy'), preds)
        np.save(os.path.join(path, 'true.npy'), trues)
        print(f">>> Sonuçlar '{path}' klasörüne 'pred.npy' ve 'true.npy' olarak kaydedildi.")
        # ===============================================================================================

        metrics = {"best_test_loss": best_test_loss, "best_test_mae": best_test_mae, "best_epoch": best_epoch}
        print("===============================\n", metrics, "\n===============================")
        return metrics

    def get_metrics(self, data_loader, use_tqdm=False):
        total_mse, total_mae, total_samples = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                batch_x, batch_y = batch_x.float().to(self.device), batch_y.float().to(self.device)
                batch_x_mark, batch_y_mark = batch_x_mark.float().to(self.device), batch_y_mark.float().to(self.device)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs, batch_y = outputs[:, -self.args.pred_len:, f_dim:], batch_y[:, -self.args.pred_len:, f_dim:]
                pred, true = outputs.detach(), batch_y.detach()
                total_mse += torch.mean((pred - true) ** 2).item() * len(batch_x)
                total_mae += torch.mean(torch.abs(pred - true)).item() * len(batch_x)
                total_samples += len(batch_x)
        return total_mse / total_samples, total_mae / total_samples
    
    # ======================================================================
#                          <<< TEST (İNFERENCE) MODU >>>
# ======================================================================
if __name__ == '__main__':
    import argparse
    import torch
    from utils.tools import set_seed
    print(">>> LLM4TS Inference Mode Başlatıldı (exp_long_term_forecasting)")

    parser = argparse.ArgumentParser(description='LLM4TS Inference')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=0)
    parser.add_argument('--data', type=str, default='custom_stock')
    parser.add_argument('--data_path', type=str, default='CO1/co1_comdty.csv')
    parser.add_argument('--target', type=str, default='co1_comdty')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--checkpoints', type=str, required=True)
    args = parser.parse_args()

    # Çekirdek ayarları
    set_seed(2023)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    args.device = device
    args.use_gpu = True if device in ['cuda', 'mps'] else False
    args.use_amp = False
    args.batch_size = 64  # veya senin kullandığın batch size neyse


    # Modeli yükle
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    print(f">>> Checkpoint yükleniyor: {args.checkpoints}/checkpoint.pth")
    exp = Exp_Long_Term_Forecast(args)
    checkpoint_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint bulunamadı: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    exp.model.load_state_dict(checkpoint, strict=False)
    exp.model.to(device)
    exp.model.eval()
    print("✅ Model başarıyla yüklendi, test başlıyor...")

    # Test datasını al
    test_data, test_loader = exp._get_data(flag="test")

    preds, trues = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            batch_x_mark, batch_y_mark = batch_x_mark.float().to(device), batch_y_mark.float().to(device)
            with torch.cuda.amp.autocast(enabled=getattr(args, "use_amp", False)):
                outputs = exp.model(batch_x, batch_x_mark, None, batch_y_mark)
            f_dim = -1 if args.features == "MS" else 0
            outputs, batch_y = outputs[:, -args.pred_len:, f_dim:], batch_y[:, -args.pred_len:, f_dim:]
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds, trues = np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)
    np.save(os.path.join(args.checkpoints, 'pred.npy'), preds)
    np.save(os.path.join(args.checkpoints, 'true.npy'), trues)
    print(f"✅ Test tamamlandı. Sonuçlar '{args.checkpoints}' klasörüne kaydedildi (pred.npy, true.npy).")
