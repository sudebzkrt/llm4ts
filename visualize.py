# visualize.py / plot_results.py — LLM4TS sonuç görselleştirme (tarih hizalı, CSV+residual)
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ----------------- Argümanlar -----------------
parser = argparse.ArgumentParser(description='LLM4TS Sonuç Görselleştirme')
parser.add_argument('--folder', type=str, required=True, help='pred.npy ve true.npy klasörü')
parser.add_argument('--data_path', type=str, default='dataset/CO1/co1_comdty.csv')
parser.add_argument('--target', type=str, default='co1_comdty')
parser.add_argument('--plot_denormalized', action='store_true')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--seq_len', type=int, default=96)
args = parser.parse_args()

results_folder_path = args.folder
print(f"'{results_folder_path}' klasöründeki sonuçlar okunuyor...")

try:
    # 1) Çıktıları yükle
    preds_norm = np.load(os.path.join(results_folder_path, 'pred.npy'))
    trues_norm = np.load(os.path.join(results_folder_path, 'true.npy'))
    print(f"pred shape: {preds_norm.shape}")
    print(f"true shape: {trues_norm.shape}")

    # 2) Şekil düzleştirme
    ns, pl, nf = preds_norm.shape
    if pl == 1:
        plot_preds_norm = preds_norm.reshape(-1)
        plot_trues_norm = trues_norm.reshape(-1)
    else:
        plot_preds_norm = preds_norm[0, :, 0]
        plot_trues_norm = trues_norm[0, :, 0]

    # 3) Normalize başlangıç
    plot_preds = plot_preds_norm.copy()
    plot_trues = plot_trues_norm.copy()
    y_label = 'Normalize Edilmiş Değer'
    title = 'Model Tahmini vs Gerçek (Normalize)'

    # 4) Denormalize
    scaler = None
    df_raw = pd.read_csv(args.data_path, engine='python')
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    if args.target not in df_raw.columns:
        raise KeyError(f"'{args.target}' CSV'de yok. Sütunlar: {list(df_raw.columns)}")

    if args.plot_denormalized:
        print("Denormalizasyon başlıyor...")
        df_data = df_raw[[args.target]]
        n_total = len(df_data)
        n_train = int(n_total * args.train_ratio)
        train_data = df_data.iloc[:n_train]
        scaler = StandardScaler().fit(train_data.values)

        plot_preds = scaler.inverse_transform(plot_preds_norm.reshape(-1, 1)).flatten()
        plot_trues = scaler.inverse_transform(plot_trues_norm.reshape(-1, 1)).flatten()

        y_label = 'Gerçek Fiyat'
        title = 'Model Tahmini vs Gerçek (Gerçek Fiyatlar)'
        print("Inverse transform tamam.")

    # 5) Limit
    if args.limit:
        n = min(args.limit, len(plot_preds), len(plot_trues))
        plot_preds = plot_preds[:n]
        plot_trues = plot_trues[:n]
        title += f" (İlk {n} Gün)"

    # 6) METRİKLER
    print("\n" + "="*50)
    print(">>> MODEL PERFORMANS METRİKLERİ")
    print("="*50)

    norm_mae = mean_absolute_error(plot_trues_norm[:len(plot_preds)], plot_preds_norm[:len(plot_preds)])
    norm_mse = mean_squared_error(plot_trues_norm[:len(plot_preds)], plot_preds_norm[:len(plot_preds)])
    print(f"Normalize MAE: {norm_mae:.6f}")
    print(f"Normalize MSE: {norm_mse:.6f}")

    if args.plot_denormalized and scaler is not None:
        mae = mean_absolute_error(plot_trues, plot_preds)
        r2  = r2_score(plot_trues, plot_preds)
        print("\n--- Gerçek Değerler ---")
        print(f"Gerçek MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
    else:
        mae = norm_mae
        r2 = 1 - (norm_mse / np.var(plot_trues_norm[:len(plot_preds)]) if np.var(plot_trues_norm[:len(plot_preds)])>0 else np.nan)

    print("="*50 + "\n")

    # ============================
    # === 6.B METRİK LOG KAYIT ===
    # ============================
    metrics_row = {
        "run_folder": results_folder_path,
        "seq_len": args.seq_len,
        "pred_len": pl,
        "normalize_mae": float(norm_mae),
        "normalize_mse": float(norm_mse),
        "denormalize_mae": float(mae),
        "r2": float(r2),
    }

    metrics_path = "tuning_results.csv"
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        new = pd.concat([old, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        new = pd.DataFrame([metrics_row])

    new.to_csv(metrics_path, index=False)
    print(f"📌 tuning_results.csv güncellendi. Toplam deney: {len(new)}")
    # ============================

    # 7) Tarih hizalaması
    date_col = None
    for c in df_raw.columns:
        if 'date' in c or 'tarih' in c or c in ('ds','time','timestamp'):
            date_col = c
            break
    if date_col is None:
        date_col = df_raw.columns[0]

    dates = pd.to_datetime(df_raw[date_col], errors='coerce')
    N = len(df_raw)
    n_train = int(N * args.train_ratio)
    n_val   = int(N * args.val_ratio)
    start_idx = n_train + n_val + args.seq_len
    end_idx = min(start_idx + len(plot_trues), N)
    dates_for_plot = dates.iloc[start_idx:end_idx].reset_index(drop=True)

    # 8) CSV kayıt
    out = pd.DataFrame({
        'Tarih': dates_for_plot,
        'Gercek_Degerler': plot_trues[:len(dates_for_plot)],
        'Model_Tahminleri': plot_preds[:len(dates_for_plot)],
    })
    out.to_csv("comparison_results.csv", index=False)
    print("comparison_results.csv kaydedildi.")

    # 9) Grafik
    plt.figure(figsize=(12, 6))
    plt.plot(out['Gercek_Degerler'].values, label='Gerçek', color='blue')
    plt.plot(out['Model_Tahminleri'].values, label='Tahmin', color='orange', linestyle='--')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Zaman Adımları")
    plt.text(0.02, 0.95, f"MAE: {mae:.3f}\nR²: {r2:.3f}",
             transform=plt.gca().transAxes, va="top",
             bbox=dict(facecolor="white", alpha=0.8))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("forecast_plot_fixed.png", dpi=300)
    print("forecast_plot_fixed.png kaydedildi.")

    # 10) Residual
    res = out['Gercek_Degerler'].values - out['Model_Tahminleri'].values
    plt.figure(figsize=(12,4))
    plt.plot(res)
    plt.title("Residuals (Gerçek - Tahmin)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("forecast_residuals.png", dpi=300)
    print("forecast_residuals.png kaydedildi.")

except FileNotFoundError as e:
    print("\n!!!!!!!! DOSYA HATASI !!!!!!!!")
    print(e)
except Exception as e:
    print("\n!!!! BEKLENMEDİK HATA !!!!")
    print(e)
