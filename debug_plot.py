import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

pred = np.load("./checkpoints/long_term_forecast_co1_comdty_1_LLM4TS_ftS_sl7_ll3_pl1_dm768_nh8_ebtimeF/pred.npy").flatten()
true = np.load("./checkpoints/long_term_forecast_co1_comdty_1_LLM4TS_ftS_sl7_ll3_pl1_dm768_nh8_ebtimeF/true.npy").flatten()

print("Pred shape:", pred.shape, "True shape:", true.shape)
print("Pred min/max:", np.min(pred), np.max(pred))
print("True min/max:", np.min(true), np.max(true))

plt.figure(figsize=(12, 6))
plt.plot(true, label='Gerçek', color='blue', linewidth=2)
plt.plot(pred, label='Tahmin', color='orange', linestyle='--', linewidth=2)
plt.legend()
plt.title("DEBUG TEST GRAFİĞİ", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("debug_plot.png", dpi=300)
print("✅ Grafik 'debug_plot.png' olarak kaydedildi!")
