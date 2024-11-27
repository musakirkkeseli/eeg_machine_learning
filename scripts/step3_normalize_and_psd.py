# step3_normalize_and_psd.py
import os
import sys

# config.py'nin bulunduğu ana dizini modül arama yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FILTERED_EEG_FILE, NORMALIZED_EEG_FILE, STEP3_PSD_COMPARISON
import numpy as np
from scipy.stats import zscore
from scipy.signal import welch
import matplotlib.pyplot as plt

def plot_psd(signal, fs, label, linestyle=None):
    """
    Welch yöntemiyle PSD analizini yapar ve çizimi gerçekleştirir.
    """
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    plt.semilogy(f, Pxx, label=label, linestyle=linestyle)

def normalize_and_analyze(filtered_file, normalized_out, fs=128):
    """
    - Filtrelenmiş veriyi normalize eder.
    - Ham, filtrelenmiş ve normalize edilmiş veriler için PSD analizini gerçekleştirir.
    - Normalize edilmiş veriyi kaydeder.
    """
    # Filtrelenmiş veriyi yükle
    filtered_data = np.load(filtered_file)

    # Z-Normallizasyon
    print("Z-Normallizasyon işlemi başlıyor...")
    normalized_data = zscore(filtered_data, axis=1)
    np.save(normalized_out, normalized_data)
    print(f"Normalize edilmiş veri kaydedildi: {normalized_out}")

    # PSD Analizi
    print("PSD analizi yapılıyor...")
    plt.figure(figsize=(14, 8))
    # Filtrelenmiş veri
    plot_psd(filtered_data[0], fs, label="Filtrelenmiş Sinyal", linestyle="--")
    # Normalize edilmiş veri
    plot_psd(normalized_data[0], fs, label="Normalize Edilmiş Sinyal")
    plt.title("PSD Karşılaştırması - Filtrelenmiş ve Normalize Edilmiş")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Güç Yoğunluğu")
    plt.legend()
    plt.grid()
    plt.savefig(STEP3_PSD_COMPARISON)
    plt.show()
    print("PSD grafiği oluşturuldu: step3_psd_comparison.png")

if __name__ == "__main__":
    # Girdi ve çıktı dosyaları
    filtered_file = FILTERED_EEG_FILE # step2 tarafından üretilmiş dosya
    normalized_out = NORMALIZED_EEG_FILE 

    # Örnekleme frekansı
    fs = 128

    # Normalizasyon ve PSD analizi
    normalize_and_analyze(filtered_file, normalized_out, fs)
