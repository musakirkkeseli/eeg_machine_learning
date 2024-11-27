# step2_filter_and_psd.py
import numpy as np
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt

# Butterworth filtresi tanımı
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Band-pass filtresini veriye uygular.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

def plot_psd(signal, fs, label, linestyle=None):
    """
    PSD analizini yapar ve çizimi gerçekleştirir.
    """
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    plt.semilogy(f, Pxx, label=label, linestyle=linestyle)

def filter_and_analyze(data_file, filtered_out, fs=128):
    """
    - Ham veri üzerinde filtreleme işlemi yapar.
    - Ham ve filtrelenmiş veriler için PSD analizini gerçekleştirir.
    - Filtrelenmiş veriyi kaydeder.
    """
    # Ham veriyi yükle
    raw_data = np.load(data_file)

    # Filtreleme
    print("Filtreleme işlemi başlıyor...")
    filtered_data = np.array([bandpass_filter(channel, lowcut=1, highcut=50, fs=fs) for channel in raw_data])
    np.save(filtered_out, filtered_data)
    print(f"Filtrelenmiş veri kaydedildi: {filtered_out}")

    # PSD Analizi
    print("PSD analizi yapılıyor...")
    plt.figure(figsize=(14, 8))
    plot_psd(raw_data[0], fs, label="Ham Sinyal", linestyle="--")
    plot_psd(filtered_data[0], fs, label="Filtrelenmiş Sinyal")
    plt.title("PSD Karşılaştırması - Ham ve Filtrelenmiş")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Güç Yoğunluğu")
    plt.legend()
    plt.grid()
    plt.savefig("step2_psd_comparison.png")
    plt.show()
    print("PSD grafiği oluşturuldu: step2_psd_comparison.png")

if __name__ == "__main__":
    # Girdi ve çıktı dosyaları
    data_file = 'DataEEG.npy'  # step1 tarafından üretilmiş dosya
    filtered_out = 'Filtered_DataEEG.npy'

    # Örnekleme frekansı
    fs = 128

    # Filtreleme ve PSD analizi
    filter_and_analyze(data_file, filtered_out, fs)
