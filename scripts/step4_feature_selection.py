# step4_feature_selection.py
import os
import sys

# config.py'nin bulunduğu ana dizini modül arama yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NORMALIZED_EEG_FILE, DATA_LOCO_FILE, SELECTED_FEATURES_FILE, STEP4_FEATURE_SCORES
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

def select_features(data_file, labels_file, selected_out, k=10):
    """
    Özellik seçimi gerçekleştirir ve seçilen özellikleri kaydeder.
    - data_file: Normalize edilmiş veri dosyası (npy formatında)
    - labels_file: Hedef sınıfları içeren dosya (npy formatında)
    - selected_out: Seçilen özelliklerin kaydedileceği dosya
    - k: Seçilecek en iyi K özellik
    """
    # Veriyi ve etiketleri yükle
    print("Veri ve etiketler yükleniyor...")
    data = np.load(data_file)
    labels = np.load(labels_file)

    # Özellik seçimi
    print(f"Özellik seçimi gerçekleştiriliyor... En iyi {k} özellik seçiliyor.")
    selector = SelectKBest(score_func=f_classif, k=k)
    selected_data = selector.fit_transform(data.reshape(data.shape[0], -1), labels)

    # Skorları görselleştirme
    scores = selector.scores_
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(scores)), scores, alpha=0.7, color='b', label='Özellik Skorları')
    plt.title(f"Özellik Skorları (En iyi {k} seçildi)")
    plt.xlabel("Özellikler")
    plt.ylabel("F-Skor")
    plt.legend()
    plt.grid()
    plt.savefig(STEP4_FEATURE_SCORES)
    plt.show()
    print("Özellik skorları grafiği kaydedildi: step4_feature_scores.png")

    # Seçilen özellikleri kaydet
    np.save(selected_out, selected_data)
    print(f"Seçilen özellikler kaydedildi: {selected_out}")

    # Seçilen özelliklerin boyutunu yazdır
    print(f"Seçilen özelliklerin boyutu: {selected_data.shape}")

if __name__ == "__main__":
    # Girdi dosyaları
    data_file = NORMALIZED_EEG_FILE  # step3'ten gelen dosya
    labels_file = DATA_LOCO_FILE  # Hedef değişken

    # Çıktı dosyası
    selected_out = SELECTED_FEATURES_FILE

    # Özellik sayısı
    k = 10

    # Özellik seçimi ve görselleştirme
    select_features(data_file, labels_file, selected_out, k=k)
