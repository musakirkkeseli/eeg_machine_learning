# step3_1_minmax_scaler.py
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NORMALIZED_EEG_FILE, NORMALIZED_MINMAX_EEG_FILE


def apply_minmax_scaler(normalized_file, output_file):
    """
    - Normalize edilmiş veriyi MinMaxScaler ile 0-1 aralığına sıkıştırır.
    - Yeni veriyi kaydeder.
    """
    # Normalize edilmiş veriyi yükle
    normalized_data = np.load(normalized_file)

    # MinMaxScaler uygulama
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(normalized_data.T).T  # Transpose işlemi ile doğru eksende uygula

    # MinMaxScaler ile işlenmiş veriyi kaydet
    np.save(output_file, scaled_data)
    print(f"MinMaxScaler ile normalize edilmiş veri kaydedildi: {output_file}")

if __name__ == "__main__":
    apply_minmax_scaler(NORMALIZED_EEG_FILE, NORMALIZED_MINMAX_EEG_FILE)
