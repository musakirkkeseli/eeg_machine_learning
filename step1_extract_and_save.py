# step1_extract_and_save.py
import numpy as np
import h5py

def load_and_save_data(mat_file, dataset_path, output_file, reshape_dims=None, replicate=1):
    """
    H5 dosyasından belirtilen veri kümesini yükler, işler ve bir numpy dosyasına kaydeder.

    Args:
    - mat_file (str): .mat dosyasının yolu
    - dataset_path (str): H5 dosyası içindeki veri kümesinin yolu
    - output_file (str): Çıktı dosyasının ismi
    - reshape_dims (tuple): Verilerin yeniden şekillendirileceği boyutlar (ör. (32, 15360))
    - replicate (int): Her bir satırı kaç kez çoğaltılacağı (ör. loco için 32)
    """
    with h5py.File(mat_file, 'r') as f:
        dataset = f.get(dataset_path)

        if dataset is None:
            print(f"{dataset_path} not found in the file.")
            return

        data_list = []
        for i in range(len(dataset)):
            data_line = f[dataset[i][0]]
            data_array = np.array(data_line)
            if reshape_dims:
                data_array = data_array.reshape(reshape_dims)
            for _ in range(replicate):
                data_list.append(data_array)

        final_data = np.concatenate(data_list, axis=0) if len(data_list) > 1 else data_list[0]
        np.save(output_file, final_data)
        print(f"{dataset_path} saved to {output_file}. Shape: {final_data.shape}")

if __name__ == "__main__":
    # Dosya yolları ve parametreler
    mat_file = 'Loco_Data_EEG32.mat'

    # Data0 işlemi
    data0_path = 'Loco_Data_EEG/data0'
    data0_out = 'DataEEG.npy'
    load_and_save_data(mat_file, data0_path, data0_out, reshape_dims=(32, 15360), replicate=1)

    # Loco işlemi
    loco_path = 'Loco_Data_EEG/loco'
    loco_out = 'DataLoco.npy'
    load_and_save_data(mat_file, loco_path, loco_out, replicate=32)
