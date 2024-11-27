# step1_extract_and_save.py
import numpy as np
import h5py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MAT_FILE, DATA_EEG_FILE, DATA_LOCO_FILE


def load_and_save_data(mat_file, dataset_path, output_file, reshape_dims=None, replicate=1):
    """
    H5 dosyasından belirtilen veri kümesini yükler, işler ve bir numpy dosyasına kaydeder.
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
    # Data0 işlemi
    load_and_save_data(MAT_FILE, 'Loco_Data_EEG/data0', DATA_EEG_FILE, reshape_dims=(32, 15360), replicate=1)

    # Loco işlemi
    load_and_save_data(MAT_FILE, 'Loco_Data_EEG/loco', DATA_LOCO_FILE, replicate=32)
