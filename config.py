import os

# Proje ana dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data input/output dizinleri
DATA_INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# Script dizini
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts")

# Data dosyaları
MAT_FILE = os.path.join(DATA_INPUT_DIR, "Loco_Data_EEG32.mat")
DATA_EEG_FILE = os.path.join(DATA_OUTPUT_DIR, "DataEEG.npy")
DATA_LOCO_FILE = os.path.join(DATA_OUTPUT_DIR, "DataLoco.npy")
FILTERED_EEG_FILE = os.path.join(DATA_OUTPUT_DIR, "Filtered_DataEEG.npy")
NORMALIZED_EEG_FILE = os.path.join(DATA_OUTPUT_DIR, "Normalized_Filtered_DataEEG.npy")
NORMALIZED_MINMAX_EEG_FILE = os.path.join(DATA_OUTPUT_DIR, "Normalized_MinMax_Filtered_DataEEG.npy")
SELECTED_FEATURES_FILE = os.path.join(DATA_OUTPUT_DIR, "Selected_Features.npy")

# Çıktı görselleri
STEP2_PSD_COMPARISON = os.path.join(DATA_OUTPUT_DIR, "step2_psd_comparison.png")
STEP3_PSD_COMPARISON = os.path.join(DATA_OUTPUT_DIR, "step3_psd_comparison.png")
STEP4_FEATURE_SCORES = os.path.join(DATA_OUTPUT_DIR, "step4_feature_scores.png")
KNN_Confusion_Matrix = os.path.join(DATA_OUTPUT_DIR, "knn_confusion_matrix.png")
SVM_Confusion_Matrix = os.path.join(DATA_OUTPUT_DIR, "svm_confusion_matrix.png")

# Script dosyaları
STEP1_SCRIPT = os.path.join(SCRIPT_DIR, "step1_extract_and_save.py")
STEP2_SCRIPT = os.path.join(SCRIPT_DIR, "step2_filter_and_psd.py")
STEP3_SCRIPT = os.path.join(SCRIPT_DIR, "step3_normalize_and_psd.py")
STEP3_1_SCRIPT = os.path.join(SCRIPT_DIR, "step3_1_minmax_scaler.py")
STEP4_SCRIPT = os.path.join(SCRIPT_DIR, "step4_feature_selection.py")
STEP5_SCRIPT = os.path.join(SCRIPT_DIR, "step5_model_training.py")
