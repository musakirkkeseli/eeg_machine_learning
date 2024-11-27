import os
import subprocess
import time
from config import *

# Her adım ve bu adımın gerektirdiği giriş/çıkış dosyaları
steps = [
    {
        "name": "Step 1 - Extract and Save",
        "script": STEP1_SCRIPT,
        "outputs": [DATA_EEG_FILE, DATA_LOCO_FILE],
    },
    {
        "name": "Step 2 - Filter and PSD",
        "script": STEP2_SCRIPT,
        "inputs": [DATA_EEG_FILE],
        "outputs": [FILTERED_EEG_FILE, STEP2_PSD_COMPARISON],
    },
    {
        "name": "Step 3 - Normalize and PSD",
        "script": STEP3_SCRIPT,
        "inputs": [FILTERED_EEG_FILE],
        "outputs": [NORMALIZED_EEG_FILE, STEP3_PSD_COMPARISON],
    },
    {
        "name": "Step 4 - Feature Selection",
        "script": STEP4_SCRIPT,
        "inputs": [NORMALIZED_EEG_FILE, DATA_LOCO_FILE],
        "outputs": [SELECTED_FEATURES_FILE, STEP4_FEATURE_SCORES],
    },
    {
        "name": "Step 5 - Model Training",
        "script": STEP5_SCRIPT,
        "inputs": [SELECTED_FEATURES_FILE, DATA_LOCO_FILE],
        "outputs": [],
    },
]

def check_files_exist(files, retries=3, delay=0.5):
    """
    Dosyaların mevcut olup olmadığını kontrol eder, belirli aralıklarla yeniden dener.
    - retries: Kaç kez kontrol edeceğini belirtir.
    - delay: Kontroller arasındaki bekleme süresi (saniye).
    """
    for _ in range(retries):
        if all(os.path.exists(file) for file in files):
            return True
        time.sleep(delay)
    return False

def run_script(script_name):
    """
    Belirtilen bir adımın Python betiğini çalıştırır.
    """
    print(f"Çalıştırılıyor: {script_name}")
    result = subprocess.run(["python3", script_name])
    if result.returncode == 0:
        print(f"{script_name} başarıyla çalıştırıldı.")
    else:
        print(f"{script_name} çalıştırılırken bir hata oluştu.")
        raise Exception(f"{script_name} başarısız oldu.")

def run_pipeline(start_step=None):
    """
    Pipeline'ı baştan sona veya belirli bir adımdan başlatır.
    - start_step: Başlamak istediğiniz adımın sıfırdan başlayarak indeks numarası
    """
    # Başlangıç adımı belirlenmemişse tüm adımları çalıştır
    if start_step is None:
        start_step = 0

    for step_index in range(start_step, len(steps)):
        step = steps[step_index]
        print(f"\n==> {step['name']} başlatılıyor...")

        # Önceki adımların çıktıları gerekli ise kontrol et
        if "inputs" in step and not check_files_exist(step["inputs"]):
            raise FileNotFoundError(
                f"Gerekli dosyalar eksik: {step['inputs']}. "
                f"Lütfen önceki adımları çalıştırın."
            )

        # Adımı çalıştır
        run_script(step["script"])

        # Bu adımın çıktıları kontrol edilir
        if "outputs" in step and step["outputs"]:
            if not check_files_exist(step["outputs"]):
                raise FileNotFoundError(
                    f"Çıktı dosyaları oluşturulamadı: {step['outputs']}. "
                    f"Lütfen {step['name']} adımını kontrol edin."
                )

if __name__ == "__main__":
    print("Tüm pipeline'ı çalıştırmak için: 'run_pipeline()' çağrısını kullanın.")
    print("Belirli bir adımdan başlatmak için: 'run_pipeline(start_step=<ADIM NUMARASI>)' çağrısını kullanın.")
    # Örnek: run_pipeline(start_step=2) -> Step 3'ten başlayarak çalıştırır.
    run_pipeline()  # Tüm adımları sırayla çalıştır
