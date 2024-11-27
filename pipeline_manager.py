import os
import subprocess

# Her adım ve bu adımın gerektirdiği giriş/çıkış dosyaları
steps = [
    {
        "name": "Step 1 - Extract and Save",
        "script": "step1_extract_and_save.py",
        "outputs": ["DataEEG.npy", "DataLoco.npy"]
    },
    {
        "name": "Step 2 - Filter and PSD",
        "script": "step2_filter_and_psd.py",
        "inputs": ["DataEEG.npy"],
        "outputs": ["Filtered_DataEEG.npy", "step2_psd_comparison.png"]
    },
    {
        "name": "Step 3 - Normalize and PSD",
        "script": "step3_normalize_and_psd.py",
        "inputs": ["Filtered_DataEEG.npy"],
        "outputs": ["Normalized_Filtered_DataEEG.npy", "step3_psd_comparison.png"]
    },
    {
        "name": "Step 4 - Feature Selection",
        "script": "step4_feature_selection.py",
        "inputs": ["Normalized_Filtered_DataEEG.npy", "DataLoco.npy"],
        "outputs": ["Selected_Features.npy", "step4_feature_scores.png"]
    },
    {
        "name": "Step 5 - Model Training",
        "script": "step5_model_training.py",
        "inputs": ["Selected_Features.npy", "DataLoco.npy"],
        "outputs": []
    }
]

def check_files_exist(files):
    """
    Dosyaların mevcut olup olmadığını kontrol eder.
    """
    return all(os.path.exists(file) for file in files)

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
        if "inputs" in step:
            if not check_files_exist(step["inputs"]):
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
