# step5_model_training.py
import os
import sys

# config.py'nin bulunduğu ana dizini modül arama yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SELECTED_FEATURES_FILE, DATA_LOCO_FILE
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(features_file, labels_file):
    """
    Model eğitimi ve değerlendirmesi için özellikleri ve etiketleri kullanır.
    """
    # Veriyi yükle
    print("Veriler yükleniyor...")
    features = np.load(features_file)
    labels = np.load(labels_file)

    # Eğitim ve test setlerine böl
    print("Veriler eğitim ve test setlerine ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Model seçimi ve GridSearch ile hiperparametre optimizasyonu
    print("Model eğitimi başlıyor...")

    # KNN modeli
    knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
    knn.fit(X_train, y_train)
    print(f"KNN En iyi parametreler: {knn.best_params_}")

    # SVM modeli
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    svm = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy')
    svm.fit(X_train, y_train)
    print(f"SVM En iyi parametreler: {svm.best_params_}")

    # Modellerin test setinde değerlendirilmesi
    print("Test seti üzerinde değerlendirme yapılıyor...")
    knn_pred = knn.best_estimator_.predict(X_test)
    svm_pred = svm.best_estimator_.predict(X_test)

    print("\nKNN Performansı:")
    evaluate_model(y_test, knn_pred)

    print("\nSVM Performansı:")
    evaluate_model(y_test, svm_pred)

def evaluate_model(y_true, y_pred):
    """
    Model performansını değerlendirir ve karışıklık matrisini çizdirir.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f"Doğruluk Skoru: {acc:.2f}")
    print(f"Kesinlik: {prec:.2f}")
    print(f"F1 Skoru: {f1:.2f}")
    print(f"Hassasiyet: {recall:.2f}")

    # Karışıklık matrisi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title("Karışıklık Matrisi")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.show()

if __name__ == "__main__":
    # Girdi dosyaları
    features_file = SELECTED_FEATURES_FILE  # step4'ten gelen dosya
    labels_file = DATA_LOCO_FILE  # Hedef değişken

    # Model eğitimi ve değerlendirmesi
    train_and_evaluate(features_file, labels_file)
