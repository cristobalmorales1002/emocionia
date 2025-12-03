import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


# configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Sube a /src
DATA_PATH = os.path.join(BASE_DIR, 'data', 'emotions.json')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_svm.pkl')

def entrenar_modelo():
    print("🚀 Iniciando proceso de entrenamiento...")

    # cargamos los datos
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el dataset en {DATA_PATH}")
        return

    print(f"📂 Cargando datos desde: {DATA_PATH}")
    df = pd.read_json(DATA_PATH)
    
    # convertir texto a vectores numéricos 
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,      # lo Convertimos a minúsculas
            max_features=5000,   # Lilimitamos el 
            ngram_range=(1, 2)   # se usan palabras sueltas y pares de palabras bigramas
        )),
        ('clf', LinearSVC(dual="auto", random_state=42))
    ])

    # dividimos los datos para entrenamiento y prueba
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  entrenamiento
    print("🧠 Entrenando SVM...")
    pipeline.fit(X_train, y_train)

    # evaluación Preliminar
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Precisión (Accuracy): {acc:.4f}")
    print("📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # guardar modelo
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo guardado exitosamente en: {MODEL_PATH}")

if __name__ == "__main__":
    entrenar_modelo()