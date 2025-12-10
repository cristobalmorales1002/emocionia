import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURACIÓN DE RUTAS ---
# Ajusta esto si tu estructura de carpetas es diferente
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'emotions.json')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_svm.pkl')

# Aseguramos que la carpeta 'models' exista
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def entrenar_modelo():
    print("🚀 Cargando dataset...")
    try:
        # Leemos el JSON generado anteriormente
        df = pd.read_json(DATA_PATH)
        print(f"✅ Dataset cargado: {len(df)} registros.")
    except ValueError:
        print("❌ Error: No se pudo leer el JSON. Verifica que 'emotions.json' exista y tenga formato válido.")
        return
    except Exception as e:
        print(f"❌ Error inesperado al cargar datos: {e}")
        return

    # --- DEFINICIÓN DEL ALGORITMO ---

    # 1. Vectorización (Convertir texto a números)
    # - max_features=20000: Aumentado para entender más palabras.
    # - stop_words='english': Ignora palabras vacías (the, is, in) que meten ruido.
    # - ngram_range=(1,2): Entiende palabras sueltas ("happy") y pares ("not happy").
    tfidf = TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english',
        strip_accents='unicode'
    )

    # 2. Modelo SVM Base con Balanceo
    # - class_weight='balanced': OBLIGATORIO para que las emociones raras no sean ignoradas.
    svm_base = LinearSVC(dual=False, random_state=42, class_weight='balanced')

    # 3. Calibración de Probabilidades
    # - method='isotonic': Generalmente da mejores probabilidades reales con datasets grandes (>10k).
    # - Esto permite que la interfaz muestre "85% Alegría" en lugar de solo "Alegría".
    clf_calibrado = CalibratedClassifierCV(svm_base, method='isotonic', cv=3)

    # 4. Pipeline Completo
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf_calibrado)
    ])

    # --- ENTRENAMIENTO ---

    print("✂️ Separando datos de entrenamiento y prueba...")
    # Usamos 'stratify' para asegurar que train y test tengan la misma proporción de emociones
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['label'], # Asegúrate de que esta columna tenga el nombre de la emoción (ej: "fear", "joy")
        test_size=0.2, 
        random_state=42,
        stratify=df['label'] 
    )

    print("🧠 Entrenando modelo (esto puede tardar unos minutos)...")
    pipeline.fit(X_train, y_train)

    # --- EVALUACIÓN ---
    
    print("\n📊 Evaluación del modelo:")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Exactitud (Accuracy): {acc:.4f}")
    print("\nReporte detallado por emoción:")
    print(classification_report(y_test, y_pred))

    # --- GUARDADO ---
    
    print(f"💾 Guardando modelo en: {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)
    print("✅ ¡Entrenamiento finalizado con éxito!")

if __name__ == "__main__":
    entrenar_modelo()