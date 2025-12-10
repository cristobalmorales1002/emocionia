import joblib
import os
import numpy as np
from deep_translator import GoogleTranslator  # <--- Nueva importación

class EmotionPredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, '..', 'models', 'emotion_svm.pkl')
        self.model = None
        self.translator = GoogleTranslator(source='auto', target='en') # <--- Inicializamos el traductor
        self._cargar_modelo()

    def _cargar_modelo(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                print("⚠️ Modelo no encontrado. Ejecuta train.py primero.")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")

    def predecir(self, texto):
        """
        Traduce el texto de entrada a inglés, predice la emoción 
        y retorna la emoción ganadora Y un diccionario con los porcentajes reales.
        """
        if not self.model:
            return "Error: Modelo no cargado", {}

        try:
            # --- PASO EXTRA: TRADUCCIÓN ---
            # Traducimos de español (o auto) a inglés antes de predecir
            texto_traducido = self.translator.translate(texto)
            # (Opcional) Imprimimos para ver qué está recibiendo el modelo realmente
            # print(f"DEBUG - Input original: {texto} | Traducido: {texto_traducido}")
            
            # --- PREDICCIÓN (Usando el texto traducido) ---
            # 1. Obtener probabilidades reales
            probs = self.model.predict_proba([texto_traducido])[0]
            classes = self.model.classes_

            # 2. Crear diccionario { "joy": 0.85, "sadness": 0.02 ... }
            # Nota: Las claves 'classes' seguirán en inglés porque así se entrenó el modelo
            resultado_dict = {cls: round(float(prob), 2) for cls, prob in zip(classes, probs)}
            
            # 3. Obtener la ganadora
            emocion_ganadora = max(resultado_dict, key=resultado_dict.get)
            confianza = resultado_dict[emocion_ganadora]
            
            # Formatear salida bonita
            texto_salida = f"{emocion_ganadora.upper()} ({int(confianza*100)}%)"
            
            return texto_salida, resultado_dict

        except Exception as e:
            return f"Error en predicción/traducción: {str(e)}", {}