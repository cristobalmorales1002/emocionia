import joblib
import os
import numpy as np

class EmotionPredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, '..', 'models', 'emotion_svm.pkl')
        self.model = None
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
        Retorna la emoción ganadora Y un diccionario con los porcentajes reales.
        """
        if not self.model:
            return "Error: Modelo no cargado", {}

        try:
            # 1. Obtener probabilidades reales (Matriz de [1, n_clases])
            probs = self.model.predict_proba([texto])[0]
            classes = self.model.classes_

            # 2. Crear diccionario { "alegria": 0.85, "tristeza": 0.02 ... }
            resultado_dict = {cls: round(float(prob), 2) for cls, prob in zip(classes, probs)}
            
            # 3. Obtener la ganadora
            emocion_ganadora = max(resultado_dict, key=resultado_dict.get)
            confianza = resultado_dict[emocion_ganadora]
            
            # Formatear salida bonita
            texto_salida = f"{emocion_ganadora.upper()} ({int(confianza*100)}%)"
            
            return texto_salida, resultado_dict

        except Exception as e:
            return f"Error: {str(e)}", {}