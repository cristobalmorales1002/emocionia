import joblib
import os

class EmotionPredictor:
    def __init__(self):
        # Rutas relativas para encontrar el modelo .pkl
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.base_dir, 'models', 'emotion_svm.pkl')
        self.model = None
        self._cargar_modelo()

    def _cargar_modelo(self):
        """Carga el modelo serializado si existe."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ Modelo cargado desde {self.model_path}")
            else:
                print(f"Alerta: No se encontró el modelo en {self.model_path}. Ejecuta train.py primero.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

    def predecir(self, texto):
        """
        Recibe un texto y devuelve la emoción predicha y la confianza (opcional).
        """
        if not self.model:
            return "Error: Modelo no cargado"

        if not texto or not isinstance(texto, str):
            return "Error: Texto inválido"

        try:
            # La predicción devuelve un array, tomamos el primer elemento
            prediccion = self.model.predict([texto])[0]
            
            # Mapeo opcional para traducir al español si las etiquetas están en inglés
            traduccion = {
                'joy': 'Alegría ',
                'sadness': 'Tristeza ',
                'anger': 'Enojo ',
                'fear': 'Miedo ',
                'love': 'Amor ',
                'surprise': 'Sorpresa '
            }
            
            return traduccion.get(prediccion, prediccion)
            
        except Exception as e:
            return f"Error en predicción: {e}"