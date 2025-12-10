import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from predictor import EmotionPredictor
    print(" Importación exitosa: predictor.py encontrado localmente.")
except ImportError as e:
    try:
        from src.predictor import EmotionPredictor
        print("Importación exitosa: src.predictor encontrado.")
    except ImportError:
        print(f"❌ Error CRÍTICO: No se encuentra 'predictor.py'. \nDetalles: {e}")

        class EmotionPredictor:
            def predecir(self, t): return "Error: Backend no encontrado"


class AplicacionEmociones:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador Emocional - Avance 2")
        self.root.geometry("900x700")
        
        # Instanciamos el predictor del backend
        self.predictor = EmotionPredictor()
        
        # --- COLORES Y ESTILO ---
        self.bg_color = "#1E1E2E"       
        self.panel_color = "#282A36"    
        self.text_color = "#F8F8F2"     
        self.accent_color = "#6272A4"   
        self.success_color = "#50FA7B" 
        
        self.root.configure(bg=self.bg_color)
        self.crear_widgets()

    def crear_widgets(self):
        # titulo
        tk.Label(self.root, text="Clasificador Emocional IA", 
                 font=("Roboto", 22, "bold"), bg=self.bg_color, fg=self.text_color).pack(pady=20)

        # entrada
        frame_input = tk.Frame(self.root, bg=self.panel_color, padx=15, pady=15)
        frame_input.pack(fill="x", padx=30)

        tk.Label(frame_input, text="Escribe cómo te sientes:", 
                 bg=self.panel_color, fg=self.text_color).pack(anchor="w")

        self.txt_entrada = tk.Text(frame_input, height=4, bg="#44475A", 
                                   fg="white", insertbackground="white", 
                                   font=("Arial", 12), relief="flat")
        self.txt_entrada.pack(fill="x", pady=10)

        tk.Button(frame_input, text="Analizar Texto", command=self.realizar_analisis,
                  bg=self.accent_color, fg="white", font=("Arial", 12, "bold"), 
                  cursor="hand2", relief="flat").pack(fill="x", pady=5)

        # salida
        self.frame_resultados = tk.Frame(self.root, bg=self.bg_color)
        self.frame_resultados.pack(fill="both", expand=True, padx=30, pady=20)

        self.lbl_resultado = tk.Label(self.frame_resultados, text="Esperando análisis...",
                                      font=("Arial", 18, "bold"), bg=self.bg_color, fg="gray")
        self.lbl_resultado.pack(pady=10)

        self.frame_grafico = tk.Frame(self.frame_resultados, bg=self.bg_color)
        self.frame_grafico.pack(fill="both", expand=True)

    def realizar_analisis(self):
        texto = self.txt_entrada.get("1.0", "end-1c").strip()
        if not texto:
            messagebox.showwarning("Alerta", "Escribe algo primero.")
            return

        # AHORA EL PREDICTOR DEVUELVE 2 COSAS: TEXTO Y DICCIONARIO REAL
        resultado_str, data_dict = self.predictor.predecir(texto)

        if "Error" in resultado_str:
             self.lbl_resultado.config(text=resultado_str, fg="#FF5555")
             return

        self.lbl_resultado.config(text=f"Predicción: {resultado_str}", fg=self.success_color)
        
        # Pasamos el diccionario REAL al gráfico
        self.actualizar_grafico(data_dict)
    
    def actualizar_grafico(self, data_dict):
        # Limpiar anterior
        for widget in self.frame_grafico.winfo_children(): widget.destroy()

        if not data_dict: return

        # Extraer datos
        emociones = list(data_dict.keys())
        valores = list(data_dict.values())

        # Dibujar
        fig = Figure(figsize=(6, 4), dpi=100, facecolor=self.bg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        colores = ['#BD93F9', '#FF79C6', '#8BE9FD', '#50FA7B', '#FFB86C', '#FF5555']
        
        # Barras con valores reales
        barras = ax.bar(emociones, valores, color=colores[:len(emociones)])
        
        ax.set_title("Confianza Real del Modelo", color=self.text_color)
        ax.tick_params(colors=self.text_color)
        ax.set_ylim(0, 1.0) # Escala fija de 0 a 100%
        
        # Ocultar bordes
        ax.spines['bottom'].set_color(self.text_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionEmociones(root)
    root.mainloop()