import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ==========================================
# ZONA DE SIMULACIÓN (MOCK) - BACKEND FALSO
# ==========================================
def predecir_emocion_simulada(texto):
    """
    Simula la respuesta del modelo SVM de tu compañero.
    Retorna: (str) emoción ganadora, (dict) probabilidades
    """
    # Simular una pequeña espera de procesamiento
    time.sleep(0.5) 
    
    # Si el texto es muy corto, simular error o advertencia
    if len(texto) < 5:
        return None, None

    emociones = ["Alegría", "Ira", "Tristeza", "Miedo"]
    
    # Generar probabilidades aleatorias que sumen 1 (aprox)
    raw_probs = [random.random() for _ in range(4)]
    total = sum(raw_probs)
    probs_normalizadas = [p/total for p in raw_probs]
    
    resultado_dict = dict(zip(emociones, probs_normalizadas))
    
    # Determinar ganadora
    emocion_ganadora = max(resultado_dict, key=resultado_dict.get)
    
    return emocion_ganadora, resultado_dict

# ==========================================
# INTERFAZ GRÁFICA (FRONTEND) - TU TRABAJO
# ==========================================
class AplicacionEmociones:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador Emocional")
        self.root.geometry("900x650")
        
        # --- COLORES TEMA OSCURO (Basado en tu boceto) ---
        self.bg_color = "#1E1E2E"       # Fondo oscuro
        self.panel_color = "#282A36"    # Paneles
        self.text_color = "#F8F8F2"     # Texto claro
        self.accent_color = "#6272A4"   # Botones/Bordes
        self.success_color = "#50FA7B"  # Verde neón suave
        
        self.root.configure(bg=self.bg_color)
        
        self.crear_widgets()

    def crear_widgets(self):
        # 1. TÍTULO
        lbl_titulo = tk.Label(self.root, text="Clasificador Emocional", 
                              font=("Arial", 20, "bold"), 
                              bg=self.bg_color, fg=self.text_color)
        lbl_titulo.pack(pady=15)

        # 2. ÁREA DE ENTRADA (Izquierda o Arriba)
        frame_input = tk.Frame(self.root, bg=self.panel_color, padx=10, pady=10)
        frame_input.pack(fill="x", padx=20)

        lbl_instruccion = tk.Label(frame_input, text="Escribe o pega tu texto aquí:", 
                                   bg=self.panel_color, fg=self.text_color)
        lbl_instruccion.pack(anchor="w")

        self.txt_entrada = tk.Text(frame_input, height=5, bg="#44475A", 
                                   fg="white", insertbackground="white", font=("Arial", 12))
        self.txt_entrada.pack(fill="x", pady=5)

        btn_analizar = tk.Button(frame_input, text="Analizar Emociones", 
                                 command=self.realizar_analisis,
                                 bg=self.accent_color, fg="white", 
                                 font=("Arial", 12, "bold"), cursor="hand2")
        btn_analizar.pack(pady=5)

        # 3. ÁREA DE RESULTADOS
        self.frame_resultados = tk.Frame(self.root, bg=self.bg_color)
        self.frame_resultados.pack(fill="both", expand=True, padx=20, pady=10)

        # -- Resultado Texto (Emoción Ganadora)
        self.lbl_resultado = tk.Label(self.frame_resultados, 
                                      text="Esperando análisis...",
                                      font=("Arial", 16), 
                                      bg=self.bg_color, fg="gray")
        self.lbl_resultado.pack(pady=10)

        # -- Gráfico (Placeholder para Matplotlib)
        self.frame_grafico = tk.Frame(self.frame_resultados, bg=self.bg_color)
        self.frame_grafico.pack(fill="both", expand=True)

    def realizar_analisis(self):
        # 1. Obtener texto
        texto = self.txt_entrada.get("1.0", "end-1c").strip()
        
        if not texto:
            messagebox.showwarning("Alerta", "Por favor ingresa un texto.")
            return

        # 2. LLAMADA A LA SIMULACIÓN (Aquí conectarás con tu compañero después)
        # -------------------------------------------------------------------
        ganadora, probabilidades = predecir_emocion_simulada(texto)
        # -------------------------------------------------------------------

        if ganadora is None:
             messagebox.showerror("Error", "Texto insuficiente para analizar.")
             return

        # 3. Actualizar Texto
        self.lbl_resultado.config(text=f"Emoción Dominante: {ganadora.upper()}", 
                                  fg=self.success_color)

        # 4. Actualizar Gráfico
        self.actualizar_grafico(probabilidades)

    def actualizar_grafico(self, probabilidades):
        # Limpiar gráfico anterior si existe
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()

        emociones = list(probabilidades.keys())
        valores = list(probabilidades.values())

        # Crear figura Matplotlib (Pequeña para que quepa)
        fig = Figure(figsize=(5, 4), dpi=100, facecolor=self.bg_color)
        ax = fig.add_subplot(111)
        
        # Configurar colores del gráfico para tema oscuro
        ax.set_facecolor(self.bg_color)
        colores_barras = ['#FF5555', '#50FA7B', '#8BE9FD', '#FFB86C'] # Rojo, Verde, Cyan, Naranja
        
        barras = ax.bar(emociones, valores, color=colores_barras)
        
        # Estilo de ejes y texto
        ax.set_title("Distribución de Probabilidades", color=self.text_color)
        ax.tick_params(axis='x', colors=self.text_color)
        ax.tick_params(axis='y', colors=self.text_color)
        ax.spines['bottom'].set_color(self.text_color)
        ax.spines['left'].set_color(self.text_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Integrar en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        canvas.draw()
        canvas.get_tk_widget().pack()

# --- BLOQUE PRINCIPAL ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionEmociones(root)
    root.mainloop()