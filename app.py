import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import time
import threading
import mss

# ================= CONFIGURAÇÃO DA TELA =================
# Defina a área ONDE O VÍDEO DA REUNIÃO VAI FICAR (Esquerda)
CAPTURE_CONFIG = {
    "top": 50,      # Posição Y inicial
    "left": 0,      # Posição X inicial (Canto esquerdo)
    "width": 800,   # Largura da área de captura
    "height": 600   # Altura da área de captura
}

# Configurações de Janela do Preview (Para não ficar na frente da captura)
TITULO_JANELA = "PREVIEW - Arraste para longe da area de captura"
# ========================================================

# ================= CONFIGURAÇÃO DE IA =================
EMOCOES_ALVO = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
CORES_GRAFICO = {'angry': 'red', 'disgust': 'green', 'fear': 'purple', 'happy': 'yellow', 'sad': 'blue', 'surprise': 'orange', 'neutral': 'gray'}
CORES_VIDEO = {'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128), 'happy': (0, 255, 255), 'sad': (255, 0, 0), 'surprise': (0, 165, 255), 'neutral': (200, 200, 200)}

INTERVALO_ANALISE = 0.5 
TAMANHO_HISTORICO = 50
# ========================================================

# Estruturas Globais
dados_recente = {emo: deque([0.0]*TAMANHO_HISTORICO, maxlen=TAMANHO_HISTORICO) for emo in EMOCOES_ALVO}
historico_sessao = []
resultado_atual = [] 
lock_ia = False 

# --- Configuração do Gráfico ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 4)) # Gráfico um pouco menor
linhas = {}
x_data = np.arange(TAMANHO_HISTORICO)

for emo in EMOCOES_ALVO:
    linha, = ax.plot(x_data, dados_recente[emo], label=emo.capitalize(), color=CORES_GRAFICO[emo], linewidth=1.5)
    linhas[emo] = linha

ax.set_ylim(0, 100)
ax.set_title("Analise de Tela em Tempo Real")
ax.legend(loc='upper left', fontsize='x-small')
plt.tight_layout()

# --- Função de IA ---
def analisar_frame(frame_bgr):
    global resultado_atual, lock_ia
    try:
        objs = DeepFace.analyze(frame_bgr, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
        resultado_atual = objs
    except: pass 
    finally: lock_ia = False

# --- Inicialização ---
print("\n>>> SISTEMA INICIADO <<<")
print(f"1. Mova a janela do Teams/Zoom para o CANTO SUPERIOR ESQUERDO.")
print(f"2. A janela de Preview vai abrir na Direita.")
print("3. Pressione 'q' na janela de Preview para sair.")

sct = mss.mss()
ultimo_tempo_analise = time.time()

# Configura a janela do OpenCV para abrir na direita (longe da captura)
cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
cv2.moveWindow(TITULO_JANELA, 900, 50) # Move para X=900, Y=50
cv2.resizeWindow(TITULO_JANELA, 600, 450)

try:
    while True:
        # 1. CAPTURA A TELA (Área definida no topo)
        screenshot = np.array(sct.grab(CAPTURE_CONFIG))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # 2. IA (Thread)
        agora = time.time()
        if (agora - ultimo_tempo_analise) > INTERVALO_ANALISE and not lock_ia:
            lock_ia = True
            ultimo_tempo_analise = agora
            frame_copy = frame.copy()
            threading.Thread(target=analisar_frame, args=(frame_copy,)).start()

        # 3. Desenho
        if resultado_atual and isinstance(resultado_atual, list):
            faces_detectadas = 0
            soma_emocoes = {e: 0.0 for e in EMOCOES_ALVO}
            for face_obj in resultado_atual:
                if face_obj.get('region', {}).get('w', 0) > 0:
                    faces_detectadas += 1
                    r = face_obj['region']
                    cv2.rectangle(frame, (r['x'], r['y']), (r['x']+r['w'], r['y']+r['h']), (0, 255, 0), 2)
                    for emo in EMOCOES_ALVO: soma_emocoes[emo] += face_obj['emotion'].get(emo, 0)

            if faces_detectadas > 0:
                media = {e: soma_emocoes[e]/faces_detectadas for e in EMOCOES_ALVO}
                # Atualiza dados
                timestamp = time.time()
                reg = {'timestamp': timestamp}
                for emo in EMOCOES_ALVO:
                    dados_recente[emo].append(media[emo])
                    linhas[emo].set_ydata(dados_recente[emo])
                    reg[emo] = media[emo]
                historico_sessao.append(reg)
                # Texto
                dom = max(media, key=media.get)
                cv2.putText(frame, f"{dom.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, CORES_VIDEO.get(dom), 2)
            else:
                for emo in EMOCOES_ALVO: dados_recente[emo].append(0); linhas[emo].set_ydata(dados_recente[emo])
        else:
            for emo in EMOCOES_ALVO: dados_recente[emo].append(0); linhas[emo].set_ydata(dados_recente[emo])

        # 4. Atualiza Janelas
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except: pass
        
        cv2.imshow(TITULO_JANELA, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Parando...")

finally:
    cv2.destroyAllWindows()
    plt.close()
    
    # GERA RELATÓRIOS SIMPLES
    if historico_sessao:
        print("Salvando graficos...")
        df = pd.DataFrame(historico_sessao)
        df['tempo'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        plt.figure(figsize=(10, 5))
        for emo in EMOCOES_ALVO: plt.plot(df['tempo'], df[emo], label=emo, color=CORES_GRAFICO[emo])
        plt.title("Historico Tela"); plt.legend(); plt.savefig('final_timeline.png')
        
        plt.figure(figsize=(6, 4))
        p = df['happy'].sum()
        n = df[['angry', 'fear', 'sad', 'disgust']].sum().sum()
        nt = df[['neutral', 'surprise']].sum().sum()
        sns.barplot(x=['Pos', 'Neg', 'Neutro'], y=[p, n, nt]); plt.savefig('final_balanco.png')
        print("Pronto.")