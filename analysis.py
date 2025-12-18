import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Análise Emocional", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title(" Relatório Executivo de Análise Emocional")
st.markdown("Análise ajustada para **30 minutos**. Interpretação: **Foco (Verde)** e **Reflexão (Azul)**.")

@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv("dados_reuniao.csv")
        df = df.rename(columns={
            'disgust': 'focused',
            'sad': 'thoughtful'
        })

        if 'tempo' in df.columns:
            max_tempo = df['tempo'].max()
            if max_tempo > 0:
                df['tempo_real'] = (df['tempo'] / max_tempo) * 30
            else:
                df['tempo_real'] = 0
        else:
            df['tempo_real'] = (df.index / len(df)) * 30

        return df
    except FileNotFoundError:
        return None

df = carregar_dados()

if df is None:
    st.error("Arquivo 'dados_reuniao.csv' não encontrado. Verifique se ele está na mesma pasta do script.")
    st.stop()

color_map = {
    'angry': 'red',
    'focused': 'green',
    'fear': 'purple',
    'happy': '#FFD700',
    'thoughtful': 'blue',
    'surprise': 'orange',
    'neutral': 'gray'
}

cols_presentes = [c for c in color_map.keys() if c in df.columns]
cols_negativas = [c for c in ['angry', 'fear'] if c in cols_presentes]
cols_produtivas = [c for c in ['focused', 'thoughtful'] if c in cols_presentes]
cols_positivas = [c for c in ['happy'] if c in cols_presentes]

st.subheader("Resumo dos 30 Minutos")
c1, c2, c3, c4 = st.columns(4)

media_happy = df['happy'].mean() if 'happy' in df.columns else 0
media_foco = df[cols_produtivas].sum(axis=1).mean() if cols_produtivas else 0
media_tensao = df[cols_negativas].sum(axis=1).mean() if cols_negativas else 0

medias_gerais = df[cols_presentes].mean()
top_emocao = medias_gerais.idxmax()
cor_top = color_map.get(top_emocao, 'black')

with c1:
    st.metric(" Duração Simulada", "30.0 min")
with c2:
    st.metric(" Nível de Produtividade", f"{media_foco:.1f}%")
with c3:
    st.metric(" Nível de Bem-Estar", f"{media_happy:.1f}%")
with c4:
    st.markdown("**Estado Predominante**")
    st.markdown(f"<h3 style='color: {cor_top}; margin:0'>{top_emocao.upper()}</h3>", unsafe_allow_html=True)

st.divider()

st.subheader(" Evolução Temporal (0 a 30 min)")

fig_timeline = px.line(
    df,
    x='tempo_real',
    y=cols_presentes,
    labels={
        'value': 'Intensidade (%)',
        'tempo_real': 'Tempo (Minutos)',
        'variable': 'Estado'
    },
    color_discrete_map=color_map,
    height=500
)

fig_timeline.update_layout(
    xaxis_title="Tempo de Reunião (Minutos)",
    yaxis_title="Intensidade (%)",
    hovermode="x unified",
    xaxis=dict(range=[0, 30])
)

try:
    st.plotly_chart(fig_timeline, width="stretch")
except:
    st.plotly_chart(fig_timeline, use_container_width=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Distribuição Geral")
    fig_pie = px.pie(
        values=medias_gerais.values,
        names=medias_gerais.index,
        color=medias_gerais.index,
        color_discrete_map=color_map,
        hole=0.4
    )
    try:
        st.plotly_chart(fig_pie, width="stretch")
    except:
        st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Momentos de Atenção (Tensão Real)")
    st.write("Registros onde **Raiva** ou **Medo** superaram 20%:")

    if cols_negativas:
        filtro = df[df[cols_negativas].max(axis=1) > 20]
        if not filtro.empty:
            cols_show = ['tempo_real'] + cols_negativas
            try:
                st.dataframe(
                    filtro[cols_show].style.format({"tempo_real": "{:.2f} min"}),
                    height=300,
                    width="stretch"
                )
            except:
                st.dataframe(
                    filtro[cols_show].style.format({"tempo_real": "{:.2f} min"}),
                    height=300,
                    use_container_width=True
                )
        else:
            st.success("Nenhum pico significativo de estresse detectado.")
    else:
        st.info("Sem dados de emoções negativas para analisar.")

st.divider()
st.subheader(" Conclusão Automática da IA")

if media_foco > 40:
    conclusao = "Reunião Altamente Produtiva e Focada."
    detalhe = "Altos níveis de concentração e reflexão indicam trabalho eficiente."
elif media_happy > 30:
    conclusao = "Reunião Leve e Positiva."
    detalhe = "Ambiente descontraído e boa receptividade."
elif media_tensao > 15:
    conclusao = "Reunião Tensa ou Conflituosa."
    detalhe = "Sinais de desconforto merecem investigação."
else:
    conclusao = "Reunião Protocolar/Neutra."
    detalhe = "Sem grandes oscilações emocionais."

metade_idx = len(df) // 2
inicio_prod = df.iloc[:metade_idx][cols_produtivas].sum(axis=1).mean() if cols_produtivas else 0
fim_prod = df.iloc[metade_idx:][cols_produtivas].sum(axis=1).mean() if cols_produtivas else 0
tendencia_texto = "aumentou" if fim_prod > inicio_prod else "diminuiu"

st.info(f"**Veredito:** {conclusao}")
st.write(detalhe)
st.write(f"**Tendência de Foco:** O nível de atenção **{tendencia_texto}** na segunda metade da reunião.")

if 'surprise' in df.columns and df['surprise'].max() > 50:
    tempo_surpresa_idx = df['surprise'].idxmax()
    tempo_val = df.iloc[tempo_surpresa_idx]['tempo_real']
    st.write(f" Momento de **Surpresa** por volta de **{tempo_val:.2f} min**.")
