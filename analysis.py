import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Final Analysis Dashboard", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    h1 { color: #2c3e50; }
    h3 { color: #34495e; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Executive Emotional Analysis Report")
st.markdown("Analysis adjusted for a **30-minute** total duration.")

@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("dados_reuniao.csv")

        df_raw['emotion'] = df_raw['emotion'].replace({
            'disgust': 'focused',
            'sad': 'thoughtful'
        })

        max_frame = df_raw['minute'].max()
        if max_frame > 0:
            df_raw['real_time'] = (df_raw['minute'] / max_frame) * 30
        else:
            df_raw['real_time'] = 0

        df_raw['real_time'] = df_raw['real_time'].round(2)
        df_pivot = (
            df_raw
            .groupby(['real_time', 'emotion'])['intensity']
            .mean()
            .unstack(fill_value=0)
            .reset_index()
        )
        df_pivot.columns.name = None

        return df_pivot
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ File 'dados_reuniao.csv' not found.")
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

cols_present = [c for c in color_map.keys() if c in df.columns]

cols_negative = [c for c in ['angry', 'fear'] if c in cols_present]
cols_productive = [c for c in ['focused', 'thoughtful'] if c in cols_present]
cols_positive = [c for c in ['happy'] if c in cols_present]

st.subheader("30-Minute Summary")
c1, c2, c3, c4 = st.columns(4)

avg_happy = df['happy'].mean() if 'happy' in df.columns else 0
avg_focus = df[cols_productive].sum(axis=1).mean() if cols_productive else 0
avg_tension = df[cols_negative].sum(axis=1).mean() if cols_negative else 0

overall_means = df[cols_present].mean()
top_emotion = overall_means.idxmax()
top_color = color_map.get(top_emotion, 'black')

with c1:
    st.metric(" Total Duration", "30.0 min")
with c2:
    st.metric(" Productivity Level", f"{avg_focus:.1f}%", help="Sum of Focus + Reflection")
with c3:
    st.metric(" Well-being Level", f"{avg_happy:.1f}%", help="Happiness Level")
with c4:
    st.markdown("**Predominant State**")
    st.markdown(f"<h3 style='color: {top_color}; margin:0'>{top_emotion.upper()}</h3>", unsafe_allow_html=True)

st.divider()

st.subheader(" Temporal Evolution (0 to 30 min)")

fig_timeline = px.line(
    df,
    x='real_time',
    y=cols_present,
    labels={
        'value': 'Intensity (%)',
        'real_time': 'Time (Minutes)',
        'variable': 'State'
    },
    color_discrete_map=color_map,
    height=500
)
fig_timeline.update_layout(
    xaxis_title="Meeting Time (Minutes)",
    yaxis_title="Intensity (%)",
    hovermode="x unified",
    xaxis=dict(range=[0, 30])
)
st.plotly_chart(fig_timeline, use_container_width=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Overall Distribution")
    fig_pie = px.pie(
        values=overall_means.values,
        names=overall_means.index,
        color=overall_means.index,
        color_discrete_map=color_map,
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("⚠️ Attention Moments (Real Tension)")
    st.write("Records where **Anger** or **Fear** exceeded 20%:")

    if cols_negative:
        filtered = df[df[cols_negative].max(axis=1) > 20]
        if not filtered.empty:
            cols_show = ['real_time'] + cols_negative
            st.dataframe(
                filtered[cols_show].style.format({"real_time": "{:.2f} min"}),
                height=300,
                use_container_width=True
            )
        else:
            st.success(" No significant stress peaks (Anger/Fear) detected.")
    else:
        st.info("No negative emotion data available for analysis.")

st.divider()
st.subheader(" Automatic Conclusion")

conclusion = ""
details = ""

if avg_focus > 40:
    conclusion = "Highly Productive and Focused Meeting."
    details = (
        "Participants showed high levels of concentration and information processing "
        "('Thoughtful'), usually indicating an efficient work session or deep learning."
    )
elif avg_happy > 30:
    conclusion = "Light and Positive Meeting."
    details = "The environment was relaxed, with good participant engagement."
elif avg_tension > 15:
    conclusion = "Tense or Conflict-Prone Meeting."
    details = "Clear signs of discomfort, fear, or irritation were observed and should be investigated."
else:
    conclusion = "Procedural / Neutral Meeting."
    details = "Standard interactions with no major emotional fluctuations."

half = len(df) // 2
start_prod = df.iloc[:half][cols_productive].sum(axis=1).mean() if cols_productive else 0
end_prod = df.iloc[half:][cols_productive].sum(axis=1).mean() if cols_productive else 0
trend_text = "increased" if end_prod > start_prod else "decreased"

st.info(f"**Verdict:** {conclusion}")
st.write(details)

st.write(
    f"**Focus Trend:** The participants' attention level **{trend_text}** "
    "in the second half of the meeting."
)

if 'surprise' in df.columns and df['surprise'].max() > 50:
    surprise_idx = df['surprise'].idxmax()
    time_value = df.iloc[surprise_idx]['real_time']
    st.write(
        f"💡 A strong **Surprise** moment occurred around **{time_value:.2f} min**, "
        "which may indicate an important reveal or unexpected event."
    )
