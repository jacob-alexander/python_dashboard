import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Professional AI Dashboard", page_icon="✨", layout="wide")

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    st.error("Google API Key not found...")
    st.stop()

@st.cache_data
def load_data(uploaded_file):
    try:
        file_copy = uploaded_file
        if file_copy.name.endswith('.csv'): return pd.read_csv(file_copy)
        elif file_copy.name.endswith('.xlsx'): return pd.read_excel(file_copy)
        elif file_copy.name.endswith('.parquet'): return pd.read_parquet(file_copy)
        elif file_copy.name.endswith('.txt'): return pd.read_csv(file_copy, delimiter='\t')
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_plotly_python_code(user_request, df_head):
    prompt = f"""... # Paste the "Ultimate Prompt" from above here ..."""
    try:
        response = model.generate_content(prompt)
        code = response.text.strip().removeprefix("```python").removesuffix("```")
        return code.strip()
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        st.info("AI Response:", icon="🤖").code(response.text, language='text')
        return None

if "df" not in st.session_state: st.session_state.df = None
if "analysis_history" not in st.session_state: st.session_state.analysis_history = []

st.title("✨ Professional AI-Powered Dashboard")

with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'parquet', 'csv', 'txt'], label_visibility="collapsed")
    if uploaded_file: st.session_state.df = load_data(uploaded_file)
    if st.session_state.df is not None:
        st.success("Data loaded!")
        with st.expander("Data Preview"): st.dataframe(st.session_state.df.head())
    st.divider()
    st.header("Actions")
    if st.button("Restart Session & Clear History"):
        st.session_state.clear()
        st.rerun()

if st.session_state.df is not None:
    st.header("2. Ask Your Data a Question")
    col1, col2 = st.columns([5, 1])
    with col1: text_request = st.text_input("Enter your request", placeholder="e.g., 'Show me a heatmap of correlations' or use the mic ->", label_visibility="collapsed")
    with col2: voice_request = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='voice_recorder')
    
    user_request = voice_request['text'] if voice_request and 'text' in voice_request else text_request
    if voice_request and 'text' in voice_request: st.info(f"Heard you say: \"{user_request}\"")

    if st.button("Generate Chart", key="generate") and user_request:
        with st.spinner("AI is performing analysis and creating your chart..."):
            python_code = get_plotly_python_code(user_request, st.session_state.df.head())
            if python_code: st.session_state.analysis_history.insert(0, {"request": user_request, "code": python_code})
            else: st.error("Could not generate the Python code for the chart.")
    
    st.divider()
    if st.session_state.analysis_history:
        st.header("3. Analysis History")
        for i, analysis in enumerate(st.session_state.analysis_history):
            st.subheader(f"Request: \"{analysis['request']}\"")
            tab1, tab2 = st.tabs([f"📊 Chart #{i+1}", f"📄 Code #{i+1}"])
            with tab1:
                try:
                    local_scope = {"df": st.session_state.df.copy(), "pd": pd, "px": px, "go": go}
                    exec(analysis['code'], local_scope)
                    fig = local_scope.get('fig')
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    else: st.warning("Generated code ran, but did not produce a figure named 'fig'.")
                except Exception: st.error("Error executing generated code."); st.code(traceback.format_exc(), language='text')
            with tab2: st.code(analysis['code'], language='python')
            st.divider()
else:
    st.info("Please upload a data file in the sidebar to get started.")
