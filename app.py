import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration (applied from .streamlit/config.toml) ---
st.set_page_config(
    page_title="Professional AI Dashboard",
    page_icon="✨",
    layout="wide",
)

# --- Google AI API Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    st.error("Google API Key not found. Please set it in your Streamlit secrets or as an environment variable.")
    st.stop()


# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    """Caches the data loading to improve performance."""
    try:
        file_copy = uploaded_file
        if file_copy.name.endswith('.csv'):
            return pd.read_csv(file_copy)
        elif file_copy.name.endswith('.xlsx'):
            return pd.read_excel(file_copy)
        elif file_copy.name.endswith('.parquet'):
            return pd.read_parquet(file_copy)
        elif file_copy.name.endswith('.txt'):
            return pd.read_csv(file_copy, delimiter='\t')
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_plotly_python_code(user_request, df_head):
    """Generates Python code for a Plotly chart using Google's Gemini AI."""
    # This is the final, most robust prompt.
    prompt = f"""
    Act as an expert Python data scientist specializing in Plotly.
    Your task is to generate a Python script to create a Plotly figure based on a user request.

    User Request: "{user_request}"

    DataFrame Head for context (the full DataFrame is available in the variable `df`):
    ```
    {df_head.to_string()}
    ```

    **CRITICAL INSTRUCTIONS:**
    1.  Your entire output MUST BE a single block of runnable Python code. Do not include any explanation or markdown.
    2.  The script MUST create a Plotly Figure object and assign it to a variable named `fig`.
    3.  **SYNTAX - `subplots`:** Do NOT use `subplots()` with Plotly (e.g., `px.subplots()`). This is a Matplotlib function. Create Plotly figures directly: `fig = px.bar(...)`.
    4.  **SYNTAX - `heatmap`:** For heatmaps, you MUST use `px.imshow()` (for matrix-like data like correlations) or `go.Heatmap()`. The function `px.heatmap` does NOT exist and will cause an error.
    5.  **DATA - Date/Time:** If a column looks like a date (e.g., 'SALES_DATE'), you MUST convert it to datetime using `pd.to_datetime()` before any time-based analysis.
    6.  **DATA - Summarization:** If a user requests a chart with more than 15 categories (e.g., a bar chart), ONLY plot the "Top 15". The title should reflect this.
    7.  **STYLE:** Make charts look professional. Use a clean template like `template='plotly_dark'` to match the app's theme.

    **Example of a PERFECT output for 'heatmap of the correlation between numerical columns':**
    ```python
    import pandas as pd
    import plotly.express as px

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include='number')
    
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create the heatmap figure using px.imshow
    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    labels=dict(color="Correlation"),
                    title="Correlation Heatmap of Numerical Columns")
    
    fig.update_layout(template='plotly_dark', title_x=0.5)
    ```
    Now, generate the Python code for the user request.
    """
    try:
        response = model.generate_content(prompt)
        code = response.text.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        st.info("AI Response that may have caused the error:")
        st.code(response.text, language='text')
        return None

# --- Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# --- Main Application ---
st.title("✨ Professional AI-Powered Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a file", type=['xlsx', 'parquet', 'csv', 'txt'], label_visibility="collapsed"
    )
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)

    if st.session_state.df is not None:
        st.success("Data loaded successfully!")
        with st.expander("Data Preview"):
            st.dataframe(st.session_state.df.head())
    
    st.markdown("---")
    st.header("Actions")
    if st.button("Restart Session & Clear History"):
        st.session_state.clear()
        st.rerun()

# --- Main Content ---
if st.session_state.df is not None:
    st.header("2. Ask Your Data a Question")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        text_request = st.text_input(
            "Enter your request", 
            placeholder="e.g., 'Show me a heatmap of correlations' or use the mic ->",
            label_visibility="collapsed"
        )
    with col2:
        voice_request = mic_recorder(
            start_prompt="🎤", 
            stop_prompt="⏹️", 
            key='voice_recorder'
        )
    
    user_request = text_request
    if voice_request and 'text' in voice_request:
        user_request = voice_request['text']
        st.info(f"Heard you say: \"{user_request}\"")

    if st.button("Generate Chart", key="generate") and user_request:
        with st.spinner("AI is performing analysis and creating your chart..."):
            df_head = st.session_state.df.head()
            python_code = get_plotly_python_code(user_request, df_head)
            
            if python_code:
                st.session_state.analysis_history.insert(0, {
                    "request": user_request,
                    "code": python_code
                })
            else:
                st.error("Could not generate the Python code for the chart.")
    elif not user_request and st.session_state.get('generate'):
         st.warning("Please enter a request.")

    st.markdown("---")
    if st.session_state.analysis_history:
        st.header("3. Analysis History")
        
        for i, analysis in enumerate(st.session_state.analysis_history):
            with st.container():
                st.subheader(f"Request: \"{analysis['request']}\"")
                
                tab1, tab2 = st.tabs([f"📊 Chart #{i+1}", f"📄 Code #{i+1}"])

                with tab1:
                    try:
                        local_scope = {
                            "df": st.session_state.df.copy(),
                            "pd": pd,
                            "px": px,
                            "go": go
                        }
                        exec(analysis['code'], local_scope)
                        
                        fig = local_scope.get('fig', None)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Generated code ran, but did not produce a Plotly figure named 'fig'.")

                    except Exception as e:
                        st.error("Error executing the generated Python code.")
                        st.code(traceback.format_exc(), language='text')

                with tab2:
                    st.code(analysis['code'], language='python')
                
                st.markdown("---")
else:
    st.info("Please upload a data file in the sidebar to get started.")
