import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
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
        if file_copy.name.endswith('.csv'): return pd.read_csv(file_copy)
        elif file_copy.name.endswith('.xlsx'): return pd.read_excel(file_copy)
        elif file_copy.name.endswith('.parquet'): return pd.read_parquet(file_copy)
        elif file_copy.name.endswith('.txt'): return pd.read_csv(file_copy, delimiter='\t')
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_python_code(user_request, df_head):
    """Generates Python code for a Plotly chart OR a pandas DataFrame."""
    prompt = f"""
    Act as an expert Python data analyst. Your task is to generate a Python script based on a user request.
    The script will produce EITHER a Plotly figure OR a pandas DataFrame for display.

    User Request: "{user_request}"

    DataFrame Head for context (the full DataFrame is available in the variable `df`):
    ```
    {df_head.to_string()}
    ```

    **CRITICAL INSTRUCTIONS:**
    1.  Your entire output MUST BE a single block of runnable Python code. No explanations.
    2.  **CHOOSE YOUR OUTPUT:**
        - If the user asks for a **chart, plot, graph, or visualization**, create a Plotly Figure object and assign it to a variable named `fig`.
        - If the user asks for a **table, data, list, or a summary**, create a pandas DataFrame with the result and assign it to a variable named `result_df`.
    3.  **CHARTING RULES (if you create a `fig`):**
        - Do NOT use `subplots()`. Create figures directly (e.g., `fig = px.bar(...)`).
        - For heatmaps, use `px.imshow()`. `px.heatmap` does not exist.
        - Use `template='plotly_dark'`.
    4.  **DATA RULES (if you create a `result_df`):**
        - The final output in the script should be the creation of the `result_df` DataFrame.
    5.  **GENERAL RULES:**
        - For any analysis involving dates, convert date-like columns to datetime objects first using `pd.to_datetime()`.
        - For categorical analysis with more than 15 categories, focus on the "Top 15" for readability.

    **EXAMPLE 1: User asks for a chart**
    *   REQUEST: 'Show me a bar chart of sales by location'
    *   PERFECT OUTPUT:
    ```python
    import pandas as pd
    import plotly.express as px
    top_15_locations = df.groupby('LOCATION')['SALES_VALUE'].sum().nlargest(15).reset_index()
    fig = px.bar(top_15_locations, x='LOCATION', y='SALES_VALUE', title='Top 15 Locations by Sales')
    fig.update_layout(template='plotly_dark', title_x=0.5)
    ```

    **EXAMPLE 2: User asks for a table**
    *   REQUEST: 'total sales by store in a table'
    *   PERFECT OUTPUT:
    ```python
    import pandas as pd
    result_df = df.groupby('LOCATION')['SALES_VALUE'].sum().reset_index()
    result_df = result_df.rename(columns={{'LOCATION': 'Store Location', 'SALES_VALUE': 'Total Sales'}})
    result_df = result_df.sort_values(by='Total Sales', ascending=False)
    ```
    Now, generate the Python code for the user request.
    """
    try:
        response = model.generate_content(prompt)
        code = response.text.strip().removeprefix("```python").removesuffix("```")
        return code.strip()
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        st.info("AI Response that may have caused the error:").code(response.text, language='text')
        return None

# --- Session State Initialization ---
if "df" not in st.session_state: st.session_state.df = None
if "analysis_history" not in st.session_state: st.session_state.analysis_history = []


# --- Main Application ---
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
    col1, col2 = st.columns()
    with col1: text_request = st.text_input("Enter your request", placeholder="e.g., 'Show top 10 stores by sales' or use the mic ->", label_visibility="collapsed")
    with col2: voice_request = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key='voice_recorder')
    
    user_request = voice_request['text'] if voice_request and 'text' in voice_request else text_request
    if voice_request and 'text' in voice_request: st.info(f"Heard you say: \"{user_request}\"")

    if st.button("Generate Output", key="generate") and user_request:
        with st.spinner("AI is analyzing your data..."):
            python_code = get_python_code(user_request, st.session_state.df.head())
            if python_code: st.session_state.analysis_history.insert(0, {"request": user_request, "code": python_code})
            else: st.error("Could not generate the Python code for your request.")
    
    st.divider()
    if st.session_state.analysis_history:
        st.header("3. Analysis History")
        for i, analysis in enumerate(st.session_state.analysis_history):
            with st.container(border=True):
                st.subheader(f"Request: \"{analysis['request']}\"")
                
                # THIS IS THE LINE THAT HAS BEEN MANUALLY CORRECTED
                tab1, tab2 = st.tabs([f"📊 Output #{i+1}", f"📄 Code #{i+1}"])

                with tab1:
                    try:
                        local_scope = {"df": st.session_state.df.copy(), "pd": pd, "px": px, "go": go}
                        exec(analysis['code'], local_scope)
                        
                        fig = local_scope.get('fig')
                        result_df = local_scope.get('result_df')

                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        elif result_df is not None:
                            # Dynamically adjust height to avoid excessive scrolling
                            height = (len(result_df) + 1) * 35 + 3
                            st.dataframe(result_df, height=height, use_container_width=True)
                        else:
                            st.warning("The generated code ran, but did not produce a `fig` or a `result_df`.")
                    except Exception as e:
                        st.error("Error executing generated code.")
                        st.code(traceback.format_exc(), language='text')
                with tab2:
                    st.code(analysis['code'], language='python')
else:
    st.info("Please upload a data file in the sidebar to get started.")
