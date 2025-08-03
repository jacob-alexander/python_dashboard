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
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            return pd.read_csv(uploaded_file, delimiter='\t')
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_plotly_python_code(user_request, df_head):
    """Generates Python code for a Plotly chart using Google's Gemini AI."""
    # This is the updated and corrected prompt
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
    3.  **IMPORTANT SYNTAX:** Do NOT use `subplots()` with Plotly (e.g., `px.subplots()` or `go.subplots()` will cause an error). The `subplots` function is for the Matplotlib library. Create Plotly figures directly, for example: `fig = px.bar(...)` or `fig = go.Figure(...)`.
    4.  **Handle Date/Time Data:** If a column looks like a date (e.g., 'SALES_DATE'), you MUST convert it to a datetime object first using `df['COLUMN_NAME'] = pd.to_datetime(df['COLUMN_NAME'])` before performing any time-based analysis.
    5.  **Use Plotly Express or Graph Objects:** Use `plotly.express` (as `px`) for simple charts or `plotly.graph_objects` (as `go`) for more complex ones.
    6.  **Summarization Rule:** If the user asks for a chart with more than 15 categories (e.g., bar chart by location), ONLY plot the "Top 15". The title should reflect this.
    7.  Make the chart look professional. Use a clean template like `template='plotly_dark'` to match the app's dark theme.

    **Example of a PERFECT output for 'daily bar graph of total sales':**
    ```python
    import pandas as pd
    import plotly.express as px

    # Ensure the date column is in datetime format
    df['SALES_DATE'] = pd.to_datetime(df['SALES_DATE'])
    
    # Group by date and sum sales
    daily_sales = df.groupby(df['SALES_DATE'].dt.date)['SALES_VALUE'].sum().reset_index()
    
    # Create the figure directly with Plotly Express
    fig = px.bar(daily_sales, 
                 x='SALES_DATE', 
                 y='SALES_VALUE', 
                 title='Total Sales Value per Day',
                 labels={{'SALES_DATE': 'Date', 'SALES_VALUE': 'Total Sales'}})
    
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
    if st.button("Restart Session & Clear"):
        st.session_state.clear()
        st.rerun()

# --- Main Content ---
if st.session_state.df is not None:
    st.header("2. Ask Your Data a Question")
    
    user_request = st.text_input(
        "Enter your request", 
        placeholder="e.g., 'What are the daily sales trends?'",
        label_visibility="collapsed"
    )

    if st.button("Generate Chart", key="generate"):
        if user_request:
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
        else:
            st.warning("Please enter a request.")

    # --- Display the LATEST analysis result ---
    if st.session_state.analysis_history:
        latest_analysis = st.session_state.analysis_history[0]
        st.header("3. Visualization Result")
        st.subheader(f"Request: \"{latest_analysis['request']}\"")
        
        tab1, tab2 = st.tabs(["📊 Chart", "📄 Generated Code"])

        with tab1:
            try:
                local_scope = {
                    "df": st.session_state.df.copy(),
                    "pd": pd,
                    "px": px,
                    "go": go
                }
                exec(latest_analysis['code'], local_scope)
                
                fig = local_scope.get('fig', None)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("The generated code ran, but did not produce a Plotly figure named 'fig'.")

            except Exception as e:
                st.error(f"Error executing the generated Python code.")
                st.code(traceback.format_exc(), language='text')

        with tab2:
            st.code(latest_analysis['code'], language='python')
else:
    st.info("Please upload a data file in the sidebar to get started.")
