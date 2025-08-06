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

def get_python_code(user_request, df_head):
    """Generates Python code for a Plotly chart OR a pandas DataFrame."""
    # This prompt is our most advanced, handling both charts and data tables.
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

# --- Sidebar ---
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

# --- Main Content ---
if st.session_state.df is not None:
    st.header("2. Ask Your Data a Question")
    col1, col2 = st.columns([5, 1])
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
                tab1, tab2 = st.tabs([f"📊 Output #{i+1}", f"📄 Code #{i+alues']
        #   ax.set_title('Total Sales Value by Location')
        #   ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ```
    
        Now, generate the Python code for the user request.
        """
        try:
            response = model.generate_content(prompt)
            # Clean up the response to get only the code part
            code = response.text.strip()
            if code.startswith("```python"):
                code = code[9:]
            if code.endswith("```"):
                code = code[:-3]
            return code.strip()
        except Exception as e:
            st.error(f"An error occurred with the AI model: {e}")
            return None
    
    # --- Session State Initialization ---
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'voice_text' not in st.session_state:
        st.session_state.voice_text = ""
    
    
    # --- Main Application ---
    st.title("🤖 AI-Powered Interactive Dashboard")
    st.markdown("Upload your data, ask questions in plain English (or use your voice!), and get visualizations instantly.")
    
    # --- Sidebar for Data Upload and Options ---
    with st.sidebar:
        st.header("1. Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a file (xlsx, parquet, csv, txt)",
            type=['xlsx', 'parquet', 'csv', 'txt']
        )
    
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)
    
        if st.session_state.df is not None:
            st.success("File loaded successfully!")
            st.markdown("---")
            st.header("Data Preview")
            st.write("Column Names:", st.session_state.df.columns.tolist())
            st.write("Data Sample:", st.session_state.df.head())
        
        st.markdown("---")
        st.header("Options")
        if st.button("Restart Session & Clear History"):
            st.session_state.clear()
            st.rerun()
    
    
    # --- Main Content Area for Analysis ---
    if st.session_state.df is not None:
        st.header("2. Analyze Your Data")
    
        # --- User Input: Text and Voice ---
        col1, col2 = st.columns()
        with col1:
            text_request = st.text_input(
                "Enter your analysis request (e.g., 'Plot a bar chart of sales by region')",
                key="text_request_input"
            )
        with col2:
            st.write("Or use your voice:")
            voice_result = mic_recorder(
                start_prompt="🎤",
                stop_prompt="⏹️",
                key='voice_recorder'
            )
            if voice_result and 'text' in voice_result:
                st.session_state.voice_text = voice_result['text']
    
    
        # --- Combine Inputs and Process ---
        user_request = text_request or st.session_state.voice_text
        
        if st.session_state.voice_text:
            st.info(f"Voice request: \"{st.session_state.voice_text}\"")
            # Set voice text as the main request and clear it
            user_request = st.session_state.voice_text
            st.session_state.voice_text = ""
    
        if user_request:
            if st.button("Generate Visualization", key="generate_viz"):
                with st.spinner("AI is generating visualization code..."):
                    df_head = st.session_state.df.head()
                    matplotlib_code = get_matplotlib_code(user_request, df_head)
    
                    if matplotlib_code:
                        st.session_state.analysis_history.append({
                            "request": user_request,
                            "code": matplotlib_code
                        })
                    else:
                        st.error("Failed to generate visualization code.")
    
    # --- Display Analysis History ---
    if st.session_state.analysis_history:
        st.header("3. Visualization Results")
        
        for analysis in reversed(st.session_state.analysis_history):
            st.subheader(f"Request: \"{analysis['request']}\"")
            
            with st.expander("Show Generated Code"):
                st.code(analysis['code'], language='python')
            
            try:
                # Clear any previous plots to start with a clean slate
                plt.clf()
                
                # Execute the generated code
                df = st.session_state.df # Make the dataframe available
                # We also provide pd and plt in the scope for the AI to use if needed
                local_scope = {"df": df, "plt": plt, "pd": pd}
                exec(analysis['code'], local_scope)
                
                # Get the figure from the local scope
                fig = local_scope.get('fig', None)
                if fig:
                    st.pyplot(fig)
                else:
                    # This is the error you were seeing. Now it's more helpful.
                    st.warning("Generated code ran successfully, but did not produce a Matplotlib figure named 'fig'.")
                    st.info("This can happen if the AI's response format is slightly off. The prompt has been improved to prevent this.")
    
            except Exception as e:
                # If the code itself has an error, show the details.
                st.error(f"Error executing the generated code for request: \"{analysis['request']}\"")
                st.info("AI Generated Code (that caused the error):")
                st.code(analysis['code'], language='python')
                st.info("Error Details:")
                st.code(traceback.format_exc())
            
            st.markdown("---")
    
    elif st.session_state.df is None:
        st.info("Upload a file to get started.")
    ```
    
    ### `requirements.txt` File
    
    Don't forget this file. It's essential for telling Streamlit Cloud which libraries your app needs. Make sure this file is in the same folder as your `app.py`.
    
    ```
    streamlit
    pandas
    google-generativeai
    openpyxl
    pyarrow
    streamlit-mic-recorder
    ```
    
    You now have a complete, robust, and much more intelligent version of your interactive dashboard application.
    
    
    
    
    
