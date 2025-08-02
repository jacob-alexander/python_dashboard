import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="Interactive Dashboard AI",
    page_icon="📊",
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
def load_data(uploaded_file):
    """Loads data from an uploaded file into a pandas DataFrame."""
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
            st.error("Unsupported file format. Please upload a .csv, .xlsx, .parquet, or .txt file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_matplotlib_code(user_request, df_head):
    """Generates Matplotlib code using Google's Gemini AI."""
    prompt = f"""
    Act as an expert Python data visualization specialist. Your ONLY task is to generate Python code to create a Matplotlib plot based on a user request.

    User Request: "{user_request}"

    DataFrame Head for context (the full DataFrame is available in the variable `df`):
    ```
    {df_head.to_string()}
    ```

    **CRITICAL INSTRUCTIONS:**
    1.  Your entire output MUST BE a single block of Python code. Do not include any explanation, markdown, or text.
    2.  You MUST name the plot's figure object `fig`. Always start your code with `fig, ax = plt.subplots()`. This is mandatory.
    3.  **Visualization Rule 1:** If the user requests a pie chart AND the number of categories to plot is greater than 10, generate a horizontal bar chart instead. A horizontal bar chart is much more readable.
    4.  **Visualization Rule 2:** If you generate a bar chart and the number of categories is greater than 15, ONLY PLOT THE TOP 15. This is crucial for readability. The title should reflect this (e.g., "Top 15...").
    5.  For all bar charts, sort the values so the chart is easy to interpret.
    6.  The code must use the pandas DataFrame variable named `df`.
    7.  Do NOT include `plt.show()`. The environment will handle rendering.
    8.  Use `plt.tight_layout()` at the end to prevent labels from being cut off.

    **Example of a PERFECT output for 'bar chart of sales by location' (with many locations):**
    ```python
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots()
    # Group, sum, and get the top 15 locations
    data_to_plot = df.groupby('LOCATION')['SALES_VALUE'].sum().nlargest(15)
    # Sort again for horizontal bar chart display (largest at top)
    data_to_plot.sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Top 15 Locations by Total Sales Value')
    ax.set_xlabel('Total Sales Value')
    ax.set_ylabel('Location')
    plt.tight_layout()
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

    col1, col2 = st.columns([4, 1])
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

    user_request = text_request or st.session_state.voice_text
    
    if st.session_state.voice_text:
        st.info(f"Voice request: \"{st.session_state.voice_text}\"")
        user_request = st.session_state.voice_text
        st.session_state.voice_text = ""

    if user_request:
        if st.button("Generate Visualization", key="generate_viz"):
            with st.spinner("AI is generating visualization code..."):
                df_head = st.session_state.df.head()
                matplotlib_code = get_matplotlib_code(user_request, df_head)

                if matplotlib_code:
                    st.session_state.analysis_history.insert(0, {
                        "request": user_request,
                        "code": matplotlib_code
                    })
                else:
                    st.error("Failed to generate visualization code.")
                st.rerun() # Rerun to display the new chart immediately

# --- Display Analysis History ---
if st.session_state.analysis_history:
    st.header("3. Visualization Results")
    
    analysis = st.session_state.analysis_history[0]
    st.subheader(f"Request: \"{analysis['request']}\"")
    
    with st.expander("Show Generated Code"):
        st.code(analysis['code'], language='python')
    
    try:
        plt.clf()
        df = st.session_state.df
        local_scope = {"df": df, "plt": plt, "pd": pd}
        exec(analysis['code'], local_scope)
        
        fig = local_scope.get('fig', None)
        if fig:
            st.pyplot(fig)
        else:
            st.warning("Generated code ran successfully, but did not produce a Matplotlib figure named 'fig'.")

    except Exception as e:
        st.error(f"Error executing the generated code for request: \"{analysis['request']}\"")
        st.info("AI Generated Code (that caused the error):")
        st.code(analysis['code'], language='python')
        st.info("Error Details:")
        st.code(traceback.format_exc())

elif st.session_state.df is None:
    st.info("Upload a file to get started.")