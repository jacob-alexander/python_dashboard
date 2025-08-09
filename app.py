import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
# --- CHANGE 1: Import make_subplots here ---
from plotly.subplots import make_subplots
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Dashboard Composer",
    page_icon="🎨",
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
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_python_code(user_request, df_head):
    """Generates Python code for complex Plotly charts, including subplots."""
    prompt = f"""
    Act as a world-class Python data scientist creating advanced data visualizations with Plotly.
    Your task is to generate a Python script to create a single Plotly Figure, which may contain multiple subplots.

    User Request: "{user_request}"

    DataFrame Head for context (the full DataFrame is available in the variable `df`):
    ```    {df_head.to_string()}
    ```

    **CRITICAL INSTRUCTIONS:**
    1.  Your entire output MUST BE a single block of runnable Python code. No explanations.
    2.  The script's final output MUST be a single Plotly Figure object assigned to a variable named `fig`.
    3.  **SUBPLOTS:** If the user asks for multiple charts or views (e.g., "show sales and profit over time"), you MUST use `plotly.subplots.make_subplots` to create a figure with multiple traces. The `make_subplots` function will be provided in the execution scope.
    4.  **DATA MANIPULATION:** Perform any necessary data manipulation (like grouping, sorting, calculating correlations, or handling dates with `pd.to_datetime`) within the script.
    5.  **ADVANCED ANALYSIS:** You are capable of complex analysis. If a user asks for "trends," consider generating a time series plot with a moving average. If they ask for "relationships," consider a scatter plot with a regression line.
    6.  **SYNTAX RULES:** Do not use `px.subplots()` or `px.heatmap`. The correct functions are `make_subplots` and `px.imshow`.
    7.  **STYLE:** Use the `template='plotly_dark'` to ensure all charts look professional and match the app's theme.

    **EXAMPLE 1: User asks for a single, complex chart**
    *   REQUEST: 'Show me the correlation heatmap for all numeric columns'
    *   PERFECT OUTPUT:
    ```python
    import pandas as pd
    import plotly.express as px
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
    fig.update_layout(template='plotly_dark', title_x=0.5)
    ```

    **EXAMPLE 2: User asks for multiple views (SUBPLOTS)**
    *   REQUEST: 'Show me daily sales as a bar chart and a 7-day moving average line'
    *   PERFECT OUTPUT:
    ```python
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Prepare the data
    df['SALES_DATE'] = pd.to_datetime(df['SALES_DATE'])
    daily_sales = df.groupby(df['SALES_DATE'].dt.date)['SALES_VALUE'].sum().reset_index()
    daily_sales['7-Day MA'] = daily_sales['SALES_VALUE'].rolling(window=7).mean()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(go.Bar(x=daily_sales['SALES_DATE'], y=daily_sales['SALES_VALUE'], name='Daily Sales'), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_sales['SALES_DATE'], y=daily_sales['7-Day MA'], name='7-Day Moving Average', mode='lines'), secondary_y=True)

    # Style the figure
    fig.update_layout(title_text='Daily Sales and 7-Day Moving Average', template='plotly_dark', title_x=0.5)
    fig.update_yaxes(title_text="Daily Sales", secondary_y=False)
    fig.update_yaxes(title_text="Moving Average", secondary_y=True)
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
if "dashboard_items" not in st.session_state: st.session_state.dashboard_items = []


# --- Main Application ---
st.title("🎨 AI-Powered Dashboard Composer")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'parquet', 'csv', 'txt'], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.session_state.dashboard_items = []

    if st.session_state.df is not None:
        st.success("Data loaded!")
        with st.expander("Data Preview"): st.dataframe(st.session_state.df.head())
    
    st.divider()
    st.header("Actions")
    if st.button("Clear Dashboard"):
        st.session_state.dashboard_items = []
        st.rerun()

# --- Main Content ---
if st.session_state.df is not None:
    st.header("2. Add a Visualization to Your Dashboard")
    
    user_request = st.text_input(
        "Enter your request", 
        placeholder="e.g., 'Show daily sales and a 30-day moving average'",
        label_visibility="collapsed"
    )

    if st.button("Add to Dashboard", key="generate") and user_request:
        with st.spinner("AI is composing your visualization..."):
            python_code = get_python_code(user_request, st.session_state.df.head())
            if python_code:
                st.session_state.dashboard_items.append({
                    "request": user_request,
                    "code": python_code
                })
            else:
                st.error("Could not generate the Python code for your request.")
    
    st.divider()

    if st.session_state.dashboard_items:
        st.header("3. Your Composed Dashboard")
        
        cols = st.columns(2)
        
        for i, item in enumerate(st.session_state.dashboard_items):
            with cols[i % 2]:
                with st.container(border=True):
                    st.subheader(f"Request: \"{item['request']}\"")
                    try:
                        # --- CHANGE 2: Add make_subplots to the scope ---
                        local_scope = {
                            "df": st.session_state.df.copy(), 
                            "pd": pd, 
                            "px": px, 
                            "go": go, 
                            "make_subplots": make_subplots
                        }
                        exec(item['code'], local_scope)
                        
                        fig = local_scope.get('fig')
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Generated code ran, but did not produce a `fig`.")
                        
                        with st.expander("Show Generated Code"):
                            st.code(item['code'], language='python')
                            
                    except Exception as e:
                        st.error("Error executing generated code.")
                        st.code(traceback.format_exc(), language='text')

else:
    st.info("Please upload a data file in the sidebar to get started.")
