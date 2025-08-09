import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_mic_recorder import mic_recorder
import io

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
    st.error("Google API Key not found...")
    st.stop()


# --- Helper Functions ---
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

def get_python_code(user_request, df_head):
    prompt = f"""
    Act as a world-class Python data scientist creating advanced data visualizations with Plotly.
    Your task is to generate a Python script to create EITHER a Plotly Figure OR a pandas DataFrame.

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
        - For complex requests with multiple views, you MUST use `plotly.subplots.make_subplots`.
        - Do NOT use `px.subplots()` or `px.heatmap`. The correct functions are `make_subplots` and `px.imshow`.
        - Use the `template='plotly_dark'`.
    4.  **DATA RULES (if you create a `result_df`):**
        - The final output in the script should be the creation of the `result_df` DataFrame.
    5.  **GENERAL RULES:**
        - Always perform necessary data manipulation (grouping, sorting, date conversion with `pd.to_datetime`, etc.) inside the script.
        - For high-category charts, summarize to the "Top 15" for readability.

    **EXAMPLE 1: User asks for a table**
    *   REQUEST: 'total sales by store in a table'
    *   PERFECT OUTPUT:
    ```python
    import pandas as pd
    result_df = df.groupby('LOCATION')['SALES_VALUE'].sum().reset_index()
    result_df = result_df.rename(columns={{"LOCATION": "Store Location", "SALES_VALUE": "Total Sales"}})
    result_df = result_df.sort_values(by='Total Sales', ascending=False)
    ```

    **EXAMPLE 2: User asks for a complex chart (SUBPLOTS)**
    *   REQUEST: 'Show me daily sales as a bar chart and a 7-day moving average line'
    *   PERFECT OUTPUT:
    ```python
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df['SALES_DATE'] = pd.to_datetime(df['SALES_DATE'])
    daily_sales = df.groupby(df['SALES_DATE'].dt.date)['SALES_VALUE'].sum().reset_index()
    daily_sales['7-Day MA'] = daily_sales['SALES_VALUE'].rolling(window=7).mean()

    fig = make_subplots(specs=[[{{'secondary_y': True}}]])
    fig.add_trace(go.Bar(x=daily_sales['SALES_DATE'], y=daily_sales['SALES_VALUE'], name='Daily Sales'), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_sales['SALES_DATE'], y=daily_sales['7-Day MA'], name='7-Day Moving Average', mode='lines'), secondary_y=True)
    
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

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Session State Initialization ---
if "df" not in st.session_state: st.session_state.df = None
if "dashboard_items" not in st.session_state: st.session_state.dashboard_items = []

# --- Main Application ---
st.title("🎨 AI-Powered Dashboard Composer")

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

if st.session_state.df is not None:
    st.header("2. Add an Item to Your Dashboard")
    
    user_request = st.text_input(
        "Enter your request", 
        placeholder="e.g., 'Show daily sales and a 30-day moving average'",
        label_visibility="collapsed"
    )

    if st.button("Add to Dashboard", key="generate") and user_request:
        with st.spinner("AI is composing your output..."):
            python_code = get_python_code(user_request, st.session_state.df.head())
            if python_code:
                st.session_state.dashboard_items.append({
                    "request": user_request,
                    "code": python_code
                })
    
    st.divider()

    if st.session_state.dashboard_items:
        st.header("3. Your Composed Dashboard")
        
        cols = st.columns(2)
        
        for i, item in enumerate(st.session_state.dashboard_items):
            with cols[i % 2]:
                with st.container(border=True, key=f"item_{i}"):
                    st.subheader(f"Request: \"{item['request']}\"")
                    try:
                        local_scope = {
                            "df": st.session_state.df.copy(), 
                            "pd": pd, "px": px, "go": go, 
                            "make_subplots": make_subplots
                        }
                        exec(item['code'], local_scope)
                        
                        fig = local_scope.get('fig')
                        result_df = local_scope.get('result_df')

                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        elif result_df is not None:
                            height = (len(result_df) + 1) * 35 + 3
                            st.dataframe(result_df, height=height, use_container_width=True)
                        else:
                            st.warning("Generated code ran, but did not produce a `fig` or a `result_df`.")

                        st.divider()
                        dl_cols = st.columns(3)
                        with dl_cols[0]:
                            if fig is not None:
                                img_buffer = io.BytesIO()
                                # Use the orca engine
                                fig.write_image(img_buffer, format="png", scale=2, engine="orca")
                                st.download_button(
                                    label="Download PNG",
                                    data=img_buffer.getvalue(),
                                    file_name=f"chart_{i+1}.png",
                                    mime="image/png"
                                )
                        with dl_cols[1]:
                            if result_df is not None:
                                csv_data = convert_df_to_csv(result_df)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv_data,
                                    file_name=f"data_{i+1}.csv",
                                    mime="text/csv"
                                )
                        with dl_cols[2]:
                             with st.expander("Show Code"):
                                st.code(item['code'], language='python')
                            
                    except Exception as e:
                        st.error("Error executing generated code.")
                        st.code(traceback.format_exc(), language='text')

else:
    st.info("Please upload a data file in the sidebar to get started.")
