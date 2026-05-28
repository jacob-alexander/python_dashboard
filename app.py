import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import tempfile
import sqlite3
import html
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_mic_recorder import mic_recorder
from io import StringIO

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Dashboard Composer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (Aesthetics) ---
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #e0e7ff 100%);
    }
    
    /* Sleek card styling for light mode glassmorphism */
    div[data-testid="stVerticalBlockBorder"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: #1e293b !important;
    }
    
    div[data-testid="stVerticalBlockBorder"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.12);
    }
    
    /* Glassmorphic pills for suggestion chips in light mode */
    .suggestion-chip {
        display: inline-block;
        padding: 8px 16px;
        margin: 6px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.08);
        color: #4f46e5;
        border: 1px solid rgba(99, 102, 241, 0.2);
        cursor: pointer;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .suggestion-chip:hover {
        background: rgba(99, 102, 241, 0.18);
        color: #4338ca;
        border-color: rgba(99, 102, 241, 0.4);
        transform: scale(1.03);
    }

    /* Style the main title with a bright gradient */
    .main-title {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    /* Input & general widgets in light mode */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 12px !important;
        color: #1e293b !important;
    }
    
    /* Make code elements look premium in light mode */
    code {
        color: #be185d !important;
        background-color: rgba(243, 244, 246, 0.9) !important;
        border-radius: 6px;
        padding: 2px 6px;
    }
    
    /* Let sidebar look clean and light */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Session State Initialization ---
if "df" not in st.session_state: st.session_state.df = None
if "dashboard_items" not in st.session_state: st.session_state.dashboard_items = []
if "suggestions" not in st.session_state: st.session_state.suggestions = []
if "voice_input_value" not in st.session_state: st.session_state.voice_input_value = ""
if "loaded_filename" not in st.session_state: st.session_state.loaded_filename = ""
if "prev_df_key" not in st.session_state: st.session_state.prev_df_key = None
if "multi_files" not in st.session_state: st.session_state.multi_files = {} # filename -> DataFrame
if "inferred_relations" not in st.session_state: st.session_state.inferred_relations = None
if "main_query_input" not in st.session_state: st.session_state.main_query_input = ""
if "pending_query" not in st.session_state: st.session_state.pending_query = None
if "custom_advice_output" not in st.session_state: st.session_state.custom_advice_output = ""

# --- Google AI API Configuration & Key Manager ---
st.sidebar.markdown("### 🔑 API Key & Model Configuration")

# Try obtaining key from secrets/env as fallback
system_key = ""
try:
    system_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    system_key = os.getenv("GOOGLE_API_KEY", "")

# Hide the API key input completely if loaded from the system secrets, showing only a secure success badge.
api_key = ""
if system_key:
    st.sidebar.info("🔒 API Key loaded securely from Secrets")
    override_key = st.sidebar.checkbox("Use a custom API Key instead", value=False)
    if override_key:
        api_key = st.sidebar.text_input(
            "Custom Google API Key",
            key="custom_api_key",
            type="password",
            autocomplete="new-password",
            help="Enter your own Google Cloud Gemini API key to override the secure system secret.",
            placeholder="AIzaSy..."
        )
    else:
        api_key = system_key
else:
    # No system key is present, show standard input field
    api_key = st.sidebar.text_input(
        "Google API Key",
        value=st.session_state.get("google_api_key", ""),
        type="password",
        autocomplete="new-password",
        help="Enter your Google Cloud Gemini API key. It will be stored securely in session state.",
        placeholder="AIzaSy..."
    )

# Model Selection (strictly validated models)
selected_model = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-lite-latest", "gemini-flash-latest"],
    index=0,
    help="Choose the Gemini model version. gemini-3.5-flash is the recommended premium model."
)
st.session_state.selected_model = selected_model

# Configure and validate API
api_active = False
if api_key:
    try:
        genai.configure(api_key=api_key)
        st.session_state.google_api_key = api_key
        st.session_state.api_key_active = True
        api_active = True
        is_system = system_key and not (override_key if 'override_key' in locals() else False)
        if is_system:
            st.sidebar.success("🟢 API Key Active (System Secret)")
        else:
            st.sidebar.success("🟢 API Key Active (Custom)")
    except Exception as e:
        st.sidebar.error(f"🔴 API Key Error: {e}")
        st.session_state.api_key_active = False
else:
    st.sidebar.warning("🟡 Please enter a Google API Key")
    st.session_state.api_key_active = False


# --- Advanced Folder Scanning Helpers ---
def scan_directory(folder_path):
    """Scans a local directory for any analytical / database / document file formats."""
    if not folder_path or not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return []
    supported_extensions = ('.csv', '.xlsx', '.xls', '.parquet', '.txt', '.tsv', '.json', '.jsonl', '.xml', '.html', '.db', '.sqlite', '.pdf', '.png', '.jpg', '.jpeg')
    files = []
    try:
        for f in os.listdir(folder_path):
            if f.lower().endswith(supported_extensions):
                files.append(f)
    except Exception as e:
        st.sidebar.error(f"Error scanning directory: {e}")
    return sorted(files)


# --- Advanced Data Loaders & AI Extractor ---
def extract_data_using_gemini(uploaded_file):
    """Sends document/image to Gemini to parse tables/data into a clean CSV format."""
    if not st.session_state.get("api_key_active"):
        st.error("Please configure your Google API Key in the sidebar first to use document intelligence features.")
        return None
        
    with st.spinner("🧙‍♂️ Gemini Vision is parsing your file & extracting tabular data..."):
        try:
            file_bytes = uploaded_file.getvalue()
            mime_type = uploaded_file.type
            
            file_part = {
                "mime_type": mime_type,
                "data": file_bytes
            }
            
            prompt = """
            You are a world-class document understanding and data extraction system.
            Analyze the attached document/image (PDF or image containing tables, lists, or structured records) and extract the primary structured data.
            
            INSTRUCTIONS:
            1. Extract the main data table into STRICT CSV (Comma Separated Values) format.
            2. Do NOT wrap your CSV in markdown code block markers (like ```csv). Output ONLY the raw CSV text.
            3. Ensure columns are separated properly by commas. Use double quotes around values containing commas or special characters.
            4. Make reasonable assumptions for headers if they are missing or unclear.
            5. Return nothing but the clean, parsable CSV stream. No explanations, no markdown wrapper.
            """
            
            model = genai.GenerativeModel(st.session_state.selected_model)
            response = model.generate_content([prompt, file_part])
            
            csv_text = response.text.strip()
            # Clean up potential markdown wrappers
            if csv_text.startswith("```"):
                lines = csv_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                csv_text = "\n".join(lines).strip()
                
            df = pd.read_csv(StringIO(csv_text))
            st.success("🎉 Data successfully extracted from your file using Gemini Multimodal intelligence!")
            return df
        except Exception as e:
            st.error(f"Failed to extract data: {e}")
            st.code(traceback.format_exc(), language='text')
            return None

def load_data(uploaded_file):
    """Handles loading a wide range of file formats from standard uploads or local directory files."""
    try:
        if isinstance(uploaded_file, str):
            # It's a local file path
            name = uploaded_file.lower()
            if name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            elif name.endswith('.parquet'):
                return pd.read_parquet(uploaded_file)
            elif name.endswith(('.txt', '.tsv')):
                return pd.read_csv(uploaded_file, delimiter='\t')
            elif name.endswith('.json'):
                return pd.read_json(uploaded_file)
            elif name.endswith('.jsonl'):
                return pd.read_json(uploaded_file, lines=True)
            elif name.endswith('.xml'):
                return pd.read_xml(uploaded_file)
            elif name.endswith('.html'):
                tables = pd.read_html(uploaded_file)
                if tables: return tables[0]
                return None
            elif name.endswith(('.db', '.sqlite')):
                conn = sqlite3.connect(uploaded_file)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                if not tables:
                    conn.close()
                    return None
                df = pd.read_sql_query(f"SELECT * FROM `{tables[0]}`", conn)
                conn.close()
                return df
            elif name.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                # In folder scanner context, we can read local bytes and wrap in a Mock streamlit uploaded file
                class MockUploadedFile:
                    def __init__(self, path):
                        self.path = path
                        self.name = os.path.basename(path)
                        with open(path, 'rb') as f:
                            self.bytes = f.read()
                        if self.name.lower().endswith('.pdf'): self.type = 'application/pdf'
                        else: self.type = 'image/png'
                    def getvalue(self): return self.bytes
                return extract_data_using_gemini(MockUploadedFile(uploaded_file))
        else:
            # Standard streamlit uploaded file
            name = uploaded_file.name.lower()
            if name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(uploaded_file)
            elif name.endswith('.parquet'):
                return pd.read_parquet(uploaded_file)
            elif name.endswith(('.txt', '.tsv')):
                return pd.read_csv(uploaded_file, delimiter='\t')
            elif name.endswith('.json'):
                return pd.read_json(uploaded_file)
            elif name.endswith('.jsonl'):
                return pd.read_json(uploaded_file, lines=True)
            elif name.endswith('.xml'):
                return pd.read_xml(uploaded_file)
            elif name.endswith('.html'):
                tables = pd.read_html(uploaded_file)
                if tables: return tables[0]
                return None
            elif name.endswith(('.db', '.sqlite')):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                conn = sqlite3.connect(tmp_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                if not tables:
                    conn.close()
                    os.unlink(tmp_path)
                    return None
                df = pd.read_sql_query(f"SELECT * FROM `{tables[0]}`", conn)
                conn.close()
                os.unlink(tmp_path)
                return df
            elif name.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                return extract_data_using_gemini(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# --- AI Data Profiling & Suggestion Generator ---
def get_ai_suggestions(df):
    """Generates 5 tailored business visualization suggestions based on schema."""
    if not st.session_state.get("api_key_active") or df is None:
        return [
            "Show a distribution histogram of all numeric columns",
            "Create a bar chart comparing values across categories",
            "Show correlation heatmap for all variables",
            "Draw a line chart showcasing trends",
            "Generate a scatter plot to identify relationships"
        ]
        
    try:
        schema_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            unique_vals = df[col].nunique()
            schema_info.append(f"- Column: `{col}` | Type: `{dtype}` | Missing: `{missing}` | Unique: `{unique_vals}`")
            
        schema_str = "\n".join(schema_info)
        
        prompt = f"""
        Act as an elite Business Intelligence consultant. You are exploring a dataset with the following schema:
        
        {schema_str}
        
        DataFrame head preview:
        {df.head(15).to_string()}
        
        Task: Provide exactly 5 distinct, highly insightful, and complex charting queries that a business stakeholder would want to ask about this specific dataset to uncover hidden trends or insights.
        
        RULES:
        1. Tailor the queries strictly to the columns present in the schema.
        2. Keep each suggestion concise (under 12 words) and easy to read.
        3. Output EXACTLY a list of 5 numbered items. No other headers, introductory, or concluding remarks.
        """
        
        model = genai.GenerativeModel(st.session_state.selected_model)
        response = model.generate_content(prompt)
        
        suggestions = []
        for line in response.text.strip().splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                clean_line = line.split(".", 1)[-1].strip() if "." in line else line.split("-", 1)[-1].strip()
                suggestions.append(clean_line)
                
        if len(suggestions) >= 5:
            return suggestions[:5]
        return [
            "Show distribution of variables",
            "Compare trends over time",
            "Create a correlation heatmap",
            "Show outlier box plots",
            "Display grouped bar comparison"
        ]
    except Exception:
        return [
            "Show distribution of variables",
            "Compare trends over time",
            "Create a correlation heatmap",
            "Show outlier box plots",
            "Display grouped bar comparison"
        ]


def generate_data_advisory(df, model_name):
    """Generates concrete, numerical business observations, actionable optimization advice, and recommended graph requests."""
    if not st.session_state.get("api_key_active") or df is None:
        return "Configure your Google API Key and import a dataset to receive AI Advisory insights."
        
    try:
        # Build schema summary
        cols_summary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            uniques = df[col].nunique()
            cols_summary.append(f"col '{col}': type={dtype}, nulls={nulls}, unique={uniques}")
            
        summary_str = "\n".join(cols_summary[:15]) # Limit columns to prevent token exhaustion
        preview_str = df.head(15).to_string()
        
        prompt = f"""
        Act as a highly precise, hands-on Data Auditor and Commercial Business Analyst.
        You are looking at a dataset preview showing 15 actual rows of data:
        
        {preview_str}
        
        Schema Summary:
        {summary_str}
        
        Task: Inspect the actual records, columns, and numbers in the 15-row preview to extract highly specific, concrete numerical trends and action points. Generate exactly:
        
        1. **📊 Analytical Inquiries**: 5 to 6 hyper-specific data observations detailing actual changes, drops, spikes, or relationships with numbers, countries, item types, or categories visible in the data (e.g., "sales in country Austria for item type cereal down by 50%", "Clothes down by 75% due to high margin"). Keep them extremely concrete, numeric, and simple.
        2. **💡 Optimization Advisory**: 4 to 5 actionable, numerical business recommendations with specific action values (e.g., "reduce margin to 30% for Meat in Switzerland, you will increase the sales to 100%").
        3. **📈 Suggested Graph Requests**: For each analytical inquiry, suggest the exact natural language query the user should type in the composer to visualize it (e.g., "Compare cereal sales in Austria vs other European countries").
        
        CRITICAL RULES:
        - DO NOT write corporate jargon, buzzwords, or abstract business theory (like 'Fulfillment Velocity', 'consolidated contribution margin', 'consolidated EBITDA', 'capital efficiency').
        - Avoid bullet headings with high-level titles. Just output the concrete observations, recommendations, and graph requests directly as clear, simple sentences.
        - Ensure every observation and advice refers to actual values, countries, products, and specific estimated percentages/numbers.
        - Keep the entire response strictly under 280 words. Write in crisp, clean, direct markdown.
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Advisory generation skipped: {e}"


def get_custom_business_advice(df, user_query, model_name):
    """Formulates highly targeted, concrete, and numerical business advice based on a specific user inquiry and 15-row preview."""
    try:
        cols_summary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = df[col].isnull().sum()
            uniques = df[col].nunique()
            cols_summary.append(f"col '{col}': type={dtype}, nulls={nulls}, unique={uniques}")
            
        summary_str = "\n".join(cols_summary[:15])
        preview_str = df.head(15).to_string()
        
        prompt = f"""
        Act as an elite Data Auditor and Strategic Business Advisor.
        You are looking at a dataset preview showing 15 actual rows of data:
        
        {preview_str}
        
        Schema Summary:
        {summary_str}
        
        The user has asked this complex business/optimization question:
        "{user_query}"
        
        Task: Formulate a highly targeted, concrete, and numerical business advice answer.
        
        CRITICAL RULES:
        1. Base your advice directly on the data values, columns, and records in the 15-row preview.
        2. Keep your answer highly specific and action-oriented. Provide exact numerical recommendations (e.g., "increase price of X by 10% in region Y to capture Z% higher profit").
        3. STRICTLY AVOID corporate jargon, abstract fluff, or data quality/cleaning comments.
        4. Keep the entire response under 150 words. Write in simple, direct markdown bullet points.
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate advice: {e}"


# --- Core AI Code Generator ---
def get_python_code(user_request, df_head, model_name):
    """Uses Gemini to generate custom, executable Plotly Python code to compose dashboards."""
    prompt = f"""
    Act as a world-class Python data scientist generating interactive, professional data visualizations with Plotly.
    Your task is to generate Python code that creates a single Plotly Figure object assigned to a variable named `fig`.

    User Request: "{user_request}"

    DataFrame head for context (the full DataFrame is available inside the variable `df`):
    ```
    {df_head.to_string()}
    ```

    **CRITICAL CODE RULES:**
    1. Your output MUST BE a single block of highly clean, executable Python code. No markdown backticks, no explanations, no text wrappers. Just Python code.
    2. Define, manipulate, and structure all data inside the script as needed. For example, use `pd.to_datetime` for timestamps, and `.groupby` or `.rolling` for aggregations.
    3. The code's final output MUST assign a fully customized Plotly Figure object to the variable `fig`.
    4. **SUBPLOTS:** If the user asks for multiple charts, grids, or multi-metric views, you MUST import and use `from plotly.subplots import make_subplots` to return a unified `fig`.
    5. **SYNTAX RULES:** Never use `px.subplots()` or `px.heatmap()`. Use `make_subplots` and `px.imshow()` instead.
    6. **AESTHETICS:** Style the figure beautifully. Use the template `'plotly_white'` to match the light theme. Choose professional color schemes (e.g. indigo `#4f46e5`, emerald `#059669`, fuchsia `#db2777`, amber `#d97706`). Ensure comfortable margins (`margin=dict(l=40, r=40, t=60, b=40)`), enable grid lines, and align the title to the center (`title_x=0.5`).

    **EXAMPLE 1: Heatmaps**
    import pandas as pd
    import plotly.express as px
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', title_x=0.5, margin=dict(t=60, b=40, l=40, r=40))

    **EXAMPLE 2: Subplots / Secondary Y-Axis**
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df['DATE'] = pd.to_datetime(df['DATE'])
    daily = df.groupby(df['DATE'].dt.date).sum().reset_index()
    daily['MovingAvg'] = daily['VALUE'].rolling(window=7).mean()

    fig = make_subplots(specs=[[{{"secondary_y": True}}]])
    fig.add_trace(go.Bar(x=daily['DATE'], y=daily['VALUE'], name='Daily Value', marker_color='#4f46e5'), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily['DATE'], y=daily['MovingAvg'], name='7-Day Average', line=dict(color='#059669', width=3)), secondary_y=True)

    fig.update_layout(title_text='Daily Metrics & Rolling Averages', template='plotly_white', title_x=0.5)
    fig.update_yaxes(title_text="Value", secondary_y=False)
    fig.update_yaxes(title_text="Moving Average", secondary_y=True)

    Now, generate the Plotly Python script.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        code = response.text.strip()
        
        # Strip markdown syntax safely
        if code.startswith("```python"):
            code = code.removeprefix("```python")
        elif code.startswith("```"):
            code = code.removeprefix("```")
        if code.endswith("```"):
            code = code.removesuffix("```")
            
        return code.strip()
    except Exception as e:
        st.error(f"Error generating Python code: {e}")
        return None


# --- Standalone Responsive HTML Exporter ---
def generate_html_export(dashboard_items, df):
    """Compiles all active widgets and constructs a beautiful standalone HTML file with TailwindCSS grid."""
    html_charts = ""
    for idx, item in enumerate(dashboard_items):
        try:
            local_scope = {
                "df": df.copy(),
                "pd": pd,
                "px": px,
                "go": go,
                "make_subplots": make_subplots
            }
            exec(item['code'], local_scope)
            fig = local_scope.get('fig')
            if fig is not None:
                chart_inner = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='500px')
                html_charts += f"""
                <div class="bg-slate-800/40 backdrop-blur-lg border border-slate-700/60 rounded-2xl p-6 shadow-2xl hover:border-indigo-500/30 transition-all duration-300">
                    <div class="mb-4">
                        <span class="text-xs uppercase font-semibold text-indigo-400 tracking-wider">Visualization #{idx + 1}</span>
                        <h3 class="text-xl font-bold text-slate-100 mt-1">"{item['request']}"</h3>
                    </div>
                    <div class="w-full h-[500px]">{chart_inner}</div>
                </div>
                """
        except Exception:
            continue
            
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Composed AI Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Outfit', sans-serif;
                background: linear-gradient(135deg, #090d16 0%, #111424 100%);
            }}
        </style>
    </head>
    <body class="min-h-screen text-slate-100 p-6 md:p-12">
        <div class="max-w-7xl mx-auto">
            <header class="mb-12 text-center md:text-left flex flex-col md:flex-row justify-between items-center border-b border-slate-800/60 pb-8">
                <div>
                    <h1 class="text-4xl font-extrabold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                        🎨 AI Composed Dashboard
                    </h1>
                    <p class="text-slate-400 mt-2">Standalone, interactive analytical dashboard compiled dynamically via Plotly</p>
                </div>
                <div class="mt-4 md:mt-0 text-sm text-slate-500 text-right">
                    <p>Generated: 2026-05-28</p>
                    <p>Powered by Google Gemini</p>
                </div>
            </header>
            
            <div class="grid grid-cols-1 gap-8">
                {html_charts}
            </div>
            
            <footer class="mt-20 text-center text-xs text-slate-600 border-t border-slate-900 pt-8">
                &copy; 2026 AI Dashboard Composer. All rights reserved. Created with &hearts;
            </footer>
        </div>
    </body>
    </html>
    """
    return full_html


# Render Single Chart Card
def render_dashboard_card(item, idx):
    """Renders an interactive card containing a Plotly chart, custom error handler, code editor, and re-order widgets."""
    with st.container(border=True):
        card_header_cols = st.columns([10, 2])
        with card_header_cols[0]:
            st.markdown(f"**Request:** *\"{item['request']}\"*")
        with card_header_cols[1]:
            # Delete button
            if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
                st.session_state.dashboard_items.pop(idx)
                st.rerun()

        # Try to execute Plotly code
        try:
            local_scope = {
                "df": st.session_state.df.copy(),
                "pd": pd,
                "px": px,
                "go": go,
                "make_subplots": make_subplots
            }
            # Execute generated code block
            exec(item['code'], local_scope)
            fig = local_scope.get('fig')
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"chart_widget_{idx}")
            else:
                st.warning("Visualization executed successfully, but no `fig` object was assigned.")
        except Exception as e:
            st.error("Error executing generated visualization code:")
            st.code(traceback.format_exc(), language='text')

        # Code Editor & Manipulation Dropdowns
        with st.expander("🛠️ Edit Python Code / Advanced Options"):
            # Live Interactive Code Editor
            updated_code = st.text_area(
                "Source Python Code (Plotly)",
                value=item['code'],
                height=220,
                key=f"editor_{idx}"
            )
            
            editor_cols = st.columns([4, 4, 4])
            with editor_cols[0]:
                if st.button("🔄 Update Chart", key=f"update_{idx}"):
                    st.session_state.dashboard_items[idx]['code'] = updated_code
                    st.rerun()
            with editor_cols[1]:
                # Move item up
                if idx > 0:
                    if st.button("⬆️ Move Up", key=f"up_{idx}"):
                        items = st.session_state.dashboard_items
                        items[idx], items[idx-1] = items[idx-1], items[idx]
                        st.session_state.dashboard_items = items
                        st.rerun()
            with editor_cols[2]:
                # Move item down
                if idx < len(st.session_state.dashboard_items) - 1:
                    if st.button("⬇️ Move Down", key=f"down_{idx}"):
                        items = st.session_state.dashboard_items
                        items[idx], items[idx+1] = items[idx+1], items[idx]
                        st.session_state.dashboard_items = items
                        st.rerun()


# --- AI Join Studio Helper Functions ---
def infer_relationships_with_gemini(multi_files, model_name):
    """Sends loaded file schemas to Gemini to automatically infer relationships and generate Mermaid ER Diagram."""
    if not multi_files or not st.session_state.get("api_key_active"):
        return None
        
    with st.spinner("🧠 AI Database Architect is analyzing table column relationships..."):
        try:
            schemas = []
            for name, df in multi_files.items():
                col_types = [f"{col} ({str(df[col].dtype)})" for col in df.columns]
                schemas.append(f"Table: `{name}`\nColumns: {', '.join(col_types)}\nHead Preview:\n{df.head(15).to_string()}")
                
            schemas_str = "\n\n---\n\n".join(schemas)
            
            prompt = f"""
            Act as an elite database architect and relationship modeling expert.
            Analyze the following table schemas and previews, and identify relationships/join possibilities between them.
            
            {schemas_str}
            
            INSTRUCTIONS:
            1. Identify any clear join/foreign key relationships (e.g., column 'customer_id' in 'orders.csv' corresponds to 'id' in 'customers.csv').
            2. Propose the optimal way to join these tables.
            3. Format your response into two distinct parts:
               - Part 1: A brief, friendly plain-text explanation of the suggested relationships (under 100 words).
               - Part 2: A Mermaid Entity Relationship (ER) diagram showing the tables and their relations (wrapped inside a markdown ```mermaid code block). Keep the Mermaid diagram syntax extremely standard and correct.
            
            Example of Part 2:
            ```mermaid
            erDiagram
                ORDERS ||--o{{ CUSTOMERS : "joins on customer_id = id"
            ```
            
            Output ONLY these two parts, clearly labeled as '### 🔗 Inferred Relationships' and '### 📐 Entity-Relationship Diagram'.
            """
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Could not infer relationships: {e}"

def extract_mermaid_code(inferred_text):
    """Parses out the raw Mermaid diagram text block from the AI's response."""
    if not inferred_text: return None
    import re
    match = re.search(r'```mermaid\s*(.*?)\s*```', inferred_text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return None

def render_mermaid(mermaid_code):
    """Embeds the official Mermaid JS engine and beautifully renders the ER diagram in Streamlit."""
    html_content = f"""
    <div class="mermaid" style="background: transparent; color: white; display: flex; justify-content: center; align-items: center; min-height: 250px;">
    {mermaid_code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'loose' }});
    </script>
    """
    components.html(html_content, height=350, scrolling=True)


# --- Main Web App Interface ---
st.markdown('<h1 class="main-title">🎨 AI-Powered Dashboard Composer</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size:1.1rem; margin-top:-10px; margin-bottom: 25px;'>Scan local folders first, import/join datasets, and compose interactive Plotly dashboards with voice or text advice.</p>", unsafe_allow_html=True)

# Define Tabs (Data & AI Join Studio comes first!)
active_tabs = st.tabs(["📁 Data & AI Join Studio", "🚀 Visualization Dashboard"])

# --- Sidebar: Shared directory configuration ---
with st.sidebar:
    st.markdown("### 📁 1. Active Workspace Directory")
    default_dir = os.path.abspath(".")
    folder_path = st.text_input(
        "Local Folder Directory Path",
        value=st.session_state.get("folder_path", default_dir),
        placeholder="/home/user/my_dataset_folder",
        help="Input a local directory path to scan files dynamically."
    )
    st.session_state.folder_path = folder_path
    
    st.divider()
    st.markdown("### ⚙️ Workspace Actions")
    if st.button("🧹 Clear Workspace", use_container_width=True):
        st.session_state.dashboard_items = []
        st.session_state.suggestions = []
        st.session_state.df = None
        st.session_state.prev_df_key = None
        st.session_state.loaded_filename = ""
        st.session_state.multi_files = {}
        st.session_state.inferred_relations = None
        st.session_state.custom_advice_output = ""
        st.rerun()


# ==========================================
# TAB 1: DATA & AI JOIN STUDIO (Index 0)
# ==========================================
with active_tabs[0]:
    st.markdown("### 📂 Load Datasets from Directory")
    st.markdown("<p style='color: #64748b; font-size:0.9rem; margin-top:-8px;'>Scan your active local folder, select multiple files to load, union similar datasets automatically, or build complex joins interactively.</p>", unsafe_allow_html=True)
    
    cols = st.columns([8, 4])
    
    with cols[0]:
        st.markdown(f"**Path:** `{st.session_state.folder_path}`")
        found_files = scan_directory(st.session_state.folder_path)
        
        if found_files:
            st.write("Select files to import into active memory:")
            selected_files = []
            
            # Render file selectors in columns
            file_grid_cols = st.columns(3)
            for idx, filename in enumerate(found_files):
                with file_grid_cols[idx % 3]:
                    checkbox_key = f"file_check_{filename}"
                    is_checked = st.checkbox(f"📄 {filename}", key=checkbox_key)
                    if is_checked:
                        selected_files.append(filename)
            
            # Load selected files
            st.write("") # Spacer
            if st.button("📥 Import Checked Files", use_container_width=True):
                if not selected_files:
                    st.warning("Please check at least one file to import.")
                else:
                    with st.spinner("Loading selected files into workspace memory..."):
                        st.session_state.multi_files = {}
                        for fn in selected_files:
                            full_path = os.path.join(st.session_state.folder_path, fn)
                            loaded_df = load_data(full_path)
                            if loaded_df is not None:
                                st.session_state.multi_files[fn] = loaded_df
                        st.session_state.inferred_relations = None # Reset
                        
                        # Automatically activate the first successfully loaded file!
                        if st.session_state.multi_files:
                            first_file = list(st.session_state.multi_files.keys())[0]
                            st.session_state.df = st.session_state.multi_files[first_file]
                            st.session_state.loaded_filename = first_file
                            st.session_state.suggestions = get_ai_suggestions(st.session_state.df)
                            st.session_state.data_advisory = generate_data_advisory(st.session_state.df, st.session_state.selected_model)
                            st.session_state.custom_advice_output = ""
                            st.success(f"Successfully loaded {len(st.session_state.multi_files)} files! `{first_file}` is set as your active dataset.")
                        else:
                            st.session_state.df = None
                        st.rerun()
        else:
            st.info("No supported files found in this directory. Paste a valid local folder path in the sidebar to scan.")

    with cols[1]:
        with st.container(border=True):
            st.markdown("#### 🧠 Workspace Datasets in Memory")
            if st.session_state.multi_files:
                # If there's only one file in memory, automatically set it as active if st.session_state.df is None
                if len(st.session_state.multi_files) == 1 and st.session_state.df is None:
                    only_file = list(st.session_state.multi_files.keys())[0]
                    st.session_state.df = st.session_state.multi_files[only_file]
                    st.session_state.loaded_filename = only_file
                    st.session_state.suggestions = get_ai_suggestions(st.session_state.df)
                    st.session_state.data_advisory = generate_data_advisory(st.session_state.df, st.session_state.selected_model)
                
                for fn, df in st.session_state.multi_files.items():
                    card_cols = st.columns([7, 5])
                    with card_cols[0]:
                        st.write(f"📄 **`{fn}`**\n`{df.shape[0]}x{df.shape[1]}`")
                    with card_cols[1]:
                        if st.button("📊 Set Active", key=f"set_active_{fn}", use_container_width=True):
                            st.session_state.df = df
                            st.session_state.loaded_filename = fn
                            st.session_state.dashboard_items = [] # Reset dashboard
                            st.session_state.suggestions = get_ai_suggestions(df)
                            st.session_state.data_advisory = generate_data_advisory(df, st.session_state.selected_model)
                            st.session_state.custom_advice_output = ""
                            st.success(f"`{fn}` is now the active dataset!")
                            st.rerun()
            else:
                st.write("No files currently loaded. Select files on the left and click 'Import Checked Files'.")

    # Smart Union (Vertical Concatenation) section
    if len(st.session_state.multi_files) >= 2:
        st.divider()
        st.markdown("### 🧬 Smart Unions (Schema Concatenation)")
        st.markdown("<p style='color: #64748b; font-size:0.9rem; margin-top:-8px;'>Automatically detects loaded files with identical schemas and unions them vertically.</p>", unsafe_allow_html=True)
        
        schema_groups = {}
        for fn, df in st.session_state.multi_files.items():
            cols_signature = tuple(sorted(df.columns))
            if cols_signature not in schema_groups:
                schema_groups[cols_signature] = []
            schema_groups[cols_signature].append(fn)
            
        unionable_groups = [g for g in schema_groups.values() if len(g) >= 2]
        
        if unionable_groups:
            for g_idx, group in enumerate(unionable_groups):
                st.info(f"✨ Found identical schemas in: {', '.join([f'`{f}`' for f in group])}")
                new_union_name = st.text_input(f"Union dataset name (optional)", value=f"union_dataset_{g_idx+1}.csv", key=f"union_name_{g_idx}")
                
                if st.button(f"🧬 Combine {len(group)} Similar Files", key=f"union_btn_{g_idx}"):
                    try:
                        dfs_to_concat = [st.session_state.multi_files[f] for f in group]
                        merged_df = pd.concat(dfs_to_concat, ignore_index=True)
                        st.session_state.multi_files[new_union_name] = merged_df
                        st.success(f"Combined successfully into memory as `{new_union_name}` ({merged_df.shape[0]} rows)!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to union files: {e}")
        else:
            st.write("No files with identical column signatures detected in memory. Standard table joining is available below.")

    # Relationship and Joining Section
    if len(st.session_state.multi_files) >= 2:
        st.divider()
        st.markdown("### 🔗 AI Join & Relationship Studio")
        
        join_studio_cols = st.columns([6, 6])
        
        with join_studio_cols[0]:
            st.markdown("#### 1. AI relationship Analysis")
            if st.button("🕵️‍♂️ Auto-Infer Relationships with Gemini"):
                st.session_state.inferred_relations = infer_relationships_with_gemini(
                    st.session_state.multi_files,
                    st.session_state.selected_model
                )
                st.rerun()
                
            if st.session_state.inferred_relations:
                st.markdown(st.session_state.inferred_relations)
                
                mermaid_code = extract_mermaid_code(st.session_state.inferred_relations)
                if mermaid_code:
                    with st.expander("📐 Interactive Entity-Relationship Layout", expanded=True):
                        render_mermaid(mermaid_code)

        with join_studio_cols[1]:
            with st.container(border=True):
                st.markdown("#### 🛠️ Custom Interactive Join Configurator")
                st.markdown("<p style='color: #64748b; font-size:0.85rem;'>Configure left and right tables, select the exact keys to join on, and execute the merge.</p>", unsafe_allow_html=True)
                
                filenames = list(st.session_state.multi_files.keys())
                
                left_table_name = st.selectbox("Left Table (Primary)", filenames, index=0)
                right_table_name = st.selectbox("Right Table (Foreign)", filenames, index=min(1, len(filenames)-1))
                
                left_df = st.session_state.multi_files[left_table_name]
                right_df = st.session_state.multi_files[right_table_name]
                
                left_key = st.selectbox(f"Select Join Column from '{left_table_name}'", left_df.columns)
                right_key = st.selectbox(f"Select Join Column from '{right_table_name}'", right_df.columns)
                
                join_type = st.selectbox(
                    "Join / Merge Type",
                    ["inner", "left", "right", "outer"],
                    index=1,
                    format_func=lambda x: f"{x.upper()} Join"
                )
                
                st.divider()
                if st.button("🚀 Execute Custom Join & Set as Active Data", use_container_width=True):
                    try:
                        merged_df = pd.merge(
                            left_df,
                            right_df,
                            left_on=left_key,
                            right_on=right_key,
                            how=join_type
                        )
                        st.session_state.df = merged_df
                        st.session_state.loaded_filename = f"Join: {left_table_name} [{left_key}] = {right_table_name} [{right_key}]"
                        st.session_state.dashboard_items = [] # Reset dashboard
                        st.session_state.suggestions = get_ai_suggestions(merged_df)
                        st.session_state.data_advisory = generate_data_advisory(merged_df, st.session_state.selected_model)
                        st.session_state.custom_advice_output = ""
                        
                        st.success(f"Tables successfully joined! Resulting dataset has `{merged_df.shape[0]}` rows and `{merged_df.shape[1]}` columns. It is now loaded as your active dataset in Tab 2!")
                    except Exception as ex:
                        st.error(f"Error performing join: {ex}")


# ==========================================
# TAB 2: VISUALIZATION DASHBOARD (Index 1)
# ==========================================
with active_tabs[1]:
    # Apply pending updates to main_query_input before widgets are instantiated
    if st.session_state.get("pending_query") is not None:
        st.session_state.main_query_input = st.session_state.pending_query
        st.session_state.pending_query = None

    if st.session_state.df is not None:
        # Render profiling suggestions if available
        if st.session_state.suggestions:
            st.markdown("### 💡 Recommended Visualizations")
            st.markdown("<p style='color: #64748b; font-size:0.9rem; margin-top:-8px;'>Click on any suggested query below to pre-fill the charting input.</p>", unsafe_allow_html=True)
            
            # Render suggestion chips as clickable items
            cols = st.columns(len(st.session_state.suggestions))
            for idx, sug in enumerate(st.session_state.suggestions):
                with cols[idx]:
                    if st.button(sug, key=f"sug_{idx}", use_container_width=True):
                        st.session_state.pending_query = sug
                        st.rerun()

        st.markdown("### ➕ Add a Visualization")
        
        # Text input for request (fully resolved state key, no conflicts!)
        user_request = st.text_input(
            "Enter your query",
            placeholder="e.g., 'Show daily sales as a bar chart and the 14-day rolling average as a trendline'",
            key="main_query_input",
            label_visibility="collapsed"
        )
        
        # Layout choice & control buttons
        input_cols = st.columns([2, 2, 8])
        with input_cols[0]:
            add_btn = st.button("🚀 Generate Chart", use_container_width=True)
        with input_cols[1]:
            # Mic recorder for voice commands
            audio_data = mic_recorder(
                start_prompt="🎤 Voice Command",
                stop_prompt="🛑 Stop Recording",
                key="dashboard_mic",
                use_container_width=True
            )

        # Handle voice transcribing
        if audio_data:
            if not st.session_state.get("api_key_active"):
                st.error("Please configure your Google API Key to transcribe voice commands.")
            else:
                with st.spinner("🎙️ Transcribing voice command..."):
                    try:
                        audio_bytes = audio_data['bytes']
                        audio_part = {
                            "mime_type": "audio/wav",
                            "data": audio_bytes
                        }
                        
                        transcribe_prompt = """
                        You are a voice commands transcribing service. Transcribe the user's spoken analytics request.
                        Output ONLY the raw transcription text, no punctuation wrappers, no descriptions, and no edits.
                        """
                        
                        model = genai.GenerativeModel(st.session_state.selected_model)
                        response = model.generate_content([transcribe_prompt, audio_part])
                        text = response.text.strip()
                        
                        if text:
                            st.session_state.voice_input_value = text
                            st.session_state.pending_query = text
                            st.success(f"Transcribed command: \"{text}\"")
                            st.rerun()
                    except Exception as ex:
                        st.error(f"Voice transcription failed: {ex}")

        # Process chart generation
        if add_btn and user_request:
            if not st.session_state.get("api_key_active"):
                st.error("Please configure your Google API Key in the sidebar first to generate visualizations.")
            else:
                with st.spinner("✨ Generating advanced Plotly code..."):
                    python_code = get_python_code(user_request, st.session_state.df.head(15), st.session_state.selected_model)
                    if python_code:
                        st.session_state.dashboard_items.append({
                            "request": user_request,
                            "code": python_code
                        })
                        st.session_state.pending_query = "" # Reset
                        st.rerun()
                    else:
                        st.error("Could not construct visualization code for the given request.")

        # AI Data Advisory panel
        if st.session_state.get("data_advisory"):
            st.write("") # Spacer
            with st.expander("🧠 AI Data Advisory & Optimization Insights", expanded=True):
                st.markdown(st.session_state.data_advisory)

        # Active Dataset Preview panel
        st.write("") # Spacer
        with st.expander("📄 Active Dataset Preview (First 15 Rows)", expanded=False):
            st.dataframe(st.session_state.df.head(15), use_container_width=True)

        # Interactive "Ask AI Data Advisor & Get Advice" panel
        st.write("") # Spacer
        st.divider()
        st.markdown("### 💬 Ask AI Data Advisor & Get Business Advice")
        st.markdown("<p style='color: #64748b; font-size:0.9rem; margin-top:-8px;'>Ask complex business questions about your columns and trends, and get concrete, numeric optimization advice.</p>", unsafe_allow_html=True)
        
        advisor_query = st.text_input(
            "Ask a business question",
            placeholder="e.g., 'Why are our profit margins in Europe lower than other regions and how do we fix it?'",
            key="advisor_query_input",
            label_visibility="collapsed"
        )
        
        if st.button("🧠 Get Business Advice", key="get_advice_btn") and advisor_query:
            with st.spinner("AI Data Advisor is auditing your data and formulating strategic advice..."):
                advice_result = get_custom_business_advice(st.session_state.df, advisor_query, st.session_state.selected_model)
                if advice_result:
                    st.session_state.custom_advice_output = advice_result
                    st.rerun()
                    
        if st.session_state.get("custom_advice_output"):
            st.info("💡 **AI Data Advisor Strategic Advice:**")
            st.markdown(st.session_state.custom_advice_output)
 
        # Render Dashboard Workspace
        if st.session_state.dashboard_items:
            st.divider()
            
            # Dashboard controls and header
            st.markdown("### 🖥️ Active Dashboard Workspace")
            
            ctrl_cols = st.columns([3, 5, 4])
            with ctrl_cols[0]:
                layout_style = st.selectbox(
                    "Dashboard Layout Layout",
                    ["Grid View (2-Column)", "Single Column (Large)", "Tabbed Workspace"],
                    index=0
                )
            
            # Combined download buttons
            with ctrl_cols[2]:
                st.write("") # spacing
                st.write("") # spacing
                combined_script = "# Composed Dashboard Python Script\n"
                combined_script += "import pandas as pd\nimport plotly.express as px\nimport plotly.graph_objects as go\nfrom plotly.subplots import make_subplots\n\n"
                combined_script += "# Load your data here:\n# df = pd.read_csv('your_file.csv')\n\n"
                for idx, item in enumerate(st.session_state.dashboard_items):
                    combined_script += f"# --- Visualization {idx + 1} (Request: '{item['request']}') ---\n"
                    combined_script += f"{item['code']}\nfig.show()\n\n"
                
                st.download_button(
                    "🐍 Download Python Code",
                    combined_script,
                    file_name="dashboard_source.py",
                    mime="text/plain",
                    use_container_width=True
                )
                
                html_export_bytes = generate_html_export(st.session_state.dashboard_items, st.session_state.df)
                st.download_button(
                    "🌐 Download Standalone HTML Dashboard",
                    html_export_bytes,
                    file_name="interactive_dashboard.html",
                    mime="text/html",
                    use_container_width=True
                )

            st.write("") # vertical gap

            # Render elements depending on layout choice
            items = st.session_state.dashboard_items
            
            if layout_style == "Tabbed Workspace":
                tab_names = [f"📊 Chart {i+1}: {item['request'][:35]}..." for i, item in enumerate(items)]
                tabs = st.tabs(tab_names)
                
                for i, item in enumerate(items):
                    with tabs[i]:
                        render_dashboard_card(item, i)
                        
            elif layout_style == "Single Column (Large)":
                for i, item in enumerate(items):
                    render_dashboard_card(item, i)
                    st.write("") # Spacer
                    
            else: # Grid View (2-Column)
                cols = st.columns(2)
                for i, item in enumerate(items):
                    with cols[i % 2]:
                        render_dashboard_card(item, i)
                        st.write("") # Spacer

    else:
        st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 40px; text-align: center; margin-top: 20px;'>
            <span style='font-size: 3.5rem;'>🚀</span>
            <h2 style='color: #f8fafc; font-weight: 700; margin-top: 15px;'>Your Active Dashboard is Empty</h2>
            <p style='color: #94a3b8; font-size:1rem; max-width: 600px; margin: 10px auto 25px auto;'>
                Before you can generate beautiful interactive charts, you need an active dataset. 
                Go to the first tab (<b>📁 Data & AI Join Studio</b>), scan your local folder, load your datasets, and combine them. 
                Once loaded or joined, the combined data will appear here automatically!
            </p>
        </div>
        """, unsafe_allow_html=True)
