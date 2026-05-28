import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import traceback
import tempfile
import sqlite3
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
        background: linear-gradient(135deg, #0f111a 0%, #161824 100%);
    }
    
    /* Sleek card styling for dashboards */
    div[data-testid="stVerticalBlockBorder"] {
        background: rgba(22, 28, 45, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="stVerticalBlockBorder"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.15);
    }
    
    /* Glassmorphic pills for suggestion chips */
    .suggestion-chip {
        display: inline-block;
        padding: 8px 16px;
        margin: 6px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        cursor: pointer;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .suggestion-chip:hover {
        background: rgba(99, 102, 241, 0.3);
        color: #ffffff;
        border-color: rgba(99, 102, 241, 0.6);
        transform: scale(1.03);
    }

    /* Style the main title */
    .main-title {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    /* Input & general widgets */
    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    /* Make code elements look premium */
    code {
        color: #f472b6 !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 6px;
        padding: 2px 6px;
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

# --- Google AI API Configuration & Key Manager ---
st.sidebar.markdown("### 🔑 API Key & Model Configuration")

# Try obtaining key from secrets/env as fallback
default_key = ""
try:
    default_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    default_key = os.getenv("GOOGLE_API_KEY", "")

# Key Input with masking
api_key = st.sidebar.text_input(
    "Google API Key",
    value=st.session_state.get("google_api_key", default_key),
    type="password",
    help="Enter your Google Cloud Gemini API key. It will be stored securely in session state.",
    placeholder="AIzaSy..."
)

# Model Selection
selected_model = st.sidebar.selectbox(
    "Gemini Model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"],
    index=0,
    help="Choose the Gemini model version. gemini-1.5-flash is highly stable and widely accessible."
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
        st.sidebar.success("🟢 API Key Configured & Active")
    except Exception as e:
        st.sidebar.error(f"🔴 API Key Error: {e}")
        st.session_state.api_key_active = False
else:
    st.sidebar.warning("🟡 Please enter a Google API Key")
    st.session_state.api_key_active = False


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

@st.cache_data
def load_data(uploaded_file):
    """Caches data loading and handles a wide range of file formats."""
    try:
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
            if tables:
                return tables[0]
            st.error("No tables found in the uploaded HTML file.")
            return None
        elif name.endswith(('.db', '.sqlite')):
            # Save uploaded bytes to a tempfile to read via sqlite3
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                st.error("No tables found in the SQLite database.")
                conn.close()
                os.unlink(tmp_path)
                return None
            
            # Since this is cached, we load the first table by default
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
        st.code(traceback.format_exc(), language='text')
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
        {df.head(3).to_string()}
        
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
    6. **AESTHETICS:** Style the figure beautifully. Use the template `'plotly_dark'` to match the dark theme. Choose professional color schemes (e.g. indigo `#6366f1`, emerald `#10b981`, fuchsia `#d946ef`, amber `#f59e0b`). Ensure comfortable margins (`margin=dict(l=40, r=40, t=60, b=40)`), enable grid lines, and align the title to the center (`title_x=0.5`).

    **EXAMPLE 1: Heatmaps**
    import pandas as pd
    import plotly.express as px
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap", color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_dark', title_x=0.5, margin=dict(t=60, b=40, l=40, r=40))

    **EXAMPLE 2: Subplots / Secondary Y-Axis**
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df['DATE'] = pd.to_datetime(df['DATE'])
    daily = df.groupby(df['DATE'].dt.date).sum().reset_index()
    daily['MovingAvg'] = daily['VALUE'].rolling(window=7).mean()

    fig = make_subplots(specs=[[{{"secondary_y": True}}]])
    fig.add_trace(go.Bar(x=daily['DATE'], y=daily['VALUE'], name='Daily Value', marker_color='#6366f1'), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily['DATE'], y=daily['MovingAvg'], name='7-Day Average', line=dict(color='#10b981', width=3)), secondary_y=True)

    fig.update_layout(title_text='Daily Metrics & Rolling Averages', template='plotly_dark', title_x=0.5)
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
                    <p>Powered by Google Gemini 2.0</p>
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


# --- Main Web App Interface ---
st.markdown('<h1 class="main-title">🎨 AI-Powered Dashboard Composer</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size:1.1rem; margin-top:-10px; margin-bottom: 25px;'>Upload raw datasets or documents, analyze schemas automatically, and generate complex analytics with natural voice or text requests.</p>", unsafe_allow_html=True)

# --- Sidebar: Data Uploader ---
with st.sidebar:
    st.markdown("### 📊 1. Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload any data format",
        type=['csv', 'xlsx', 'xls', 'parquet', 'txt', 'tsv', 'json', 'jsonl', 'xml', 'html', 'db', 'sqlite', 'pdf', 'png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )
    
    # Process uploaded file
    if uploaded_file:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Reset dashboard items if a new file is uploaded
        if file_key != st.session_state.prev_df_key:
            st.session_state.df = load_data(uploaded_file)
            st.session_state.prev_df_key = file_key
            st.session_state.loaded_filename = uploaded_file.name
            st.session_state.dashboard_items = []
            
            # Generate AI profiling suggestions instantly if API active
            if st.session_state.df is not None:
                st.session_state.suggestions = get_ai_suggestions(st.session_state.df)
            else:
                st.session_state.suggestions = []

    # Display loaded status and dynamic schemas
    if st.session_state.df is not None:
        st.success(f"Loaded: `{st.session_state.loaded_filename}`")
        with st.expander("🔍 Dataset Schema & Columns"):
            st.dataframe(pd.DataFrame({
                "Column Name": st.session_state.df.columns,
                "Type": [str(t) for t in st.session_state.df.dtypes],
                "Null Values": st.session_state.df.isnull().sum().values
            }), use_container_width=True)
            
        with st.expander("📄 Raw Data Preview (First 5 Rows)"):
            st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    st.divider()
    st.markdown("### ⚙️ Reset / Actions")
    if st.button("🧹 Clear Workspace", use_container_width=True):
        st.session_state.dashboard_items = []
        st.session_state.suggestions = []
        st.session_state.df = None
        st.session_state.prev_df_key = None
        st.session_state.loaded_filename = ""
        st.rerun()


# --- Main Workspace ---
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
                    st.session_state.voice_input_value = sug
                    st.rerun()

    st.markdown("### ➕ Add a Visualization")
    
    # Text input for request
    user_request = st.text_input(
        "Enter your query",
        value=st.session_state.voice_input_value,
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
                python_code = get_python_code(user_request, st.session_state.df.head(), st.session_state.selected_model)
                if python_code:
                    st.session_state.dashboard_items.append({
                        "request": user_request,
                        "code": python_code
                    })
                    st.session_state.voice_input_value = "" # Reset
                    st.rerun()
                else:
                    st.error("Could not construct visualization code for the given request.")

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
            # Gather all python scripts into a clean aggregate file
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
            
            # Export standalone responsive HTML
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
    # Large beautiful welcoming landing section when no file is uploaded
    st.markdown("""
    <div style='background: rgba(30, 41, 59, 0.4); border: 1px dashed rgba(255, 255, 255, 0.15); border-radius: 20px; padding: 40px; text-align: center; margin-top: 20px;'>
        <span style='font-size: 3.5rem;'>🚀</span>
        <h2 style='color: #f8fafc; font-weight: 700; margin-top: 15px;'>Welcome to the Next-Gen Dashboard Composer!</h2>
        <p style='color: #94a3b8; font-size:1rem; max-width: 600px; margin: 10px auto 25px auto;'>
            Start by entering a Google API Key and selecting a Gemini model. Then, upload a tabular dataset (Excel, CSV, Parquet, JSON), a SQLite database, or even a PDF report/image of a table. Our AI will automatically profile your schema and help you construct interactive, breathtaking data dashboards in seconds!
        </p>
    </div>
    """, unsafe_allow_html=True)
