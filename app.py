import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Page and App Configuration ---
st.set_page_config(
    page_title="Data Viewer and Downloader",
    page_icon="📊",
    layout="wide"
)

st.title("Employee Data Analysis")

# --- Helper Functions ---

@st.cache_data
def create_dummy_data():
    """
    This function generates a sample DataFrame to simulate real data.
    In a real application, you would replace this with your data loading logic
    (e.g., pd.read_csv, pd.read_sql, etc.).
    """
    locations = [
        "WHITE TOWER COMPANY", "SALMIYAH STORE", "AL SHARQ STORE", 
        "BOULEVARD STORE", "MANGAF STORE", "HAWALLY STORE", 
        "MISHREF STORE", "SHUWAIKH STORE", "AL KOUT SOUQ STORE", 
        "JAHRA STORE", "NEW DHAJEEJ STORE", "ADAILIYA STORE", 
        "SALWA STORE", "JABRIYA STORE", "SULAIBIYA STORE", 
        "FAHAHEEL STORE", "PROMENADE STORE", "AL KOUT MALL STORE", 
        "AHMADI STORE"
    ]
    
    # Generate a different number of employees for each location to make it realistic
    data = []
    for i, loc in enumerate(locations):
        # Assign more employees to the first few locations
        num_employees = np.random.randint(150 - i*6, 160 - i*6)
        for emp_id in range(num_employees):
            data.append({"Location": loc, "EmployeeID": f"E{i:02d}{emp_id:04d}"})
            
    return pd.DataFrame(data)

@st.cache_data
def to_excel(df: pd.DataFrame):
    """
    Converts a DataFrame to an Excel file in memory (as bytes).
    This is a reusable helper function.
    """
    # Create an in-memory binary stream
    output = BytesIO()
    # Use a writer to save the dataframe to the BytesIO object
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Report')
    # Get the bytes from the stream
    processed_data = output.getvalue()
    return processed_data

# --- Main Application Logic ---

# 1. Load or generate the data
# In a real app, you'd load your data here. For this example, we generate it.
df = create_dummy_data()

# 2. Perform the requested analysis
st.header("Total Count of Employees by Location")
st.markdown("This table shows the total number of employees aggregated by their work location.")

# Perform the groupby and count operation
employee_counts = df.groupby('Location')['EmployeeID'].count().reset_index()

# Rename columns for better readability and sort by count
employee_counts.rename(columns={'EmployeeID': 'Employee Count'}, inplace=True)
employee_counts = employee_counts.sort_values(by='Employee Count', ascending=False)

# 3. Display the full, scrollable table
st.markdown("### Data in Tabular Format")
st.info("This table is scrollable. Use the download button below to export the full dataset.")
st.dataframe(
    employee_counts, 
    height=600,  # Set a fixed height in pixels to enable scrolling
    use_container_width=True # Make the table expand to the full width of the container
)

# 4. Provide the download button
st.download_button(
    label="📥 Download as Excel",
    data=to_excel(employee_counts), # Convert the dataframe to Excel bytes
    file_name="employee_counts_by_location.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# You can also add an expander to show the raw, un-aggregated data
with st.expander("View Raw Employee Data"):
    st.dataframe(df)
