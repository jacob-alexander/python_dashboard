import pandas as pd
import streamlit as st
from io import BytesIO

# --- Function to convert DataFrame to Excel in memory ---
# This is a reusable helper function you can put at the top of your script
@st.cache_data
def to_excel(df):
    output = BytesIO()
    # Use a writer to save the dataframe to the BytesIO object
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='EmployeeData')
    # It's important to get the value of the BytesIO object
    processed_data = output.getvalue()
    return processed_data

# --- In your main app logic ---

# 1. Create your final table DataFrame
employee_counts = df.groupby('Location')['EmployeeID'].count().reset_index()
employee_counts.rename(columns={'EmployeeID': 'Total Count'}, inplace=True)

# 2. Display the table in Streamlit
st.dataframe(employee_counts)

# 3. Create the download button
st.download_button(
    label="📥 Download as Excel",
    data=to_excel(employee_counts), # Use the helper function to process the data
    file_name="employee_counts_by_location.xlsx",
    mime="application/vnd.ms-excel"
)
