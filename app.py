import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
import re
from google import genai

# Configure page
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("📊 AI Enterprise Data Analyst Agent")

# Sidebar - API Key and File Upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to get started.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# Helper function to extract clean Python code from LLM response
def clean_code(llm_output: str) -> str:
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", llm_output, re.DOTALL)
    if code_match:
        return code_match.group(1)
    return llm_output.strip()

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Dataset metadata to pass to the agent
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    schema_prompt = f"""
    Dataset Columns and Types:
    {df_info}
    
    First 3 rows:
    {df.head(3).to_dict()}
    """

    st.divider()
    user_query = st.text_input("Ask a question about your data or request a chart:")

    if user_query:
        with st.spinner("Agent is analyzing data and generating code..."):
            prompt = f"""
            You are an expert Python Data Analyst.
            Given the following dataset schema:
            {schema_prompt}

            User Goal: {user_query}

            Instructions:
            - Write Python code using pandas, matplotlib, or seaborn.
            - The dataframe is already loaded as `df`.
            - Do NOT re-read or create a new dataframe.
            - If generating a chart, use `plt.figure()` and make it look clean. Do not call `plt.show()`; the script will render it.
            - Print the key numerical results or tables using `print()`.
            - Return ONLY valid executable Python code inside a single ```python ``` codeblock.
            """

           # Call Gemini
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            raw_code = clean_code(response.text)

        # Show the generated code
        with st.expander("Inspect Generated Agent Code"):
            st.code(raw_code, language="python")

        # Execute code in a safe local namespace
        st.subheader("Results")
        local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        
        # Capture standard output (print statements)
        stdout_backup = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        plt.clf()  # Clear previous plot
        execution_success = False

        try:
            exec(raw_code, {}, local_vars)
            execution_success = True
        except Exception as e:
            st.error(f"Execution Error: {e}")
        finally:
            sys.stdout = stdout_backup

        output_text = captured_output.getvalue()
        if output_text:
            st.text(output_text)

        # If a plot was generated, display it
        fig = plt.gcf()
        if fig.axes:
            st.pyplot(fig)

        # Generate Executive Summary
        if execution_success:
            with st.spinner("Generating executive summary..."):
                summary_prompt = f"""
                User Question: {user_query}
                Execution Output: {output_text}
                
                Provide a crisp, 2-to-3 bullet point business insight explaining what this result means.
                """
                summary_res = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=summary_prompt,
                )
                st.success("**Executive Summary:**\n" + summary_res.text)