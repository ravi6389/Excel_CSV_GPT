import streamlit as st
import pandas as pd

from langchain.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline
# from langchain import LLMChain
from langchain.chains import LLMChain

import os
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


llm = ChatGroq(temperature=0.8, groq_api_key=st.secrets['GROQ_API_KEY'], model_name="llama3-70b-8192")


# ---------------------------
# Define the Prompt Template
# ---------------------------

prompt_template = """
I have the following CSV data with the columns: "{columns}". 
Data in in dataframe called 'df' already.
Dont give any description or explanation or any english sentence,
just write relevant Python code and store output in a variable called result.
Please generate a Python script using this 'df' as input dataframe and pandas to answer this question: "{question}".
Do not write any Python script which alters the dataframe 'df.
Write only read only Python script.        
      
"""

# Create the PromptTemplate object using LangChain
template = PromptTemplate(
    input_variables=["columns", "question"],
    template=prompt_template,
)

# Create the LLMChain to manage the model and prompt interaction
llm_chain = LLMChain(prompt=template, llm=llm)

# llm_chain = llm |template

# ---------------------------
# Streamlit App Interface
# ---------------------------

st.title("Excel or CSV GPT")

st.markdown("""
It can read any CSV or excel file, take user input, write python script to 
            answer user query and execute it to return answer to user's question.\n
Developed by Ravi Shankar Prasad. 

""")

df = pd.DataFrame()
uploaded_file = st.file_uploader("Upload an excel  or csv file",\
                                  type=["xlsx", "xls", "csv"])
if(uploaded_file):

    if uploaded_file.name.endswith('.csv'):
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.success("CSV file successfully read.")
            st.write('Preview of the uploaded file')
            st.write(df.head(5))
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
           

    if uploaded_file.name.endswith(('.xls', '.xlsx')):
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            st.success("Excel file successfully read.")
            st.write('Preview of the uploaded file')
            st.write(df.head(5))
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            
    
    # Display the first few rows of the dataset
  
      


       



    # Ask the user for a question about the data
    question = st.text_input("❓ Ask a question about the data")

    if question:
        with st.spinner("Generating Python code..."):
            try:
                # Run the LLMChain to generate the Python script based on the question and CSV columns
                python_script = llm_chain.invoke({
                    "columns": ", ".join(df.columns),
                    "question": question
                })

                # Display the generated Python code
                st.write("### 📝 Generated Python Code:")
                st.code(python_script, language='python')
                
                # Option to execute the generated Python code
                if st.button("▶️ Run Code"):
                    try:
                        # Provide the DataFrame in the exec environment
                        exec_globals = {"df": df, "pd": pd}
                        exec_locals = {}
                        # st.write(python_script)
                        st.write(python_script['text'].strip('`'))
                        python_script['text'] = python_script['text'].strip('`')
                        exec(python_script['text'], exec_globals, exec_locals)
                        # st.write(exec_globals)
                        # If a result variable is present, display it
                        if 'result' in exec_globals:
                            st.write("### 📊 Result:")
                            st.write(exec_globals['result'])

                        elif 'result' in exec_locals:
                            st.write("### 📊 Result:")
                            st.write(exec_locals['result'])
                            
                        else:
                            st.warning("⚠️ The code did not produce a 'result' variable.")
                    except Exception as e:
                        st.error(f"🚫 Error running the code: {e}")
            except Exception as e:
                st.error(f"🚫 Error generating the code: {e}")
