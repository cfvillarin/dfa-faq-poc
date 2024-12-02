__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from prompts import qe_prompt, rag_prompt, email_format_prompt, in_scope_prompt, satisfactory_prompt

import ast

def process_page_content(page_content):
    try:
        # Extract the structured dictionary part from the string
        start_idx = page_content.find("{")
        structured_part = page_content[start_idx:]
        # Safely evaluate the string as a Python dictionary
        doc_dict = ast.literal_eval(structured_part)
        return doc_dict
    except (ValueError, SyntaxError) as e:
        st.error(f"Error parsing document: {e}")
        return None

# Create columns for the title and logo
col1, col2 = st.columns([3.5, 1])  # Adjust the ratio as needed

# Title in the first column
with col1:
    st.title("📄 Sanggun-E PoC 🤖")
    st.write(
        "This app answers questions based on FAQs found [here](https://consular.dfa.gov.ph/faqs-menu?). "
    )
# Logo and "Developed by CAIR" text in the second column
with col2:
    st.image("images/CAIR_cropped.png", use_column_width=True)
    st.markdown(
        """
        <div style="text-align: center; margin-top: -10px;">
            Developed by CAIR
        </div>
        """, 
        unsafe_allow_html=True)

email_body = st.text_area(
    "Enter your email text here!",
    placeholder="""Hi DFA, 

My name is James. My passport has been damaged. What should I do?

Thank you very much!
    """,
    height=200
)

output_dict = {'flagged': False, 'email_body': email_body}

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableAssign, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# RETRIEVER 
CHROMA_PATH = "chroma"
n_retrieved_docs = 5

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever =  db.as_retriever(search_type="similarity_score_threshold", search_kwargs={'k': n_retrieved_docs, 'score_threshold': 0.8})

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  
# repo_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.01)

# output_dict = RunnablePassthrough.assign(translated=prompt | llm).invoke(temp_dict)

if email_body:
    output_dict = RunnablePassthrough.assign(extracted_query=qe_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())).invoke(output_dict)
    output_dict = RunnablePassthrough.assign(in_scope=in_scope_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip().split()[0])).invoke(output_dict)
    if 'yes' in output_dict['in_scope'].lower():
        output_dict['in_scope'] = True
        # output_dict = RunnablePassthrough.assign(keywords=keyword_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: eval(x.strip("```python")))).invoke(output_dict)
        output_dict = RunnablePassthrough.assign(retrieved_docs=RunnableLambda(lambda x: x['extracted_query']) | retriever).invoke(output_dict)
        output_dict = RunnablePassthrough.assign(generated_answer=rag_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())).invoke(output_dict)
        output_dict = RunnablePassthrough.assign(satisfactory_answer=satisfactory_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip().split()[0])).invoke(output_dict)
        if 'yes' in output_dict['satisfactory_answer'].lower():
            output_dict['satisfactory_answer'] = True
            output_dict = RunnablePassthrough.assign(email_autoreply=email_format_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())).invoke(output_dict)
        else:
            output_dict['satisfactory_answer'] = False
            output_dict['flagged'] = True
    else:
        output_dict['in_scope'] = False
        output_dict['flagged'] = True
    
    if output_dict['flagged']:
        output_dict['email_autoreply'] = """Thank you for contacting us. Your email has been received and flagged for a manual response by one of our agents, as it requires assistance beyond our FAQs. Please stand by, and we will get back to you as soon as possible."""

    # First box: AI-Generated Autoreply
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(255, 255, 255, 0.2); 
            padding: 10px; 
            border-radius: 5px; 
            background-color: rgba(255, 255, 255, 0.1); 
            color: inherit; 
            margin-bottom: 20px;">
            <strong>AI-Generated Autoreply:</strong><br>
            {output_dict['email_autoreply']}
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        ---
        ### RAG Breakdown
        #### Extracted Query
        <div style="
            border: 1px solid rgba(255, 255, 255, 0.2); 
            padding: 10px; 
            border-radius: 5px; 
            background-color: rgba(255, 255, 255, 0.1); 
            color: inherit; 
            margin-bottom: 20px;">
            {output_dict['extracted_query']}
        </div>
        """, 
        unsafe_allow_html=True
    )

    if output_dict.get('retrieved_docs'):
        # Display the retrieved documents
        st.markdown("#### Retrieved FAQs")
        with st.expander("View Retrieved FAQs"):
            for i, doc in enumerate(output_dict['retrieved_docs']):
                doc_dict = process_page_content(doc.page_content + '}')
                if doc_dict:
                    st.markdown(
                        f"""
                        <div style="
                            border: 1px solid rgba(255, 255, 255, 0.2); 
                            padding: 10px; 
                            border-radius: 5px; 
                            background-color: rgba(255, 255, 255, 0.05); 
                            color: inherit; 
                            margin-bottom: 10px;">
                            <strong>FAQ {i + 1}:</strong><br>
                            <strong>Category:</strong> {doc_dict.get('category', 'N/A')}<br>
                            <strong>Question:</strong> {doc_dict.get('question', 'N/A')}<br>
                            <strong>Answer:</strong> {doc_dict.get('answer', 'N/A')}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st.write(f"Document {i + 1} could not be processed.")


        # Display the generated answer and AI-generated autoreply
        st.markdown(
            f"""
            #### Generated Answer
            <div style="
                border: 1px solid rgba(255, 255, 255, 0.2); 
                padding: 10px; 
                border-radius: 5px; 
                background-color: rgba(255, 255, 255, 0.1); 
                color: inherit; 
                margin-bottom: 20px;">
                {output_dict['generated_answer']}
            </div>
            """, 
            unsafe_allow_html=True
            )
    else:
        st.markdown("#### Retrieved FAQs")
        st.markdown(
            f"""
            <div style="
                border: 1px solid rgba(255, 255, 255, 0.2); 
                padding: 10px; 
                border-radius: 5px; 
                background-color: rgba(255, 255, 255, 0.1); 
                color: inherit; 
                margin-bottom: 20px;">
                No retrieved documents. The query is not similar to any FAQ.
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Add a collapsible section for the raw dictionary
    with st.expander("View Raw Output Dictionary"):
        st.json(output_dict)
