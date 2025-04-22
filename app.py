import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    return llm

def main():
    # Custom CSS for black-and-white legal aesthetic with fixed text visibility
    st.markdown("""
        <style>
        /* General page styling */
        .stApp {
            background-color: #E5E7EB; /* Light gray background */
            color: #1F2937; /* Dark gray text */
            font-family: 'Georgia', serif;
        }

        /* Title styling */
        h1 {
            color: #000000; /* Black for title */
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 20px;
            border-bottom: 2px solid #000000;
            padding-bottom: 10px;
        }

        /* Chat message styling */
        .stChatMessage {
            background-color: #000000; /* White background for messages */
            border: 1px solid #D1D5DB; /* Light gray border */
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease-in-out;
        }

        .stChatMessage:hover {
            transform: scale(1.01); /* Subtle hover effect */
        }

        /* User message styling - UPDATED COLOR */
        .stChatMessage[data-testid="stChatMessage-user"] {
            border-left: 4px solid #000000; /* Black accent for user messages */
            color: #003366; /* Dark blue color for user text */
        }

        /* Assistant message styling - UPDATED COLOR */
        .stChatMessage[data-testid="stChatMessage-assistant"] {
            border-left: 4px solid #4B5563; /* Dark gray accent for assistant messages */
            color: #660033; /* Dark maroon color for assistant text */
        }

        /* Chat input styling */
        .stTextInput > div > div > input {
            background-color: #F3F4F6; /* Light gray input background */
            color: #1F2937; /* Dark gray text for visibility */
            border: 1px solid #000000; /* Black border */
            border-radius: 8px;
            padding: 10px;
            font-family: 'Georgia', serif;
        }

        /* Placeholder text styling */
        .stTextInput > div > div > input::placeholder {
            color: #4B5563; /* Darker gray for placeholder text */
            opacity: 1; /* Ensure placeholder is fully visible */
        }

        /* Error message styling */
        .stError {
            background-color: #F9FAFB; /* Very light gray for error */
            color: #B91C1C; /* Red for error text */
            border: 1px solid #B91C1C;
            border-radius: 8px;
            padding: 10px;
            font-family: 'Georgia', serif;
        }

        /* Button styling (if any) */
        .stButton > button {
            background-color: #1F2937; /* Dark gray button */
            color: #FFFFFF; /* White text */
            border: 1px solid #000000;
            border-radius: 8px;
            padding: 8px 16px;
            font-family: 'Georgia', serif;
            transition: background-color 0.2s ease-in-out;
        }

        .stButton > button:hover {
            background-color: #4B5563; /* Slightly lighter gray on hover */
        }

        /* Remove Streamlit's default branding */
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    st.title("Legal Assistant Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Enter your query here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context.

                Context: {context}
                Question: {question}

                Keep you tone warm and welcoming.
                """
        
        HUGGINGFACE_REPO_ID="meta-llama/Llama-3.1-8B-Instruct"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()