import os
import pdfplumber
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



# price is a factor for our company, so we're going to use a low cost model
MODEL = "gpt-4o-mini"
db_name = "vector_db"

# Load environment variables in a file called .env

load_dotenv(override=True)


def process_pdf(pdf_file):
    try:
        loader = PyPDFLoader(pdf_file.name)
        pages = loader.load()
        if not pages:
            raise ValueError("No text found in padf.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(pages)
        if not chunks:
            raise ValueError("Unable to split the PDF into chunks.")
        if not chunks:
            raise ValueError(f"ERROR: File is ecrypted/protected No text chunks generated fro {pdf_file}.")
        embeddings = OpenAIEmbeddings()
        #print(chunks)
        if os.path.exists(db_name):
            Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
        # Embed the chunks with OpenAI Embeddings
       
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
        
        # Sample embedding dimension
        collection = vectorstore._collection
        sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        dimensions = len(sample_embedding)
        print(f"The vectors have {dimensions:,} dimensions")
        
        # Create the OpenAI Chat Model
        llm = ChatOpenAI(temperature=0.7, model=MODEL)  # Or another model
      
        # Set up conversation memory
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        # Set up the retriever (vector store)
        retriever = vectorstore.as_retriever()
            
        # Set up the Conversational Retrieval Chain
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        
        # Return the conversation chain
    
        return conversation_chain
    except Exception as e:
        raise RuntimeError(f"PDF processing failed: {str(e)}")

# Function to upload PDF
def upload_pdf(file):
    global chain
    if file is None:
        chain = None
        return "pleae upload the file!"
    chain = process_pdf(file)
    return "processed the file ask questions"

# ask_question function
def ask_question(message, history):
    if chain is None:
        return "upload the pdf first"
    else:
        try:
            result = chain.invoke({"question":message})
            answer = result.get("answer", "No answer found.")
        except Exception as e:
            answer = f"Error:{str(e)}"
    history.append((message, answer))
    return history, history, ""
# Building Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Chat with your pdf!!")
    # File uploader
    file_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
    # Status text
    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat history!!!")
    msg=gr.Textbox(label="Ask anything related to pdf...")
    clear = gr.Button("Clear chat")

    state = gr.State([]) 

    file_input.change(upload_pdf, inputs=[file_input], outputs=[status])
    msg.submit(ask_question, [msg, state], [chatbot, state, msg])
    clear.click(lambda: ([],[]), None, [chatbot, state])
    chain = None  # global QA chain

# Launch the app
demo.launch(inline=False)
