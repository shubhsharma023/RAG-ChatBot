import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Path for FAISS vector store
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom prompt template for Chainlit QA model
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Setup prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Setup QA retrieval chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Load local LLM
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Setup the QA bot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Output function for handling query responses
def final_result(query):
    qa = qa_bot()
    response = qa({'query': query})
    answer = response["result"]
    
    # Retrieve and format sources if available
    sources = response.get("source_documents", [])
    if sources:
        source_texts = "\n".join([f"- {source.metadata.get('source')}" for source in sources])
        answer += f"\n\nSources:\n{source_texts}"
    else:
        answer += "\n\nNo sources found."
        
    return answer

# Load environment variables from .env file
load_dotenv()

# Retrieve the Telegram token from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi, Welcome to Advanced Text Mining and Social Media Analysis ChatBot 🤖. What is your query?")

# Handle incoming messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    response = final_result(query)
    await update.message.reply_text(response)

# Main function to start the bot
def main():
    # Initialize the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
