from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import os

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    # elif file.endswith('.docx') or file.endswith('.doc'):
    #     doc_path = "./docs/" + file
    #     loader = Docx2txtLoader(doc_path)
    #     documents.extend(loader.load())
    # elif file.endswith('.txt'):
    #     text_path = "./docs/" + file
    #     loader = TextLoader(text_path)
    #     documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embeddings, persist_directory="./data")
vectordb.persist()

retriever=vectordb.as_retriever(search_kwargs={'k': 6})

repo_id = "mistralai/Mistral-7B-v0.1"
llm = HuggingFaceHub(huggingfacehub_api_token='hf_xx, 
                     repo_id=repo_id, model_kwargs={"temperature":0.2, "max_new_tokens":50})

# Create the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa_chain.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))
