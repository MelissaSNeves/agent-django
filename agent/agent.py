import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.vectorstores.azuresearch import AzureSearchVectorStoreRetriever
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.tools.retriever import create_retriever_tool
import chromadb
from langchain_chroma import Chroma
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer


def get_vector_database(embedding):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("agent_tools")
    client = chromadb.HttpClient(host="localhost", port=8000)
    db = Chroma(
        client=client,
        collection_name=collection.name,
        embedding_function=embedding
    )
    return db
    
def get_embedding_model(text_splited):
    # embedding = OpenAIEmbeddings()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text_splited)
    return embeddings

    
def get_retriever_tool(directory):
    loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
   
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300, 
        chunk_overlap=0,
        length_function=len )
    
    texts = text_splitter.split_documents(docs)
    
    embeddings = get_embedding_model(texts)  
    db = get_vector_database(embeddings)
    
    retriever = db.as_retriever()
    tool_retriever = create_retriever_tool(
        retriever,
        "retrieve_tool",
        "guarde o arquivo e responda perguntas com base nele"
    )
    return tool_retriever

    
def get_files_tools(directory):
    tools = FileManagementToolkit(
    root_dir=str(directory),
    selected_tools=["file_delete", "file_search", "move_file","read_file", "write_file", "list_directory"]).get_tools()
    return tools

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
working_directory = TemporaryDirectory()
tools = get_files_tools(working_directory.name)
tools.append(get_retriever_tool(working_directory.name))


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f'Você é um assistente. Certifique-se de usar as ferramentas retrieve_tool para obter informações. E execute algumas ações de acordo com as ferramentas tool_delete, tool_search, move_tool,read_tool, write_tool, list_tool. Todas as manipulações de arquivos acontecerão no diretório {working_directory}',
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) 


def handle_userinput(user_input):
    response = agent_executor.invoke({"input": user_input})
        
    return response



