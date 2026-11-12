import nbformat as nbf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 28: Conversational RAG Chatbot (Grand Final Submission)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
Building a stateful RAG chatbot where retrieved context flows automatically through an LCEL pipeline, including history trimming and proper fallback instructions.
""")

# Code: Task 1 (Ingestion)
code_cell_1 = nbf.v4.new_code_cell("""# Task 1: Load Documents
loader = TextLoader('data/sample.txt')
docs = loader.load()
print(f"Number of documents: {len(docs)}")
print(f"Sample content: {docs[0].page_content[:100]}...")

# Task 2: Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
""")

# Code: Task 5, 6, 8 (Pipeline)
code_cell_2 = nbf.v4.new_code_cell("""# Task 5, 6, 8: RAG Chain with History and Trimming
def format_docs(docs):
    return "\\n\\n".join([d.page_content for d in docs])

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on context: {context}. If the answer is not in context, say 'I don't know'."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Trimming History (Task 8)
def trim_history(history, max_msgs=4):
    return history[-max_msgs:]

# LCEL Chain (Task 6)
chain = (
    RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough(), "history": lambda x: trim_history(x["history"])})
    | prompt
    | llm
    | StrOutputParser()
)

# State Management
store = {}
def get_session_history(session_id):
    if session_id not in store: store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
""")

# Code: Task 9 (Evaluation)
code_cell_3 = nbf.v4.new_code_cell("""# Task 9: 3-turn Evaluation
config = {"configurable": {"session_id": "eval_user"}}
print("Q1 (Factual):", rag_with_history.invoke({"question": "What is FAISS?"}, config=config))
print("Q2 (Follow-up):", rag_with_history.invoke({"question": "Why is it fast?"}, config=config))
print("Q3 (Clarification):", rag_with_history.invoke({"question": "What do you mean by vector similarity?"}, config=config))
""")

nb['cells'] = [text_cell_1, code_cell_1, code_cell_2, code_cell_3]
nbf.write(nb, 'D:\\techtude\\GenAI-Task28-PatanKhalilullaKhan\\notebook\\GenAI_Assignment28_Conversational_RAG.ipynb')
print("Assignment 28 Final Notebook generated!")
