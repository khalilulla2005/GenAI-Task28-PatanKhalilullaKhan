import nbformat as nbf
from operator import itemgetter
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

# Title
nb['cells'].append(nbf.v4.new_markdown_cell("""# Assignment 28: Conversational RAG Chatbot (Grand Final Rework)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  
---
"""))

# Task 1 & 2
nb['cells'].append(nbf.v4.new_markdown_cell("## Tasks 1-4: Ingestion, Indexing, and Embedding"))
nb['cells'].append(nbf.v4.new_code_cell("""# Task 1: Load Documents
loader = TextLoader('data/large_sample.txt')
docs = loader.load()
print(f"Loaded {len(docs)} document(s).")

# Task 2: Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print("Vector store created with {len(chunks)} chunks.")
"""))

# Task 6, 8 (Chain & Trimming)
nb['cells'].append(nbf.v4.new_markdown_cell("## Tasks 5-8: RAG Pipeline with History Management"))
nb['cells'].append(nbf.v4.new_code_cell("""# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the context to answer. If unknown, say 'I don't know'. Context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Task 8: Trimming
def trim_history(history, max_msgs=4):
    return history[-max_msgs:]

# Task 6: RAG Chain connecting retriever to prompt
# Using itemgetter('question') to ensure input is passed as a string
chain = (
    RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": lambda x: trim_history(x["history"])
    })
    | prompt
    | llm
    | StrOutputParser()
)

# History Management
store = {}
def get_session_history(session_id):
    if session_id not in store: store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
"""))

# Task 9 (Evaluation)
nb['cells'].append(nbf.v4.new_markdown_cell("## Task 9: Evaluation & Insights"))
nb['cells'].append(nbf.v4.new_code_cell("""# Task 9: 3-turn Evaluation
config = {"configurable": {"session_id": "final_eval"}}
print("Q1:", rag_chain.invoke({"question": "What is RAG?"}, config=config))
print("Q2:", rag_chain.invoke({"question": "How does FAISS help?"}, config=config))
print("Q3:", rag_chain.invoke({"question": "Can you clarify the previous point about vector search?"}, config=config))
"""))

# Task 11 (Observations)
nb['cells'].append(nbf.v4.new_markdown_cell("""## Task 11: Observations & Insights
1. **Normal vs Conversational**: Normal RAG lacks state; Conversational RAG maintains state for follow-ups.
2. **History importance**: Required for understanding reference-based questions ("what about that last point?").
3. **Trade-offs**: Memory improves conversational flow but increases token cost and latency over time.
4. **Trimming**: Essential to keep context windows clean and within cost budgets.
"""))

nbf.write(nb, 'D:\\techtude\\GenAI-Task28-PatanKhalilullaKhan\\notebook\\GenAI_Assignment28_Conversational_RAG.ipynb')
print("Assignment 28 Final Notebook generated!")
