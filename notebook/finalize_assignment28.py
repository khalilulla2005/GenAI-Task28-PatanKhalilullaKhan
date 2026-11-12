import nbformat as nbf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 28: Q&A RAG Chatbot with Message History
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
This assignment demonstrates building a conversational RAG assistant that maintains context-aware state.
""")

# Code: Setup RAG
code_cell_1 = nbf.v4.new_code_cell("""# Setup: Load, Split, Embed
from langchain_core.documents import Document

# Create sample large docs
texts = ["GenAI is a field of AI that creates new content.", "RAG connects LLMs to external data.", "LangChain provides tools for RAG."]
docs = [Document(page_content=t) for t in texts]

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("RAG components ready.")
""")

# Markdown: RAG Chain
text_cell_2 = nbf.v4.new_markdown_cell("""## PART 4 — Q&A RAG Chain with History
Building a chain that accepts user question, retrieves context, and references conversation history.
""")

# Code: RAG Chain
code_cell_2 = nbf.v4.new_code_cell("""# Prompt with History
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from retrieved context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# History store
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# RAG Chain
chain = prompt | llm | StrOutputParser()

# Wrapped with History
rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

print("=== Task 9/10: Stateful RAG Test ===")
config = {"configurable": {"session_id": "user1"}}
print("Q1:", rag_with_history.invoke({"context": "GenAI is AI content creation.", "question": "What is GenAI?"}, config=config))
print("Q2 (Follow-up):", rag_with_history.invoke({"context": "GenAI creates content.", "question": "Can you explain more?"}, config=config))
""")

# Markdown: Observations
text_cell_3 = nbf.v4.new_markdown_cell("""## Task 11: Observations & Insights
1. **Normal RAG vs Conversational RAG**: Normal RAG is stateless (treats queries independently). Conversational RAG uses past context to understand follow-up references.
2. **Role of History**: Enables contextual grounding for ambiguous follow-up questions (e.g., "Tell me more").
3. **Trade-offs**: Long memory improves performance but consumes more tokens and increases latency.
4. **Trimming**: Essential to ensure the history fits within the LLM's context window without losing essential context.
""")

nb['cells'] = [text_cell_1, code_cell_1, text_cell_2, code_cell_2, text_cell_3]
nbf.write(nb, 'D:\\techtude\\GenAI-Task28-PatanKhalilullaKhan\\notebook\\GenAI_Assignment28_Conversational_RAG.ipynb')
print("Assignment 28 Notebook finalized!")
