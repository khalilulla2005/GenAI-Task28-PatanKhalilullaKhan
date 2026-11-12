# Assignment 28: Q&A RAG Chatbot with Message History

**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

## Overview
This assignment demonstrates building a conversational RAG assistant using **LangChain**. It focuses on maintaining stateful conversation history while grounding answers in document context to avoid hallucinations.

## Key Features
- **Conversational RAG**: Combines vector retrieval with RunnableWithMessageHistory to manage conversation context.
- **State Management**: Uses ChatMessageHistory and MessagesPlaceholder to pass past turns to the LLM.
- **Stateful Design**: Handles follow-up queries naturally based on context.

## Setup Instructions
1. Install dependencies: `pip install langchain-openai faiss-cpu langchain-community`
2. Set your OpenAI API key: ` = "your_key"`
3. Run the Jupyter Notebook in the `notebook/` directory.
