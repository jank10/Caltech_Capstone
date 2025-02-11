import pandas as pd
import faiss
import numpy as np
import openai
import os
from sentence_transformers import SentenceTransformer

# Load CSV files with correct file paths
confluence_path = "rag_data/Confluence_Data__Technical_Documentation_.csv"
erp_path = "rag_data/ERP_Data__Materials__Prices__and_Stock_.csv"
crm_path = "rag_data/CRM_Data__Customer_Information_.csv"

confluence_df = pd.read_csv(confluence_path)
erp_df = pd.read_csv(erp_path)
crm_df = pd.read_csv(crm_path)

# Combine data into text format for embedding
confluence_texts = confluence_df.apply(lambda row: f"Material: {row.get('Material', 'N/A')}, Description: {row.get('Description', 'N/A')}, Processing: {row.get('Processing', 'N/A')}, Compliance: {row.get('Compliance', 'N/A')}", axis=1).tolist()
erp_texts = erp_df.apply(lambda row: f"Material: {row.get('Material', 'N/A')}, Price: {row.get('Price_per_kg', 'N/A')} per kg, Stock: {row.get('Stock_kg', 'N/A')} kg, Supplier: {row.get('Supplier', 'N/A')}", axis=1).tolist()
crm_texts = crm_df.apply(lambda row: f"Customer: {row.get('Name', 'N/A')}, Company: {row.get('Company', 'N/A')}, Inquiry: {row.get('Inquiry_Topic', 'N/A')}" if 'Inquiry_Topic' in row else "", axis=1).tolist()

documents = confluence_texts + erp_texts + crm_texts

# Load sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index
print("Creating FAISS index...")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

print("FAISS index created successfully.")

def search_rag(query, top_k=3, conversation_history=[]):
    """Search the vector database for relevant results and generate an AI-powered response."""
    print(f"Query: {query}")
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    
    context = "\n".join(retrieved_docs)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    client = openai.OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are an AI assistant that answers questions based on retrieved information."}
    ]
    
    # Include conversation history
    messages.extend(conversation_history)
    
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    ai_response = response.choices[0].message.content
    
    # Update conversation history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": ai_response})
    
    return ai_response, conversation_history

# AI Agent Loop
conversation_history = []
print("Welcome to the AI agent. Type 'exit' to quit.")
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Exiting AI agent.")
        break
    response, conversation_history = search_rag(user_query, conversation_history=conversation_history)
    print(f"AI: {response}")