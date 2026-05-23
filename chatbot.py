import streamlit as st
from rag import load_index, retrieve, generate_answer, get_index_stats
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

st.title("💬 RAG Chatbot")
st.write("Ask questions about your indexed documents")

# Check if index exists
if not os.path.exists("faiss_index/index.faiss"):
    st.error("⚠️ No document index found!")
    st.info("Please run the **Document Indexer** first to build your knowledge base.")
    st.stop()

# Load index
try:
    index, chunks = load_index()
    stats = get_index_stats()
    
    # Display index info
    with st.sidebar:
        st.header("📊 Knowledge Base Info")
        st.metric("Total Chunks", stats['total_chunks'])
        st.metric("Documents", stats['total_documents'])
        st.caption(f"Last updated: {stats['last_updated']}")
        
        st.divider()
        
        # Retrieval settings
        st.header("⚙️ Settings")
        top_k = st.slider("Number of chunks to retrieve", 1, 10, 3)
        show_context = st.checkbox("Show retrieved context", value=True)
        
except Exception as e:
    st.error(f"Error loading index: {str(e)}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show context if available
        if message["role"] == "assistant" and "context" in message and show_context:
            with st.expander("📄 View Source Context"):
                for i, ctx in enumerate(message["context"], 1):
                    st.caption(f"**Chunk {i}:**")
                    st.text(ctx.strip())

# Chat input
if query := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            retrieved = retrieve(query, index, chunks, top_k=top_k)
            
            # Generate answer
            answer = generate_answer(query, retrieved)
            
            st.markdown(answer)
            
            # Show context if enabled
            if show_context:
                with st.expander("📄 View Source Context"):
                    for i, ctx in enumerate(retrieved, 1):
                        st.caption(f"**Chunk {i}:**")
                        st.text(ctx.strip())
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "context": retrieved
            })

# Clear chat button
if st.session_state.messages:
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
