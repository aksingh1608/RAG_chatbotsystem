import streamlit as st
from ingest import smart_extract, chunk_text
from rag import create_index, list_indexed_documents, get_index_stats
import os
import json
from datetime import datetime

st.set_page_config(page_title="Document Indexer", page_icon="📚")

st.title("📚 Document Indexer")
st.write("Build your knowledge base by uploading and indexing PDF documents")

# Display current index statistics
if os.path.exists("faiss_index/index.faiss"):
    stats = get_index_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", stats['total_chunks'])
    with col2:
        st.metric("Documents Indexed", stats['total_documents'])
    with col3:
        st.metric("Last Updated", stats['last_updated'])
    
    st.divider()
    
    # Show indexed documents
    with st.expander("📋 View Indexed Documents"):
        docs = list_indexed_documents()
        if docs:
            for doc in docs:
                st.write(f"- {doc['filename']} ({doc['chunks']} chunks) - {doc['timestamp']}")
        else:
            st.info("No documents indexed yet")
else:
    st.info("No index found. Upload your first document to get started!")

st.divider()

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF documents", 
    type="pdf", 
    accept_multiple_files=True,
    help="You can upload multiple PDF files at once"
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} file(s) selected**")
    
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
    
    if st.button("🔨 Process and Index Documents", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_files = len(uploaded_files)
        all_chunks = []
        doc_metadata = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save temporary file
            temp_path = f"temp_{idx}.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Extract and chunk
            try:
                text = smart_extract(temp_path)
                chunks = chunk_text(text)
                
                # Store metadata about this document
                doc_metadata.append({
                    'filename': uploaded_file.name,
                    'chunks': len(chunks),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                all_chunks.extend(chunks)
                
                st.success(f"✓ {uploaded_file.name}: {len(chunks)} chunks extracted")
                
            except Exception as e:
                st.error(f"✗ Error processing {uploaded_file.name}: {str(e)}")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            progress_bar.progress((idx + 1) / total_files)
        
        # Create or update index
        if all_chunks:
            status_text.text("Building FAISS index...")
            create_index(all_chunks, doc_metadata, append=os.path.exists("faiss_index/index.faiss"))
            status_text.text("")
            
            st.success(f"🎉 Successfully indexed {len(all_chunks)} chunks from {total_files} document(s)!")
            st.balloons()
            st.rerun()
        else:
            st.error("No text could be extracted from the uploaded documents")

# Option to clear the index
if os.path.exists("faiss_index/index.faiss"):
    st.divider()
    with st.expander("⚠️ Danger Zone"):
        st.warning("This will delete all indexed documents and start fresh")
        if st.button("🗑️ Clear All Indexed Documents", type="secondary"):
            import shutil
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")
            st.success("Index cleared successfully!")
            st.rerun()
