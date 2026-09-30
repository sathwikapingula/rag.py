import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="🤖"
)

st.title("🤖 Mini RAG Q&A")
st.write("Paste a document, store it in ChromaDB, and ask questions about it.")


# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# Initialize ChromaDB
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)


# Document input
document = st.text_area(
    "Paste your document here",
    height=250,
    placeholder="Paste your notes, article, syllabus, etc."
)


# Add document
if st.button("Add Document"):

    if not document.strip():
        st.warning("Please enter some text.")

    else:
        chunks = [
            document[i:i + 500]
            for i in range(0, len(document), 500)
        ]

        embeddings = embedding_model.encode(chunks)

        collection.add(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        st.success(
            f"Added {len(chunks)} chunk(s) to ChromaDB."
        )


# Question input
question = st.text_input(
    "❓ Ask a question about your document"
)


# Ask AI
if st.button("Ask AI"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please add a document first.")

    else:

        # Create question embedding
        question_embedding = embedding_model.encode(
            [question]
        )[0]

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(
                3,
                collection.count()
            )
        )

        # Get retrieved chunks
        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(
            retrieved_chunks
        )


        # Create RAG prompt
        prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided document."
"""


        # Ask Ollama
        try:

            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Display answer
            st.subheader("🤖 Answer")

            st.write(
                response["message"]["content"]
            )

        except Exception as e:

            st.error(
                f"Error while contacting Ollama: {e}"
            )