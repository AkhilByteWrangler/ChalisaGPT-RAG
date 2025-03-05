import os
import re
import streamlit as st
import numpy as np
import torch
import faiss
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering
from rank_bm25 import BM25Okapi

# Configure Streamlit app settings
st.set_page_config(page_title="Hanuman Chalisa Chatbot", page_icon="🙏", layout="wide")

# Sidebar section for entering OpenAI API key
with st.sidebar:
    st.title("API Key")
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

# Stop execution if API key is missing
if not openai_api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# App title and description
st.title("Hanuman Chalisa Q&A Chatbot")
st.write("Ask any question about **Hanuman Chalisa**, and this chatbot will provide an answer based on its text.")

#************************************************************************************************************************************#

# Load Hanuman Chalisa verses in Hindi
def load_hanuman_chalisa(filename="hanuman_chalisa_hindi.txt"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing file: {filename}. Please ensure the text file exists.")
    
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]  # Remove empty lines

# Load the original Hanuman Chalisa verses
try:
    hanuman_chalisa_chunks_hindi = load_hanuman_chalisa()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Function to translate Hanuman Chalisa verses using OpenAI GPT
def translate_chalisa():
    translations = []
    for verse in hanuman_chalisa_chunks_hindi:
        response = client.chat.completions.create(
            model="gpt-4.5-preview-2025-02-27",
            messages=[
                {"role": "system", "content": (
                    "Translate the given passage from Hanuman Chalisa into meaningful English. "
                    "Maintain its spiritual depth, cultural significance, and devotional essence. "
                    "Avoid literal translation but retain key names and phrases."
                )},
                {"role": "user", "content": verse}
            ],
            max_tokens=200
        )
        translations.append(response.choices[0].message.content.strip())
    return translations

# Load or generate translations
translation_file = "hanuman_chalisa_translated.txt"

if os.path.exists(translation_file):
    with open(translation_file, "r", encoding="utf-8") as file:
        hanuman_chalisa_chunks = [line.strip() for line in file]
else:
    hanuman_chalisa_chunks = translate_chalisa()
    with open(translation_file, "w", encoding="utf-8") as file:
        file.write("\n".join(hanuman_chalisa_chunks))

#************************************************************************************************************************************#

# Load the Sentence Transformer model and tokenizer for embedding retrieval
retrieval_model = AutoModel.from_pretrained("sentence-transformers/msmarco-MiniLM-L6-cos-v5")
retrieval_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-MiniLM-L6-cos-v5")

def get_embedding(text):
    """
    Generates a dense embedding for the given text using the Sentence-BERT model.
    """
    tokens = retrieval_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = retrieval_model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# Compute embeddings for all Hanuman Chalisa chunks
chunk_embeddings = np.array([get_embedding(chunk) for chunk in hanuman_chalisa_chunks], dtype="float32")

#************************************************************************************************************************************#

# Build FAISS Index
def build_faiss_index():
    """Creates a FAISS index for fast similarity search."""
    global faiss_index
    faiss_index = faiss.IndexFlatL2(384)  
    faiss_index.add(chunk_embeddings)
    return faiss_index

# Build FAISS index
faiss_index = build_faiss_index()

# Build BM25 Index
bm25_corpus = [chunk.split() for chunk in hanuman_chalisa_chunks]
bm25 = BM25Okapi(bm25_corpus)

def hybrid_retrieval(query, faiss_weight=0.7, bm25_weight=0.3):
    """
    Combines FAISS (semantic search) and BM25 (keyword search).
    """
    # FAISS Retrieval
    query_embedding = get_embedding(query).astype("float32").reshape(1, -1)
    _, best_faiss_idx = faiss_index.search(query_embedding, 1)
    
    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query.split())  
    best_bm25_idx = np.argmax(bm25_scores) 

    # Normalize Scores
    faiss_score = 1 - best_faiss_idx[0][0] / len(hanuman_chalisa_chunks)  
    bm25_score = bm25_scores[best_bm25_idx] / max(bm25_scores)  

    # Combine Scores
    combined_scores = (faiss_weight * faiss_score) + (bm25_weight * bm25_score)

    # Choose the best passage
    best_idx = best_faiss_idx[0][0] if combined_scores > 0.5 else best_bm25_idx
    return hanuman_chalisa_chunks[best_idx]

#************************************************************************************************************************************#

# Load RoBERTa-based Question Answering model and tokenizer
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

def extract_answer(context, question, confidence_threshold=2.0):
    """
    Extracts an answer from the given context using a RoBERTa-based QA model.
    """
    # Tokenize input with sliding window to handle long passages
    inputs = qa_tokenizer(
        question, context, return_tensors="pt", truncation=True, max_length=512, stride=128
    )

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_idx = torch.argmax(start_scores).item()
    end_idx = torch.argmax(end_scores).item() + 1  

    # Extract the confidence score
    confidence = torch.max(start_scores).item()

    if start_idx >= end_idx:
        return "I'm not confident in this answer. Could you rephrase your question?"

    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    ).strip()

    # Handle low-confidence answers
    if confidence < confidence_threshold or not answer:
        return "I'm not sure about the answer. Could you ask in a different way?"

    return answer

#************************************************************************************************************************************#

# Streamlit user input
user_input = st.text_input("Ask a question about Hanuman Chalisa:")

if user_input:
    # Retrieve the most relevant passage based on the user's query
    passage = hybrid_retrieval(user_input)

    # Generate an answer based on the retrieved passage
    answer = extract_answer(passage, user_input)

    # RAG-aware prompt for GPT-4 to evaluate answer correctness
    judge_prompt = (
        f"You are an expert in Hindu scriptures, particularly Hanuman Chalisa. Your task is to fairly and accurately assess the correctness of the given answer "
        f"in relation to the provided passage from Hanuman Chalisa. Ensure your evaluation considers both factual correctness and relevance to the passage.\n\n"
        f"**Question:** {user_input}\n"
        f"**Passage:** {passage}\n"
        f"**Generated Answer:** {answer}\n\n"
        f"### Assessment Criteria:\n"
        f"- Does the answer correctly reflect the meaning of the passage?\n"
        f"- Does it align with the teachings and essence of Hanuman Chalisa?\n"
        f"- Is it factually accurate and not misleading?\n\n"
        f"### Response Format:\n"
        f"- **Confidence Score (0-100%)**: Provide a numerical rating based on the correctness and relevance.\n"
        f"- **Justification (max 2 sentences, <300 chars)**: Briefly explain why you assigned this score, focusing on accuracy, alignment with Hanuman Chalisa, and completeness."
    )

    # Obtain feedback from GPT-4 on the quality of the answer
    judge_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fair and expert evaluator of answers related to Hanuman Chalisa. Your feedback should be unbiased, concise, and aligned with the scripture's meaning."},
            {"role": "user", "content": judge_prompt}
        ],
        max_tokens=100
    )

    judge_feedback = judge_response.choices[0].message.content.strip()

    # Extract numerical confidence score from GPT-4 response
    match = re.search(r"\b(100|[1-9][0-9]?)%\b", judge_feedback)  
    confidence_score = int(match.group(1)) if match else 50  

    # Display results
    st.subheader("Relevant Passage & Generated Answer")

    st.markdown(f"### 🏆 Confidence Score: **{confidence_score}%**")
    st.markdown(f"**📖 Relevant Passage:**\n> {passage}")
    st.success(f"**💡 Generated Answer:** {answer}")
    st.info(f"**🧐 Judge's Feedback:** {judge_feedback}")
    st.markdown("---")


