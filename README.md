# **Hanuman Chalisa Chatbot (RAG + Translation)**  

## **About**  
This chatbot answers questions about the **Hanuman Chalisa**, using **Retrieval-Augmented Generation (RAG)**. Unlike standard AI models, which rely on pre-trained knowledge, our chatbot **retrieves information only from the Hanuman Chalisa text**.  

## **How It Works**  

### **1. Translation (OpenAI GPT-4)**  
- We use **OpenAI’s GPT4** to translate the **Hanuman Chalisa** from **Hindi to English** while maintaining its **spiritual essence**.  
- This ensures a more **faithful and poetic translation**, rather than a literal one.  

### **2. Question Answering (RAG-based)**  
- Unlike generic AI chatbots, we **do not use GPT for Q&A** to prevent **pre-trained biases from affecting answers**.  
- Instead, we use **FAISS (semantic search) + BM25 (keyword search)** to **retrieve the most relevant verse** from Hanuman Chalisa.  
- A **RoBERTa-based QA model** then extracts the answer **strictly from the retrieved text**.  

### **3. Fair Answer Evaluation (LLM as a Judge)**  
- We use **GPT-4 only to evaluate** the correctness of generated answers, ensuring responses stay **faithful to Hanuman Chalisa**.  
- The LLM **does not generate new knowledge**, but acts as an **impartial judge**, assessing whether the response correctly reflects the retrieved passage.  

---

## **The True Essence of RAG in a Pre-Trained World**  
Most AI models come pre-loaded with **vast but uncontrolled** knowledge. The true power of **RAG** is **controlling the knowledge source** by adding only the latest, relevant, or even personal data.  

For example:  
- **User-Added Data (Non-Fact):** *"Akhil is the greatest admirer of Lord Hanuman."*  
  - This isn’t a universal truth, but **I could add it as retrievable knowledge** in my system.  

With **RAG**, **you control the knowledge AI uses**, instead of relying on an unpredictable, pre-trained model.  

Try asking:  
- *"What is the sacred herb?"*
- *"Who is Shri Raghubir?"*
- *"Who is the greatest admirer of Lord Hanuman?"*
