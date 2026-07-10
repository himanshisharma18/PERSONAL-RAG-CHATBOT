# 📚 Personal Knowledge Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **FAISS**, **Sentence Transformers**, and **Groq Llama 3.1**. Upload your own PDF or text documents, build a searchable knowledge base, and ask questions grounded in your documents.

---

## 🚀 Features

- 📄 Upload PDF and TXT documents
- 🔍 Automatic document chunking
- 🧠 Semantic search using Sentence Transformers
- ⚡ Fast vector retrieval with FAISS
- 🤖 Llama 3.1 (Groq) powered responses
- 💬 Interactive Streamlit interface
- 🔄 Rebuild knowledge base anytime

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web Interface |
| Sentence Transformers | Text Embeddings |
| FAISS | Vector Database |
| PyPDF | PDF Text Extraction |
| Groq API | Large Language Model |
| NumPy | Numerical Operations |

---

## 📂 Project Structure

```
personal-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── uploads/              # Uploaded documents
├── faiss_index/          # Saved vector database
└── utils/                # Helper functions (if applicable)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/personal-rag-chatbot.git

cd personal-rag-chatbot
```

---

### 2. Create a virtual environment

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision
```

---

### 4. Set your API Key

Create a **.env** file in the project root.

```
LLM_API_KEY=YOUR_GROQ_API_KEY
```

Replace `YOUR_GROQ_API_KEY` with your own API key.

---

### 5. Run the application

```bash
streamlit run app.py
```

The application will be available at

```
http://localhost:8501
```

---

# 📖 How It Works

1. Upload PDF or TXT documents.
2. Click **Build / Rebuild Index**.
3. The application:
   - Extracts text
   - Splits documents into chunks
   - Generates embeddings
   - Stores vectors in FAISS
4. Ask questions.
5. Relevant document chunks are retrieved and sent to the LLM.
6. The chatbot answers based only on your uploaded documents.

---

# 🖥️ Demo

### Home Screen

> Add a screenshot here

```
images/home.png
```

---

# 📦 Requirements

Main dependencies

```
streamlit
sentence-transformers
faiss-cpu
torch
torchvision
transformers
pypdf
requests
numpy
```

---

# 🔒 Environment Variables

| Variable | Description |
|----------|-------------|
| LLM_API_KEY | Groq API Key |

---

# 💡 Future Improvements

- Conversation Memory
- Multiple Knowledge Bases
- Hybrid Search (BM25 + Dense Retrieval)
- Document Source Citations
- OCR Support for Scanned PDFs
- Drag-and-Drop Upload
- Docker Support
- Cloud Deployment

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```
git checkout -b feature-name
```

3. Commit your changes

```
git commit -m "Added new feature"
```

4. Push

```
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Himanshi Sharma**

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: https://linkedin.com/in/YOUR_LINKEDIN

---

⭐ If you found this project useful, please consider giving it a star!
