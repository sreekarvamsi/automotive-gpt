# AutomotiveGPT 🚗💬

> **Open-source RAG system for automotive service manuals**  
> Ask questions, get answers with citations from 1,700+ pages of technical documentation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**[📖 Read the Article](#) | [🗺️ Roadmap](ROADMAP.md) | [🤝 Contributing](CONTRIBUTING.md) | [💬 Discussions](https://github.com/sreekarvamsi/automotive-gpt/discussions)**

---

## ⚡ Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/automotive-gpt.git
cd automotive-gpt

# One-command setup (creates venv, installs deps, sets up .env)
bash setup.sh

# Start the system
docker-compose up
```

Then ask a question:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the oil capacity for a 2022 Honda Civic?"}'
```

**Response:**
```json
{
  "answer": "The 2022 Honda Civic with CVT has an oil capacity of 3.7 US qt (3.5 L) including filter.",
  "sources": [{"source_file": "honda_civic_2022_manual.pdf", "page": 753}],
  "confidence": 0.83
}
```

---

## 🎯 Why This Project?

Automotive technicians spend **20+ minutes** searching through PDF manuals for simple specifications.  

**AutomotiveGPT** answers those questions in **<10 seconds** with **85% retrieval accuracy** and **full source citations**.

**Built for:**
- 🔧 **Technicians** who need fast, accurate answers
- 🏢 **Service departments** wanting to digitize documentation
- 👨‍💻 **Developers** learning production RAG systems
- 🔬 **Researchers** benchmarking retrieval methods

---

## ✨ Features

- ✅ **Hybrid Retrieval:** Dense semantic + BM25 sparse + Cohere reranking
- ✅ **Citation Support:** Every answer includes source document + page (95% accuracy)
- ✅ **Multi-Document Queries:** Compare specs across vehicles
- ✅ **5 Vehicle Manuals:** Honda, Toyota, Ford, Tesla + safety recalls (1,708 pages)
- ✅ **Production Ready:** Docker deployment, comprehensive tests, monitoring
- ✅ **Cost Optimized:** ~$0.045 per query

---

## 📊 Performance (v1.0)

| Metric | Value |
|--------|-------|
| **Success Rate** | 90% (18/20 queries) |
| **Document Retrieval Accuracy** | 85% |
| **Citation Rate** | 95% |
| **Average Latency** | 8.84s |
| **P95 Latency** | 21.29s |
| **Cost Per Query** | $0.045 |

Tested on 20 queries across 5 vehicle manuals.  
[📈 View detailed benchmark results](benchmarks/results/)

---

## 🏗️ Architecture
```
User Query → Hybrid Retrieval → Reranking → GPT-4 → Answer + Citations
             (Dense + BM25)       (Cohere)   (OpenAI)
```

**Pipeline:**
1. **Parse** PDFs with table extraction (PyMuPDF)
2. **Chunk** intelligently (512 tokens, 50 overlap, section-aware)
3. **Embed** with OpenAI text-embedding-3-large (3072-dim)
4. **Retrieve** with hybrid approach (dense + sparse)
5. **Rerank** with Cohere rerank-english-v3.0
6. **Generate** with GPT-4-turbo, strict grounding to prevent hallucinations

**Tech Stack:** Python • LangChain • OpenAI • Pinecone • Cohere • FastAPI • Streamlit • Docker

[📘 Full Architecture Documentation](docs/ARCHITECTURE.md)

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- API Keys: OpenAI, Pinecone, Cohere

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/automotive-gpt.git
cd automotive-gpt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

### Start Services

**Option A: Docker (Recommended)**
```bash
docker-compose up -d
```

**Option B: Manual**
```bash
# Start databases
docker-compose up -d postgres redis

# Ingest documents
python scripts/ingest.py --source data/

# Start API
uvicorn src.api.main:app --reload

# Start UI (new terminal)
streamlit run src/ui/app.py
```

**Access:**
- API: http://localhost:8000/docs
- UI: http://localhost:8501

---

## 💻 Usage Examples

### Python SDK
```python
from automotive_gpt import RAGSystem

# Initialize
rag = RAGSystem()

# Ask a question
result = rag.query("What is the towing capacity of a 2021 Ford F-150?")

print(result.answer)
# "The 2021 Ford F-150 has a maximum towing capacity of 14,000 lbs..."

print(result.sources)
# [{"source_file": "ford_f150_2021_manual.pdf", "page": 234, "score": 0.92}]
```

### REST API
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Compare oil capacity between Honda Civic and Toyota Camry",
    "session_id": "user123"
  }'
```

### Batch Processing
```python
queries = [
    "What is the tire pressure for Honda Civic?",
    "How to reset oil change light on Ford F-150?",
    "Tesla Model 3 battery capacity?"
]

for q in queries:
    result = rag.query(q)
    print(f"Q: {q}\nA: {result.answer}\n")
```

---

## 🗺️ Roadmap

**Current (v1.0):** Hybrid retrieval, 5 manuals, 90% success rate ✅  
**Next (v1.1):** Retry logic, <6s latency, parallel retrieval 🚧  
**Future (v2.0):** Multimodal support, 20+ manuals, fine-tuned reranker 🔮  

[📋 Full Roadmap](ROADMAP.md)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 [Report bugs](https://github.com/YOUR_USERNAME/automotive-gpt/issues/new?template=bug_report.md)
- 💡 [Request features](https://github.com/YOUR_USERNAME/automotive-gpt/issues/new?template=feature_request.md)
- 📖 Add vehicle manuals (see [guide](docs/ADDING_MANUALS.md))
- 💻 Submit pull requests
- ⭐ Star the repo!

[📋 Contributing Guidelines](CONTRIBUTING.md)

---

## 📝 Citation

If you use AutomotiveGPT in your research or project, please cite:
```bibtex
@software{automotivegpt2026,
  author = {Gajula, Sreekar},
  title = {AutomotiveGPT: Open-Source RAG System for Automotive Documentation},
  year = {2026},
  url = {https://github.com/sreekarvamsi/automotive-gpt}
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- [OpenAI](https://openai.com) - GPT-4 & embeddings
- [Pinecone](https://pinecone.io) - Vector database
- [Cohere](https://cohere.ai) - Reranking
- [LangChain](https://langchain.com) - RAG framework

---

## 📬 Contact

- **GitHub Issues:** [Report bugs](https://github.com/sreekarvamsi/automotive-gpt/issues)
- **Discussions:** [Ask questions](https://github.com/sreekarvamsi/automotive-gpt/discussions)
- **LinkedIn:** [sreekarvamsi](https://linkedin.com/in/sreekarvamsi)
- **Email:** sreekarvamsikrishnag@gmail.com

---

⭐ **If this project helps you, please star it!** ⭐

*Built for the automotive community*
