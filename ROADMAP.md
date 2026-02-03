# AutomotiveGPT Development Roadmap

## ✅ v1.0 (Current - February 2026)

**Released:** February 2, 2026

### Features
- [x] Hybrid retrieval (dense semantic + sparse BM25 + reranking)
- [x] 5 vehicle manuals ingested (1,708 pages)
- [x] FastAPI REST API with OpenAPI documentation
- [x] Streamlit web interface
- [x] Multi-turn conversation support
- [x] Citation tracking (95% citation rate)
- [x] Docker deployment

### Performance
- [x] 90% success rate (18/20 test queries)
- [x] 85% document retrieval accuracy
- [x] 8.84s average latency
- [x] 100% accuracy on multi-document comparisons
- [x] $0.045 cost per query

---

## 🚧 v1.1 (Week 1-2) - IN PROGRESS

### Focus: Reliability & Performance

- [ ] Retry logic for OpenAI API timeouts
- [ ] Reduce average latency to <6s
- [ ] Parallel retrieval (async dense + sparse)
- [ ] Better error messages for failed queries
- [ ] Request/response logging
- [ ] Cost tracking dashboard

**Target Metrics:**
- 95%+ success rate
- <6s average latency
- <15s P95 latency

---

## 📅 v1.2 (Week 2-3) - PLANNED

### Focus: Scale & Monitoring

- [ ] Add 10 more vehicle manuals (targeting 20+ total)
- [ ] Persistent BM25 cache (avoid rebuilding on restart)
- [ ] Batch ingestion API
- [ ] Prometheus metrics export
- [ ] Admin dashboard (Streamlit)
- [ ] Query analytics (most common questions, failure patterns)

**Target:**
- 20+ manuals indexed
- 50,000+ vectors in Pinecone
- Monitoring dashboard live

---

## 🔮 v2.0 - Upcomming

### Focus: Advanced Features

- [ ] **Multimodal support:** Extract and reference diagrams/images from PDFs
- [ ] **Fine-tuned reranker:** Train domain-specific reranker on automotive data
- [ ] **Query decomposition:** Automatically break complex queries into sub-queries
- [ ] **Context-aware chunking:** Preserve more document structure
- [ ] **Mobile app:** React Native interface
- [ ] **Plugin system:** Allow custom retrievers and generators
- [ ] **Offline mode:** Local LLM option (Llama 2, Mistral)

---

## 💡 Community Requests

*This section will be populated based on GitHub issues and discussions*

**How to request a feature:**
1. Check [existing issues](https://github.com/sreekarvamsi/automotive-gpt/issues)
2. If not found, [open a feature request](https://github.com/sreekarvamsi/automotive-gpt/issues/new?template=feature_request.md)
3. Upvote existing requests with 👍

---

## 🎯 Long-term Vision

**Goal:** Build the open-source standard for automotive technical documentation retrieval.

**Future directions:**
- Community-contributed manual database
- Multi-language support (Spanish, German, Japanese)
- Integration with diagnostic tools (OBD-II)
- Commercial support for dealerships/fleet operators
- Benchmarking suite for RAG systems
