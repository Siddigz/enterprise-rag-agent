# Enterprise RAG & Reconciliation Agent

An agentic RAG service built with FastAPI and PostgreSQL (pgvector). It ingests, indexes and reconciles **schema drift** and **cross-source discrepancies** across distributed sales datasets. An **Apache Airflow** pipeline runs an **LLM-as-a-judge** benchmark suite to measure how grounded the agent's answers are.

> 🚧 Work in progress. Architecture, setup and measured eval results will be documented here.

## Stack
- **API:** FastAPI
- **Storage / retrieval:** PostgreSQL 16 + pgvector (HNSW), with hybrid vector + full-text search
- **Agent & judge:** Claude (Anthropic API), tool use
- **Orchestration:** Apache Airflow
- **Infra:** Docker Compose
