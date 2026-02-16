# 10. Resumen de Decisiones de Diseño

## ¿Por qué cada decisión?

```text
DECISIÓN: Embeddings locales (no API)
  MOTIVO: Privacidad + Costo + Latencia
  TRADEOFF: Calidad ligeramente inferior

DECISIÓN: Groq como LLM provider
  MOTIVO: Gratis + Latencia ultrarrápida + Llama 3.3 es excelente
  TRADEOFF: Rate limits en free tier

DECISIÓN: ChromaDB como vector store
  MOTIVO: Simple + Persiste en disco + Sin servidor externo
  TRADEOFF: No escala a millones de documentos (usar Pinecone/Weaviate)

DECISIÓN: FlashRank como reranker
  MOTIVO: Local + Rápido + Sin API
  TRADEOFF: Menos preciso que Cohere Rerank API

DECISIÓN: Cache con normalización de intención
  MOTIVO: Cache hits mucho más frecuentes
  TRADEOFF: 1 LLM call extra por query (muy barata, ~20 tokens)

DECISIÓN: Parent Document + Neighbors
  MOTIVO: Mejor contexto sin sacrificar precisión de búsqueda
  TRADEOFF: Más tokens enviados al LLM (más costo)

DECISIÓN: Estructura plana (4 carpetas)
  MOTIVO: Navegabilidad + Onboarding rápido de nuevos devs
  TRADEOFF: Archivos más largos que con 11 carpetas
```
