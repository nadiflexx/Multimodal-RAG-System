# 9. Técnicas Avanzadas Aplicadas

## Resumen de todas las técnicas

```text
┌───────────────────────────┬──────────────────────────────────────┐
│ TÉCNICA                   │ QUÉ RESUELVE                         │
├───────────────────────────┼──────────────────────────────────────┤
│ Recursive Chunking        │ Chunks que respetan estructura       │
│ Chunk Overlap             │ Frases cortadas entre chunks         │
│ HyDE                      │ Asimetría query-documento            │
│ Hybrid Search             │ Nombres propios + semántica          │
│ Cross-Encoder Reranking   │ Precisión en top results             │
│ Parent Document Retriever │ Contexto insuficiente para LLM       │
│ Neighbor Expansion        │ Tablas/listas que cruzan páginas     │
│ Semantic Router           │ Queries irrelevantes (guardrail)     │
│ Query Contextualization   │ Pronombres y referencias ambiguas    │
│ 2-Level Semantic Cache    │ Consultas repetidas con paráfrasis   │
│ LFU Eviction              │ Cache lleno (gestión de memoria)     │
│ Intent Normalization      │ Cache misses por redacción diferente │
│ Sliding Window Memory     │ Límite de contexto del LLM           │
│ Singleton Pattern         │ Recarga innecesaria de modelos       │
│ Pydantic Settings         │ Validación de configuración          │
│ Custom Exception Hierarchy│ Manejo de errores granular           │
│ Temperature=0             │ Determinismo en respuestas           │
└───────────────────────────┴──────────────────────────────────────┘
```
