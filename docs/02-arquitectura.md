# 2. Arquitectura del Sistema

## Vista de Alto Nivel

```text
┌──────────────────────────────────────────────────────────────────┐
│                        USUARIO                                   │
│                    "¿Qué es LIME?"                               │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   STREAMLIT (cli.py)                             │
│              Interfaz visual tipo ChatGPT                        │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   PIPELINE (pipeline.py)                         │
│                  El cerebro que orquesta todo                    │
│                                                                  │
│  ┌──────────┐  ┌───────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  │
│  │ Router   │→ │ Cache │→ │ Context  │→ │Retrieval │→ │ LLM   │  │
│  │ (intent) │  │(check)│  │  ualizer │  │(búsqueda)│  │(gener)│  │
│  └──────────┘  └───────┘  └──────────┘  └──────────┘  └───────┘  │
└──────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ ChromaDB │ │   BM25   │ │  Groq    │
              │ (vectors)│ │(keywords)│ │  (LLM)   │
              └──────────┘ └──────────┘ └──────────┘
```

## Mapa de Archivos y Responsabilidades

```text
rag/
├── config.py          → Settings, Logger, Timer decorator
├── models.py          → DTOs (Intent, PipelineState)
├── exceptions.py      → Errores custom con jerarquía
├── providers.py       → Fábricas: LLM (Groq) + Embeddings (local)
├── pipeline.py        → ORQUESTADOR: une todo el flujo
├── cli.py             → Interfaz Streamlit
│
├── ingestion/
│   └── loader.py      → PDF → páginas → chunks
│
├── retrieval/
│   ├── vector_store.py → Wrapper de ChromaDB
│   ├── hybrid.py       → Vector + BM25 combinados
│   ├── hyde.py         → Generación de docs hipotéticos
│   ├── reranker.py     → FlashRank para filtrar relevancia
│   └── parent.py       → Expansión a páginas completas
│
├── chain/
│   ├── router.py       → Clasificación de intención
│   ├── contextualizer.py → Resolución de pronombres
│   ├── memory.py       → Historial de conversación
│   ├── cache.py        → Cache semántico con intención
│   └── prompts.py      → Templates del LLM
│
└── evaluation/
    ├── dataset.py      → Ground truth para tests
    └── evaluator.py    → Hit Rate @ K
```

# 3. Flujo Completo Paso a Paso

El sistema tiene dos fases principales que ocurren en momentos diferentes:

```text
FASE 1: INGESTION (ocurre UNA vez al cargar el documento)
  PDF → Lectura → Limpieza → Chunking → Embeddings → ChromaDB

FASE 2: CONVERSATION (ocurre en CADA pregunta del usuario)
  Query → Router → Cache → Contextualizer → HyDE → Hybrid Search
  → Reranking → Parent Expansion → LLM Generation → Respuesta
```
