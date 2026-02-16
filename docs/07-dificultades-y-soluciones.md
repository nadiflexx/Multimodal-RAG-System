# 8. Dificultades Encontradas y Soluciones

---

## 8.1 Asimetría Query-Documento

```text
PROBLEMA:
  Las preguntas del usuario son cortas y usan lenguaje coloquial.
  Los documentos son largos y usan lenguaje técnico.
  La similitud coseno entre ambos es baja.

SOLUCIÓN: HyDE
  Generar un documento hipotético que "vive" en el mismo
  espacio semántico que los documentos reales.

  Mejora del recall: ~15-25% en nuestras pruebas
```

---

## 8.2 Búsqueda por Nombres Propios

```text
PROBLEMA:
  Query: "LIME"
  Vector search: Confunde con "lime" (limón), "limo", etc.
  Los embeddings no son buenos con nombres propios.

SOLUCIÓN: Hybrid Search (Vector + BM25)
  BM25 busca la palabra EXACTA "LIME" → La encuentra siempre
  Vector busca el concepto → Encuentra docs relacionados
  La fusión sube al top los docs que tienen AMBOS: palabra + concepto
```

---

## 8.3 Cache Misses con Paráfrasis

```text
PROBLEMA:
  "¿Dónde aparece LIME?"     → Se cachea
  "¿Dónde se habla de LIME?" → Cache MISS (redacción diferente)
  "¿En qué parte está LIME?" → Cache MISS (otra redacción)

  3 llamadas al LLM por la misma pregunta 😞

SOLUCIÓN: Normalización de intención + 2 niveles de cache
  Nivel 1: Normalizar texto (acentos, mayúsculas, puntuación)
  Nivel 2: Normalizar intención con LLM barato
    "donde aparece lime"     → "ubicacion lime"
    "donde se habla de lime" → "ubicacion lime"
    "en que parte esta lime" → "ubicacion lime"

  → Todas generan el mismo vector → Cache HIT ✅
```

---

## 8.4 Pronombres y Referencias

```text
PROBLEMA:
  User: "¿Qué es LIME?"
  Bot: "LIME es una técnica..."
  User: "¿Y dónde aparece?"  ← ¿Dónde aparece QUÉ?

  El sistema busca "dónde aparece" → No encuentra nada relevante

SOLUCIÓN: QueryContextualizer
  Usa el historial para resolver pronombres:
  "¿Y dónde aparece?" → "¿Dónde aparece LIME?"

  GUARD: Si el contextualizer expande demasiado, se rechaza
  (len(output) > len(input) * 3 → usar input original)
```

---

## 8.5 Chunks que Cortan Tablas/Listas

```text
PROBLEMA:
  Una tabla de resultados ocupa 2 páginas.
  El chunk solo captura la mitad de la tabla.
  El LLM da una respuesta parcial.

SOLUCIÓN: Parent Document Retriever + Neighbor Expansion
  1. Buscar con chunks pequeños (preciso)
  2. Retornar página completa (contexto)
  3. Incluir páginas vecinas (tablas que cruzan páginas)
```

---

## 8.6 Dependencias Conflictivas

```text
PROBLEMA:
  TypeError: 'NoneType' object is not subscriptable
  en huggingface_hub/utils/_runtime.py

  Incompatibilidad entre versiones de:
  - huggingface_hub
  - transformers
  - importlib_metadata

SOLUCIÓN:
  uv pip install --upgrade huggingface_hub transformers importlib_metadata
  O recrear el venv limpio: rm -rf .venv && uv venv && uv pip install -e .
```
