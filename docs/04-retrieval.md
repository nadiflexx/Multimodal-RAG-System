# 5. Fase 2: RETRIEVAL (Búsqueda)

Cuando el usuario hace una pregunta, se ejecuta un pipeline de **7+ pasos** antes de generar la respuesta.

---

## Paso 2.1: Semantic Router (router.py)

```python
# ¿Qué hace?
# Clasifica la INTENCIÓN del usuario ANTES de buscar
# Es un guardrail (barrera de seguridad)

class SemanticRouter:
    def route(self, query: str) -> str:
        # Usa el LLM para clasificar en 3 categorías:
        # "SEARCH"   → Pregunta sobre el documento
        # "GREETING" → Saludo ("hola", "buenos días")
        # "OFF_TOPIC"→ Fuera de tema ("cuéntame un chiste")
        ...
```

### ¿Por qué un Router?

```text
Sin Router:
  User: "hola"
  → Sistema busca "hola" en ChromaDB
  → Encuentra chunks aleatorios con baja similitud
  → LLM genera respuesta confusa basada en contexto irrelevante
  → Gastamos 4 LLM calls + tiempo en búsqueda inútil

Con Router:
  User: "hola"
  → Router detecta: GREETING
  → Responde directamente: "¡Hola! ¿En qué puedo ayudarte?"
  → 1 LLM call, 0 búsquedas

  User: "tu madre"
  → Router detecta: OFF_TOPIC
  → Responde: "Solo respondo sobre el documento."
  → El sistema no pierde tiempo ni recursos
```

---

## Paso 2.2: Query Normalization (pipeline.py)

```python
def _normalize_query(self, query: str) -> str:
    """
    "¿Dónde se habla de LIME?"
    → "donde se habla de lime"
    """
    q = query.lower()

    # Quitar acentos: "ó" → "o"
    q = unicodedata.normalize("NFD", q)
    q = "".join(c for c in q if unicodedata.category(c) != "Mn")

    # Quitar puntuación: "¿?!¡" → ""
    q = re.sub(r"[¿?!¡.,;:\-\"']", "", q)

    # Normalizar espacios
    q = " ".join(q.split())
```

### ¿Por qué normalizar?

```text
Sin normalización, estas 3 queries son DIFERENTES para el cache:
  "¿Dónde se habla de LIME?"
  "donde se habla de lime"
  "Donde se habla de LIME???"

Con normalización, las 3 se convierten en:
  "donde se habla de lime"
  → Cache hit en la segunda y tercera pregunta
  → Ahorramos 2 llamadas al LLM
```

---

## Paso 2.3: Semantic Cache (cache.py)

Este es uno de los componentes más sofisticados. Funciona en dos niveles:

### NIVEL 1: Normalización de intención (LLM barato)

```python
# Reduce la query a su forma canónica

def _get_canonical_intent(self, query: str) -> str:
    # "donde aparece lime"     → "ubicacion lime"
    # "donde se define lime"   → "ubicacion lime"
    # "en que parte esta lime" → "ubicacion lime"
    ...
```

### NIVEL 2: Similitud por embeddings

```python
# Compara la intención canónica contra todas las cacheadas

def get(self, query: str) -> Optional[Tuple[str, List[Document]]]:
    canonical = self._get_canonical_intent(query)
    query_vector = self.embeddings.embed_query(canonical)

    # Calcular similitud contra TODOS los vectores cacheados
    # (usando álgebra lineal, muy rápido)
    similarities = self._vectors @ query_norm

    best_score = max(similarities)
    if best_score >= 0.95:  # 95% similar
        return cached_response  # ¡Cache HIT!
```

### ¿Por qué dos niveles?

```text
Solo con embeddings (Nivel 2):
  "donde aparece lime" → embedding A
  "donde se define lime" → embedding B
  similitud(A, B) = 0.80 → MISS (debajo del threshold 0.95)
  → Ejecutamos TODO el pipeline de nuevo 😞

Con normalización de intención (Nivel 1 + 2):
  "donde aparece lime"   → LLM → "ubicacion lime" → embedding X
  "donde se define lime"  → LLM → "ubicacion lime" → embedding Y
  similitud(X, Y) = 1.00 → HIT ✅
  → Retornamos respuesta cacheada en milisegundos 🚀
```

### Eviction Strategy (LFU con decay temporal)

```python
def _evict_least_used(self):
    # Cuando el cache está lleno (500 entradas), eliminamos la peor
    # Score = veces_usado / horas_de_vida

    # Ejemplo:
    # Entry A: usada 10 veces, creada hace 2 horas → score = 5.0
    # Entry B: usada 1 vez, creada hace 24 horas   → score = 0.04
    # → Eliminamos Entry B (poco usada Y vieja)
    ...
```

---

## Paso 2.4: Query Contextualization (contextualizer.py)

```python
# ¿Qué hace?
# Resuelve pronombres y referencias ambiguas usando el historial

class QueryContextualizer:
    def contextualize(self, query: str, history: list) -> str:
        # Historial: El usuario preguntó sobre LIME

        # query = "¿En qué página está?"
        # → "¿En qué página está LIME?"

        # query = "¿Y eso qué ventajas tiene?"
        # → "¿Qué ventajas tiene LIME?"
        ...
```

### ¿Por qué es necesario?

```text
Conversación:
  User: "¿Qué es LIME?"
  Bot: "LIME es una técnica de explicabilidad..."
  User: "¿En qué página está?"           ← ¿QUÉ está?

  Sin contextualización:
    Busca "en qué página está" en ChromaDB
    → No encuentra nada relevante (la query no tiene sustancia)

  Con contextualización:
    "¿En qué página está?" → "¿En qué página está LIME?"
    Busca "en qué página está LIME" en ChromaDB
    → Encuentra chunks relevantes sobre LIME
```

### Guard de seguridad

```python
# El contextualizer puede "expandir de más"
# Guard: rechazar si la salida es 3x más larga que la entrada

if len(new_query) > len(query) * 3:
    return query  # Fallback a la original

# Ejemplo que se rechaza:
# Input:  "lime" (4 chars)
# Output: "¿Qué es la técnica LIME y cómo funciona en el contexto
#          de explicabilidad de modelos de machine learning?" (90 chars)
# 90 > 4*3=12 → RECHAZADO, se usa "lime" directamente
```

---

## Paso 2.5: HyDE — Hypothetical Document Embeddings (hyde.py)

```python
# ¿Qué hace?
# Genera un "documento ficticio" que RESPONDERÍA la pregunta
# y lo usa para buscar documentos REALES similares

class HyDEGenerator:
    def generate(self, query: str) -> str:
        # Input: "¿Qué es LIME?"
        # Output: "LIME (Local Interpretable Model-agnostic Explanations)
        #          es una técnica de explicabilidad que genera
        #          perturbaciones locales para aproximar el comportamiento
        #          de un modelo complejo con un modelo interpretable..."
        ...
```

### ¿Por qué funciona mejor que buscar directamente?

```text
PROBLEMA: Asimetría query-documento

  La QUERY del usuario:    "¿Qué es LIME?"  (pregunta corta)
  El DOCUMENTO real dice:  "LIME (Local Interpretable Model-agnostic
                           Explanations) es una técnica..." (texto largo)

  Embedding de la query:     [0.1, 0.2, ...]  → Espacio de "preguntas"
  Embedding del documento:   [0.3, 0.1, ...]  → Espacio de "respuestas"

  Similitud: 0.65 → No tan alta como debería ser
  (porque preguntas y respuestas están en "zonas" diferentes del espacio)

SOLUCIÓN: HyDE

  La query del usuario:     "¿Qué es LIME?"
  Doc hipotético generado:  "LIME es una técnica que..." (texto largo)
  El documento REAL dice:    "LIME es una técnica que..." (texto largo)

  Embedding del hipotético: [0.28, 0.12, ...]  → Espacio de "respuestas"
  Embedding del real:       [0.30, 0.10, ...]  → Espacio de "respuestas"

  Similitud: 0.92 → ¡Mucho mejor!
  (porque ambos son "respuestas", están en la misma zona)
```

### Visualización

```text
                    Espacio de Embeddings

    "preguntas"                    "respuestas"
    zone                           zone

    ● query                        ● doc_real
    "¿Qué es LIME?"               "LIME es una técnica..."

         ╲                        ╱
          ╲  distancia grande    ╱
           ╲                    ╱
            ╲                  ╱
             ╲                ╱
              ╲              ╱

    CON HyDE:
              ● doc_hipotético
              "LIME es una técnica..."
                    │
                    │ distancia pequeña
                    │
              ● doc_real
              "LIME es una técnica..."
```

---

## Paso 2.6: Triple Strategy Search

Para maximizar la capacidad de recuperación (recall) y cubrir todos los tipos de consultas posibles, el sistema ejecuta **tres estrategias de búsqueda en paralelo**.

### Las 3 Estrategias

1. **HyDE + Hybrid (Conceptual)**:
   - Genera un documento hipotético que responde a la pregunta.
   - Busca vectores similares a esa respuesta hipotética.
   - Ideal para preguntas complejas o abstractas ("¿Cómo funciona la explicabilidad?").

2. **Direct Hybrid (Mixta)**:
   - Busca vectores de la pregunta original.
   - Busca palabras clave exactas (BM25).
   - Ideal para preguntas estándar con terminología técnica.

3. **Direct Vector (Estructural/Corta)**:
   - Busca vectores directamente en la base de datos sin intermediarios.
   - Ideal para preguntas muy cortas o estructurales ("bibliografía", "índice") donde HyDE puede alucinar y BM25 puede fallar por falta de contexto.

````python
# Pseudo-código de la fusión de estrategias

# 1. HyDE
hyde_docs = vector_search(hypothetical_doc)

# 2. Direct Hybrid
keyword_docs = bm25_search(original_query)

# 3. Direct Vector
direct_docs = vector_search(original_query)

# Fusión
all_candidates = unique(hyde_docs + keyword_docs + direct_docs)

### ¿Por qué combinar dos estrategias?

```text
Caso 1: VECTOR gana (Semántica)
  Query: "técnicas para explicar modelos"
  Documento: "...métodos interpretativos para cajas negras..."

  → BM25 falla: no hay coincidencia de palabras ("explicar" != "interpretativos").
  → Vector acierta: entiende que significan lo mismo.

Caso 2: BM25 gana (Exactitud)
  Query: "LIME"
  Documento: "La técnica LIME se define como..."

  → Vector puede fallar: confunde "LIME" con "lima" (fruta) o conceptos generales.
  → BM25 acierta: encuentra la palabra exacta "LIME".

Caso 3: DIRECT VECTOR gana (Estructural)
  Query: "bibliografía"
  Documento: "BIBLIOGRAFÍA [1]..."

  → HyDE falla: genera un texto largo sobre qué es una bibliografía, alejando el vector.
  → BM25 falla: si el chunk es muy heterogéneo (lista de autores), el score de la palabra es bajo.
  → Direct Vector acierta: el embedding de "bibliografía" está muy cerca del título "BIBLIOGRAFÍA".
````

### Uso inteligente de queries diferentes

```text
┌─────────────────────────────────────────────────────────┐
│  Este es un truco avanzado que mejora mucho la calidad: │
│                                                         │
│  Vector Search ← usa el doc HIPOTÉTICO (HyDE)           │
│    "LIME es una técnica de explicabilidad que..."       │
│    → Busca por significado profundo                     │
│                                                         │
│  BM25 Search ← usa la query ORIGINAL                    │
│    "¿Qué es LIME?"                                      │
│    → Busca la palabra "LIME" exacta                     │
│                                                         │
│  Cada retriever recibe la query ÓPTIMA para su tipo     │
└─────────────────────────────────────────────────────────┘
```

---

## Paso 2.7: Reranking (reranker.py)

```python
# ¿Qué hace?
# Reordena los candidatos por relevancia REAL usando un modelo
# entrenado específicamente para determinar relevancia

class Reranker:
    def __init__(self):
        # FlashRank: modelo local, rápido, sin API
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def rerank(self, query, documents, top_n=3):
        # Input: 10 candidatos del hybrid search
        # Output: Los 3 más relevantes, ordenados por score
        ...
```

### ¿Por qué reranking si ya tenemos embeddings?

```text
Los embeddings son RÁPIDOS pero IMPRECISOS:
  Son bi-encoders: codifican query y documento POR SEPARADO
  Luego comparan los vectores con similitud coseno
  → No ven la INTERACCIÓN entre query y documento

El reranker es LENTO pero PRECISO:
  Es un cross-encoder: analiza query + documento JUNTOS
  Entiende la relación profunda entre ambos
  → Pero solo puede procesar ~10 docs (no 10,000)

COMBINACIÓN ÓPTIMA:
  1. Embeddings filtran 10,000 → 10 candidatos (rápido, recall alto)
  2. Reranker reordena 10 → 3 mejores (lento, precisión alta)
```

```text
Ejemplo:

  Query: "¿Cómo funciona la explicabilidad local?"

  Después del hybrid search (10 candidatos):
    1. "LIME genera perturbaciones..."        (relevante ✅)
    2. "El modelo se entrena con Adam..."      (irrelevante ❌)
    3. "La explicabilidad es un campo..."      (parcial ⚠️)
    4. "Las predicciones locales permiten..."  (relevante ✅)
    ...

  Después del reranking (top 3):
    1. "LIME genera perturbaciones..."         score=0.95
    2. "Las predicciones locales permiten..."  score=0.87
    3. "La explicabilidad es un campo..."      score=0.72

  → Los irrelevantes se eliminaron
  → Los relevantes subieron al top
```

---

## Paso 2.8: Parent Document Expansion (parent.py)

```python
# ¿Qué hace?
# Reemplaza los chunks pequeños por sus páginas completas

def get_parents_for_chunks(self, chunks, expand_neighbors=True):
    for chunk in chunks:
        page_num = chunk.metadata["page"]  # Este chunk viene de pág 5

        pages_to_fetch = [page_num]        # Traer pág 5

        if expand_neighbors:
            pages_to_fetch.append(page_num - 1)  # También pág 4
            pages_to_fetch.append(page_num + 1)  # También pág 6

    # Deduplicar: si dos chunks vienen de la misma página,
    # solo la incluimos una vez
    ...
```

### ¿Por qué expandir a vecinos?

```text
Caso: Tabla que cruza dos páginas

  Página 5: "Tabla 3. Resultados del modelo:
             | Modelo | Accuracy | Recall |
             | RF     | 0.85     | 0.78   |"

  Página 6: "| SVM    | 0.92     | 0.88   |
             | LIME   | N/A      | N/A    |
             Conclusión: SVM obtuvo los mejores..."

  Si el chunk encontrado está en pág 5,
  sin expansión el LLM solo ve la mitad de la tabla.
  Con expansión a pág 6, ve la tabla completa.
```

---

## Resumen Visual del Retrieval

```text
         "¿Qué es LIME?"
              │
              ▼
    ┌───────────────────┐
    │ Router: SEARCH ✅  │ ← Guardrail
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐
    │ Cache: MISS ❌     │ ← No hay respuesta cacheada
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐
    │ Contextualizer    │ ← Sin historial, no cambia nada
    │ "¿Qué es LIME?"  │
    └────────┬──────────┘
             │
             ▼
    ┌───────────────────┐     ┌──────────────────────┐
    │ HyDE Generator    │────→│ "LIME es una técnica │
    │                   │     │ de explicabilidad..."│
    └────────┬──────────┘     └──────────┬───────────┘
             │                           │
             ▼                           ▼
    ┌───────────────────────────────────────────────┐
    │ Hybrid Search                                 │
    │                                               │
    │  Hyde Search ←── doc hipotético (semántica)   │
    │  BM25 Search   ←── query original (keywords)  │
    │  Direct Vector ←── query original (estructura)│
    │  → X candidatos fusionados                    │
    └────────────────────┬──────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────┐
    │ FlashRank Reranker                            │
    │ X candidatos → 5 más relevantes               │
    └────────────────────┬──────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────┐
    │ Parent Expansion                              │
    │ 3 chunks → 5-7 páginas completas              │
    │ (con vecinos para contexto)                   │
    └────────────────────┬──────────────────────────┘
                         │
                         ▼
                  CONTEXTO LISTO
                  PARA EL LLM
```
