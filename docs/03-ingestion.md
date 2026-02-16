# 4. Fase 1: INGESTION (Preparación del Documento)

## ¿Qué es?

**Ingestion** es el proceso de preparar un documento para que sea buscable.  
Es como crear el índice de un libro antes de poder buscar en él.

---

## Paso 1.1: Carga del PDF (loader.py)

```python
# ¿Qué hace?
# Lee el PDF página por página y extrae el texto

def load_pdf(self, file_path: Path) -> List[Document]:
    reader = PdfReader(file)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        # Limpieza: quitar saltos de línea innecesarios, espacios dobles
        clean_text = " ".join(text.split())

        # Metadata: información SOBRE el texto, no el texto en sí
        metadata = {
            "source": file_path.name,      # "data.pdf"
            "page": page_num + 1,           # Página 1, 2, 3...
            "is_title_page": page_num == 0, # ¿Es la portada?
        }

        doc = Document(page_content=clean_text, metadata=metadata)
```

### ¿Por qué limpiar el texto?

Los PDFs tienen formato interno caótico. Un párrafo puede tener saltos de línea cada 80 caracteres.  
Sin limpieza, los chunks se cortan en mitad de palabras.

```text
ANTES (texto crudo del PDF):
  "La técnica LIME
(Local Interpretable
Model-agnostic
Explanations)
   permite
explicar..."

DESPUÉS (limpiado):
  "La técnica LIME (Local Interpretable Model-agnostic Explanations)
   permite explicar..."
```

---

## Paso 1.2: Chunking (División en fragmentos)

El sistema soporta **dos estrategias** configurables desde la UI:

Estrategia 1: Recursive Character Splitting

Divide por tamaño fijo, respetando la estructura del texto.

- **Chunk Size**: 1000 caracteres (default).
- **Overlap**: 200 caracteres (default).

```python
# ¿Qué hace?
# Corta por longitud fija intentando respetar párrafos, líneas y espacios:

self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Máximo 500 caracteres por chunk
    chunk_overlap=100,    # 100 caracteres se repiten entre chunks vecinos
    add_start_index=True, # Guarda la posición del chunk en la página
)
```

Estrategia 2: Semantic Chunking (actual)

Utiliza embeddings para detectar cambios temáticos en el texto.

- **Breakpoint Threshold**: Percentil de diferencia (default 85).
- **Buffer Size**: Ventana de suavizado (default 3).

from langchain_experimental.text_splitter import SemanticChunker

```python
# ¿Qué hace?
# Corta por significado, no por longitud. Usa embeddings para detectar dónde cambia el tema:
self.text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
)
```

### ¿Cómo funciona?

```text
¿Cómo funciona?

  1. Divide el texto en oraciones
  2. Genera embedding de cada oración
  3. Calcula similitud entre oraciones consecutivas
  4. Si la similitud BAJA drásticamente → Corta ahí
     (significa que el tema cambió)

  Ejemplo:
    Oración 1: "LIME genera perturbaciones locales"     ─┐
    Oración 2: "Estas perturbaciones aproximan el modelo" ─┤ sim=0.89 → Mismo tema
    Oración 3: "El modelo lineal resultante es interpretable" ─┘
    --- CORTE (sim=0.35) --- → El tema cambia aquí
    Oración 4: "En el sector financiero, los bancos..."  ─┐
    Oración 5: "La regulación exige transparencia"       ─┘ Nuevo tema
```

### ¿Por qué hacer chunking?

```text
PROBLEMA SIN CHUNKING:
  Página completa = ~3000 caracteres
  Si el usuario pregunta "¿Qué es LIME?"
  La respuesta está en las líneas 5-10 de la página 3
  Pero le pasaríamos TODA la página 3 al LLM → Ruido innecesario

CON CHUNKING:
  Página 3 se divide en 6 chunks de ~500 chars
  Solo el chunk 2 contiene la definición de LIME
  Le pasamos SOLO ese chunk relevante → Respuesta más precisa
```

### ¿Por qué Semantic Chunking es mejor?

```text
Recursive (por longitud):
  Chunk 1: "LIME genera perturbaciones locales. Estas perturbaciones
            aproximan el modelo. El modelo lineal resultante es
            interpretable. En el sector financiero, los bancos..."
  → MEZCLA dos temas en un solo chunk (LIME + finanzas)
  → El embedding del chunk es confuso (¿de qué habla?)

Semantic (por significado):
  Chunk 1: "LIME genera perturbaciones locales. Estas perturbaciones
            aproximan el modelo. El modelo lineal resultante es
            interpretable."
  Chunk 2: "En el sector financiero, los bancos..."
  → Cada chunk habla de UN solo tema
  → Los embeddings son más precisos
  → La búsqueda encuentra exactamente lo que necesita
```

### ¿Qué es breakpoint_threshold_amount=85?

```text
Es el percentil de disimilitud para decidir dónde cortar:

  85 = "Corta cuando la diferencia entre oraciones consecutivas
        está en el top 15% más alto de todas las diferencias"

  Threshold BAJO (50):
    → Corta mucho → Chunks muy pequeños → Muchos chunks
    → Más preciso pero fragmentado

  Threshold ALTO (95):
    → Corta poco → Chunks muy grandes → Pocos chunks
    → Menos fragmentado pero puede mezclar temas

  85 es un buen balance para documentos técnicos en español.
```

---

## Paso 1.3: Embeddings (Vectorización)

```python
# ¿Qué hace?
# Convierte cada chunk de texto en un vector numérico
# que captura su SIGNIFICADO semántico

# Modelo MULTILINGÜE (entiende español, inglés, +50 idiomas)
embeddings = LocalPyTorchEmbeddings()
# Modelo: paraphrase-multilingual-MiniLM-L12-v2

```

Utilizamos un modelo **Multilingüe** local:

- **Modelo**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Motivo**: El modelo anterior (`all-MiniLM`) solo entendía inglés, fallando en queries en español como "agradecimientos". Este modelo captura la semántica correctamente en >50 idiomas.

### ¿Cómo funciona la similitud semántica?

```text
Imagina un espacio de 384 dimensiones (imposible de visualizar,
pero funciona igual que 2D/3D):

  "¿Qué es LIME?" ────→ Punto A en el espacio
  "Define LIME"    ────→ Punto B (muy CERCA de A)
  "Receta de pastel" ──→ Punto C (muy LEJOS de A y B)

  Similitud coseno(A, B) = 0.95  → Muy similares
  Similitud coseno(A, C) = 0.12  → Muy diferentes
```

### ¿Por qué un modelo LOCAL y no la API de OpenAI?

```text
OpenAI Embeddings (API):
  ✅ Mejor calidad
  ❌ Cuesta dinero por cada llamada
  ❌ Tus datos salen de tu servidor
  ❌ Depende de Internet

Local (paraphrase-multilingual-MiniLM-L12-v2):
  ✅ Gratis
  ✅ Tus datos nunca salen
  ✅ Funciona offline
  ✅ Entiende español nativo
  ✅ Suficientemente bueno para la mayoría de casos
```

### Singleton Pattern en Embeddings

```python
class LocalPyTorchEmbeddings(Embeddings):
    _instances: Dict[str, "LocalPyTorchEmbeddings"] = {}
    _lock: Lock = Lock()

    def __new__(cls, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        with cls._lock:
            if model_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialize_model(model_name)
                cls._instances[model_name] = instance
            return cls._instances[model_name]
```

#### ¿Por qué Singleton?

```text
Sin Singleton:
  pipeline.py crea LocalPyTorchEmbeddings() → Carga modelo (3 seg)
  cache.py crea LocalPyTorchEmbeddings()    → Carga modelo OTRA VEZ (3 seg)
  hyde.py crea LocalPyTorchEmbeddings()     → Carga modelo OTRA VEZ (3 seg)
  Total: 9 segundos, 3 copias en RAM

Con Singleton:
  pipeline.py crea LocalPyTorchEmbeddings() → Carga modelo (3 seg)
  cache.py crea LocalPyTorchEmbeddings()    → Retorna LA MISMA instancia
  hyde.py crea LocalPyTorchEmbeddings()     → Retorna LA MISMA instancia
  Total: 3 segundos, 1 copia en RAM
```

---

## Paso 1.4: Indexación en ChromaDB (vector_store.py)

```python
# ¿Qué hace?
# Guarda los chunks + sus vectores en una base de datos vectorial

vector_store = VectorStore(
    collection_name="data_pdf",          # Nombre de la "tabla"
    embedding_model=self.embeddings,     # Modelo para vectorizar
)

# Internamente usa ChromaDB:
self.db = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=str(settings.CHROMA_PATH),  # Guarda en disco
    collection_metadata={"hnsw:space": "cosine"}, # Métrica de similitud
)
```

Reindexación automática

Cuando el usuario sube un documento nuevo, el sistema reemplaza
la colección anterior automáticamente:

```python
def replace_documents(self, documents):
    self.clear_collection()  # Borra los vectores anteriores
    self.db.add_documents(documents)  # Indexa los nuevos
```

### ¿Qué guarda ChromaDB?

```text
┌─────────────────────────────────────────────────────────────┐
│  ChromaDB Collection: "data_pdf"                            │
│                                                             │
│  ID    │ Texto (chunk)              │ Vector        │ Meta  │
│  ──────┼────────────────────────────┼───────────────┼───────│
│  id_1  │ "LIME es una técnica..."   │ [0.02, -0.15] │ p=3   │
│  id_2  │ "El modelo se entrena..."  │ [0.11, 0.08]  │ p=4   │
│  id_3  │ "Los resultados muestran"  │ [-0.05, 0.22] │ p=7   │
│  ...   │ ...                        │ ...           │ ...   │
└─────────────────────────────────────────────────────────────┘
```

### ¿Por qué hnsw:space: cosine?

HNSW (Hierarchical Navigable Small World) es el algoritmo de búsqueda:

```text
Búsqueda bruta:    Compara query contra TODOS los vectores → O(n)
Búsqueda HNSW:     Usa un grafo navegable para saltar a los vecinos → O(log n)

Con 10,000 chunks:
  Bruta: 10,000 comparaciones
  HNSW:  ~50 comparaciones → 200x más rápido
```

---

## Paso 1.5: Parent Document Store (parent.py)

```python
# ¿Qué hace?
# Guarda las páginas COMPLETAS (sin chunking) como "padres"
# Cada chunk sabe de qué página viene

full_pages = self.loader.load_pdf_full_pages(file_path)
self.parent_store.store_parents(full_pages)

# Resultado en memoria:
# parents = {
#     1: Document("Texto completo página 1..."),
#     2: Document("Texto completo página 2..."),
#     3: Document("Texto completo página 3..."),
# }
```

### ¿Por qué guardar páginas completas si ya tenemos chunks?

```text
El dilema del chunking:

  Chunks PEQUEÑOS (200 chars):
    ✅ Búsqueda muy precisa (encuentra exactamente lo relevante)
    ❌ El LLM no tiene suficiente contexto para responder bien

  Chunks GRANDES (2000 chars):
    ✅ El LLM tiene mucho contexto
    ❌ Búsqueda imprecisa (mucho ruido mezclado con la señal)

  Solución: Parent Document Retriever
    1. Buscar con chunks PEQUEÑOS (precisión)
    2. Retornar la página COMPLETA del chunk encontrado (contexto)
    → Lo mejor de ambos mundos
```

---

## Paso 1.6: BM25 Index (dentro de hybrid.py)

```python
# ¿Qué hace?
# Crea un índice de búsqueda por palabras clave (como Google antiguo)

bm25_retriever = BM25Retriever.from_documents(documents)
```

### ¿Qué es BM25?

```text
BM25 es un algoritmo de ranking que puntúa documentos
por coincidencia de PALABRAS, no de significado:

  Query: "LIME explicabilidad"

  Chunk 1: "LIME es una técnica de explicabilidad"
    → Score alto (contiene "LIME" Y "explicabilidad")

  Chunk 2: "Los métodos interpretativos permiten entender modelos"
    → Score bajo (misma idea, pero palabras diferentes)

  Chunk 3: "El limón (lime en inglés) es una fruta cítrica"
    → Score medio (contiene "lime" pero no es relevante)
```

---

## Resumen Visual de Ingestion

```text
                    PDF
                     │
                     ▼
              ┌─────────────┐
              │ PdfReader   │ Extrae texto crudo
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Limpieza    │ Normaliza espacios, saltos de línea
              └──────┬──────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
    ┌─────────────┐  ┌──────────────┐
    │ Semantic    │  │ Páginas      │
    │ Chunking    │  │ completas    │
    │ (por tema)  │  │ (parents)    │
    └──────┬──────┘  └──────┬───────┘
           │                │
           ▼                ▼
    ┌─────────────┐  ┌──────────────┐
    │ ChromaDB    │  │ Parent Store │
    │ + BM25      │  │ (in-memory)  │
    │ (vectores)  │  │              │
    └─────────────┘  └──────────────┘

    LISTO PARA BÚSQUEDAS
```
