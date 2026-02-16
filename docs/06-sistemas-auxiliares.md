# 7. Sistemas Auxiliares

---

## 7.1 Chat Memory (memory.py)

```python
class ChatState:
    def __init__(self, window_size: int = 6):
        self.history = []
        self.window_size = window_size

    def get_history(self):
        return self.history[-6:]  # Solo los últimos 6 mensajes
```

### ¿Por qué ventana deslizante de 6?

```text
Problema: Los LLMs tienen un LÍMITE de contexto (tokens)

  Si la conversación tiene 100 mensajes:
    Enviar los 100 → Excede el límite del LLM → Error

  Solución: Solo enviar los últimos 6 (3 pares user/assistant):
    Suficiente para resolver pronombres y mantener coherencia
    No excede el contexto del LLM

  ¿Por qué 6 y no 10?
    Cada mensaje consume tokens del contexto
    Más historial = menos espacio para los documentos recuperados
    6 mensajes es un buen balance entre memoria y espacio
```

---

## 7.2 Error Handling (exceptions.py)

```python
class RAGException(Exception):          # Base
    ├── ConfigurationError              # .env mal configurado
    ├── DocumentNotFoundError           # PDF no existe
    ├── VectorStoreNotInitializedError  # Buscar sin cargar doc
    ├── RetrievalError                  # Fallo en búsqueda
    ├── EmbeddingError                  # Modelo de embeddings falla
    ├── LLMError                        # API de Groq falla
    ├── CacheError                      # Cache corrupto
    └── IngestionError                  # PDF ilegible
```

### ¿Por qué jerarquía de excepciones?

```python
# En la CLI puedes capturar a diferentes niveles:

try:
    response = pipeline.run_conversation_flow(query)
except VectorStoreNotInitializedError:
    # Error específico: usuario no cargó documento
    print("Carga un documento primero")
except RAGException as e:
    # Cualquier error del sistema RAG
    print(f"Error del sistema: {e}")
except Exception as e:
    # Errores inesperados (bugs)
    print(f"Error inesperado: {e}")
```

---

## 7.3 Configuration (config.py)

```python
class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CHROMA_PATH: Path = BASE_DIR / "chroma_db"
    DATA_DIR: Path = BASE_DIR / "data"
    GROQ_API_KEY: SecretStr = Field(..., min_length=1)

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",  # Carga automática del .env
    )
```

### ¿Por qué Pydantic Settings?

```text
Sin Pydantic:
  api_key = os.getenv("GROQ_API_KEY")
  # ¿Qué pasa si no existe? → api_key = None → Error críptico después
  # ¿Qué pasa si está vacía? → api_key = "" → Error críptico después

Con Pydantic:
  GROQ_API_KEY: SecretStr = Field(..., min_length=1)
  # Si no existe → Error CLARO al arrancar: "GROQ_API_KEY is required"
  # Si está vacía → Error CLARO: "min_length is 1"
  # SecretStr → No se muestra en logs (seguridad)
```

### Configuración Dinámica

Además de las variables de entorno, la configuración de ingestion (chunk size, strategy) se inyecta dinámicamente desde la interfaz de Streamlit al inicializar el `RAGPipeline`.
