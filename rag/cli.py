"""
Streamlit interface for the RAG system.

Features:
- Chat-like interface (ChatGPT style).
- Document upload and processing.
- Dynamic configuration of chunking strategies.
- Session state management.

Run with: streamlit run main.py
"""

from pathlib import Path

import streamlit as st
from loguru import logger

from rag.config import settings
from rag.exceptions import RAGException
from rag.pipeline import RAGPipeline

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════
# CUSTOM CSS - ChatGPT Style
# ═══════════════════════════════════════════════════════


def inject_custom_css() -> None:
    """Injects custom CSS to style the Streamlit app like ChatGPT."""
    st.markdown(
        """
    <style>
    /* ─── GENERAL ─── */
    .stApp {
        background-color: #212121;
    }

    /* ─── HEADER ─── */
    header[data-testid="stHeader"] {
        background-color: #212121;
        border-bottom: 1px solid #2f2f2f;
    }

    /* ─── SIDEBAR ─── */
    section[data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2f2f2f;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] label {
        color: #ececec !important;
    }

    /* ─── CHAT MESSAGES ─── */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1rem 0 !important;
    }

    /* User message bubble */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background-color: #2f2f2f !important;
        border-radius: 1.5rem !important;
        padding: 0.75rem 1.25rem !important;
        margin-left: 20% !important;
    }

    /* Assistant message */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background-color: transparent !important;
        padding: 0.75rem 0 !important;
    }

    /* Text color in chat */
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span {
        color: #ececec !important;
    }

    /* ─── CHAT INPUT ─── */
    .stChatInput {
        background-color: #2f2f2f !important;
        border: 1px solid #424242 !important;
        border-radius: 1.5rem !important;
    }

    .stChatInput textarea {
        color: #ececec !important;
    }

    .stChatInput textarea::placeholder {
        color: #8e8e8e !important;
    }

    /* ─── EXPANDER (Sources) ─── */
    .streamlit-expanderHeader {
        background-color: #2f2f2f !important;
        border-radius: 0.75rem !important;
        color: #b4b4b4 !important;
        font-size: 0.85rem !important;
    }

    .streamlit-expanderContent {
        background-color: #1e1e1e !important;
        border: 1px solid #2f2f2f !important;
        border-radius: 0 0 0.75rem 0.75rem !important;
    }

    .streamlit-expanderContent p {
        color: #b4b4b4 !important;
        font-size: 0.85rem !important;
    }

    /* ─── FILE UPLOADER ─── */
    .stFileUploader {
        background-color: #2f2f2f !important;
        border: 2px dashed #424242 !important;
        border-radius: 1rem !important;
    }

    .stFileUploader label {
        color: #ececec !important;
    }

    /* ─── BUTTONS ─── */
    .stButton > button {
        background-color: #2f2f2f !important;
        color: #ececec !important;
        border: 1px solid #424242 !important;
        border-radius: 0.75rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #424242 !important;
        border-color: #5a5a5a !important;
    }

    /* ─── STATUS / ALERTS ─── */
    .stAlert {
        background-color: #2f2f2f !important;
        border-radius: 0.75rem !important;
        color: #ececec !important;
    }

    /* ─── METRICS in sidebar ─── */
    [data-testid="stMetric"] {
        background-color: #2f2f2f;
        border-radius: 0.75rem;
        padding: 0.75rem;
    }

    [data-testid="stMetricValue"] {
        color: #10a37f !important;
    }

    [data-testid="stMetricLabel"] {
        color: #b4b4b4 !important;
    }

    /* ─── SPINNER ─── */
    .stSpinner > div {
        border-top-color: #10a37f !important;
    }

    /* ─── DIVIDER ─── */
    hr {
        border-color: #2f2f2f !important;
    }

    /* ─── SCROLLBAR ─── */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #171717;
    }
    ::-webkit-scrollbar-thumb {
        background: #424242;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #5a5a5a;
    }

    /* ─── WELCOME MESSAGE ─── */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
    }

    .welcome-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #ececec;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        font-size: 1rem;
        color: #8e8e8e;
        margin-bottom: 2rem;
    }

    /* ─── STATUS BADGE ─── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-ready {
        background-color: #062e1e;
        color: #10a37f;
        border: 1px solid #10a37f40;
    }

    .status-waiting {
        background-color: #2e2400;
        color: #f0b429;
        border: 1px solid #f0b42940;
    }

    /* ─── HIDE STREAMLIT BRANDING ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════


def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "processing" not in st.session_state:
        st.session_state.processing = False


# ═══════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save the uploaded file to the data directory.

    Args:
        uploaded_file: The file object from Streamlit uploader.

    Returns:
        Path to the saved file.
    """
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()

    # Save all files as data.pdf if it is a pdf, or keep extension otherwise
    # This simplifies ingestion logic downstream
    if suffix == ".pdf":
        target_path = settings.DATA_DIR / "data.pdf"
    else:
        safe_name = uploaded_file.name.replace(" ", "_")
        target_path = settings.DATA_DIR / safe_name

    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"File saved to {target_path}")
    return target_path


def initialize_pipeline(filename: str, strategy: str, chunk_params: dict) -> bool:
    """
    Initialize or re-initialize the RAG pipeline with specific configuration.

    Args:
        filename: Name of the file to ingest.
        strategy: Chunking strategy ("semantic" or "recursive").
        chunk_params: Dictionary of parameters for the chosen strategy.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Re-create pipeline to apply new settings
        # If a pipeline existed, we reset state first
        if st.session_state.document_loaded and st.session_state.pipeline:
            st.session_state.pipeline.reset()
            st.session_state.pipeline.cache.invalidate()

        # Initialize new pipeline with config
        st.session_state.pipeline = RAGPipeline(
            ingestion_config={"strategy": strategy, **chunk_params}
        )

        # Run ingestion
        st.session_state.pipeline.run_ingestion(filename)

        # Update state
        st.session_state.document_loaded = True
        st.session_state.document_name = filename

        return True

    except RAGException as e:
        logger.error(f"Pipeline initialization failed: {e}")
        st.error(f"❌ Error processing document: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"❌ Unexpected error: {e}")
        return False


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════


def render_sidebar() -> None:
    """Render the sidebar with configuration and upload options."""
    with st.sidebar:
        st.markdown("## 🧠 RAG Assistant")
        st.caption("Powered by Llama 3.3 + ChromaDB")
        st.divider()

        # ─── Configuración de Ingestion ───
        with st.expander("⚙️ Ingestion Settings", expanded=False):
            strategy = st.radio(
                "Chunking Strategy",
                ["semantic", "recursive"],
                index=0,
                help="Semantic: cuts by meaning. Recursive: cuts by size.",
            )

            chunk_params = {}

            if strategy == "semantic":
                chunk_params["breakpoint_threshold"] = st.slider(
                    "Breakpoint Threshold",
                    min_value=50,
                    max_value=100,
                    value=85,
                    step=5,
                    help="Percentile of semantic difference to trigger a split.",
                )
                chunk_params["buffer_size"] = st.number_input(
                    "Buffer Size",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="Number of sentences to group before splitting.",
                )
            else:
                chunk_params["chunk_size"] = st.number_input(
                    "Chunk Size",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                )
                chunk_params["chunk_overlap"] = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=200,
                    step=50,
                )

        st.divider()

        # ─── Document Upload ───
        st.markdown("### 📄 Document")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md", "docx"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)

            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📎 {uploaded_file.name}")
            with col2:
                st.caption(f"📦 {file_size_mb:.1f} MB")

            if st.button(
                "🚀 Process Document", use_container_width=True, type="primary"
            ):
                with st.spinner("Processing document..."):
                    saved_path = save_uploaded_file(uploaded_file)
                    filename = saved_path.name

                    success = initialize_pipeline(filename, strategy, chunk_params)

                    if success:
                        st.success(f"✅ **{uploaded_file.name}** loaded!")
                        st.session_state.messages = []
                        st.rerun()

        st.divider()

        # ─── Status ───
        st.markdown("### 📊 Status")

        if st.session_state.document_loaded:
            st.markdown(
                '<span class="status-badge status-ready">● Ready</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Document: **{st.session_state.document_name}**")

            # Show Cache Stats if available
            if st.session_state.pipeline:
                stats = st.session_state.pipeline.cache.get_stats()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Queries", stats["total_queries"])
                with col2:
                    st.metric("Cache Hits", stats["hits"])
                with col3:
                    st.metric("Hit Rate", stats["hit_rate"])
        else:
            st.markdown(
                '<span class="status-badge status-waiting">'
                "○ Waiting for document</span>",
                unsafe_allow_html=True,
            )
            st.caption("Upload a document to start chatting.")

        st.divider()

        # ─── Actions ───
        st.markdown("### ⚙️ Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.pipeline:
                    st.session_state.pipeline.state.clear()
                st.rerun()

        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                st.session_state.messages = []
                st.session_state.document_loaded = False
                st.session_state.document_name = None
                if st.session_state.pipeline:
                    st.session_state.pipeline.reset()
                    st.session_state.pipeline.cache.invalidate()
                    st.session_state.pipeline = None
                st.rerun()


# ═══════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════

SUGGESTIONS = [
    {
        "title": "📝 Summarize",
        "desc": "the main topics of the document",
        "query": "Summarize the main topics of the document",
    },
    {
        "title": "🔍 Find",
        "desc": "specific information",
        "query": "What is the main objective of the document?",
    },
    {
        "title": "📊 Analyze",
        "desc": "key concepts mentioned",
        "query": "What are the key concepts mentioned?",
    },
    {
        "title": "❓ Explain",
        "desc": "technical terms used",
        "query": "Explain the most important technical terms",
    },
]


def render_welcome() -> None:
    """Render the welcome screen with suggestions."""
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">🧠 RAG Assistant</div>
            <div class="welcome-subtitle">
                Ask anything about your document
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.document_loaded:
        cols = st.columns(2)
        for idx, suggestion in enumerate(SUGGESTIONS):
            with cols[idx % 2]:
                if st.button(
                    f"{suggestion['title']}\n\n{suggestion['desc']}",
                    key=f"suggestion_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": suggestion["query"],
                        }
                    )
                    st.rerun()
    else:
        st.info(
            "👈 Upload a document in the sidebar to start chatting.",
            icon="📄",
        )


# ═══════════════════════════════════════════════════════
# CHAT RENDERING
# ═══════════════════════════════════════════════════════


def render_message(message: dict) -> None:
    """Render a single chat message."""
    with st.chat_message(
        message["role"],
        avatar="🧑" if message["role"] == "user" else "🧠",
    ):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander(
                f"📚 Sources ({len(message['sources'])} references)",
                expanded=False,
            ):
                for i, source in enumerate(message["sources"]):
                    page = source.get("page", "?")
                    preview = source.get("preview", "")
                    st.markdown(f"**Ref {i + 1}** · Page {page}\n\n```\n{preview}\n```")


def render_chat_history() -> None:
    """Render the entire chat history."""
    for message in st.session_state.messages:
        render_message(message)


# ═══════════════════════════════════════════════════════
# CHAT LOGIC
# ═══════════════════════════════════════════════════════


def process_user_query(query: str) -> None:
    """Process user query through the RAG pipeline."""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking..."):
            try:
                response, docs = st.session_state.pipeline.run_conversation_flow(query)

                st.markdown(response)

                sources = []
                if docs:
                    for doc in docs:
                        sources.append(
                            {
                                "page": doc.metadata.get("page", "?"),
                                "preview": doc.page_content[:200].replace("\n", " "),
                            }
                        )

                    with st.expander(
                        f"📚 Sources ({len(sources)} references)",
                        expanded=False,
                    ):
                        for i, source in enumerate(sources):
                            st.markdown(
                                f"**Ref {i + 1}** · Page {source['page']}\n\n"
                                f"```\n{source['preview']}\n```"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    }
                )

            except RAGException as e:
                error_msg = f"❌ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
            except Exception as e:
                error_msg = f"❌ Unexpected error: {e}"
                st.error(error_msg)
                logger.error(f"Chat error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


# ═══════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════


def run() -> None:
    """Main entry point for the Streamlit app."""
    inject_custom_css()
    init_session_state()
    render_sidebar()

    if not st.session_state.messages:
        render_welcome()
    else:
        render_chat_history()

    if prompt := st.chat_input(
        placeholder=(
            "Ask about your document..."
            if st.session_state.document_loaded
            else "Upload a document first..."
        ),
        disabled=not st.session_state.document_loaded,
    ):
        process_user_query(prompt)
