"""
Semantic Cache with Intent Normalization.

Two levels of optimization:
1. Normalization: LLM reduces each query to its canonical intent.
2. Semantic similarity: Compares canonical intents via embeddings.
"""

import json
import time
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from rag.chain.prompts import get_template
from rag.providers import LLMFactory


class SemanticCache:
    """
    Response cache based on intent similarity.

    When a new query arrives:
    1. LLM reduces the query to its canonical intent (2-5 words).
    2. Generates embedding of the intent.
    3. Compares with ALL cached intents.
    4. If similarity >= threshold → Returns cached response.
    5. If not → Executes normal pipeline and saves result.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        threshold: float = 0.85,
        max_cache_size: int = 500,
        persist_path: str | None = None,
    ):
        self.embeddings = embeddings
        self.threshold = threshold
        self.max_cache_size = max_cache_size

        # Convert to Path object if provided
        self.persist_path: Path | None = Path(persist_path) if persist_path else None

        self.entries: list[dict] = []
        self._vectors: np.ndarray | None = None
        self.stats = {"hits": 0, "misses": 0, "total": 0}

        # Load cache if path exists
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

        self._intent_llm = LLMFactory.create(temperature=0)
        self._intent_template = get_template("cache")

        self._intent_chain = (
            self._intent_template | self._intent_llm | StrOutputParser()
        )

        logger.info(
            f"Semantic Cache initialized "
            f"(threshold={threshold}, max_size={max_cache_size})"
        )

    # ═══════════════════════════════════════════════════════
    # INTENT NORMALIZATION
    # ═══════════════════════════════════════════════════════

    def _get_canonical_intent(self, query: str) -> str:
        """
        Normalizes the query to its canonical intent.

        Example: "where does LIME appear" -> "LIME location"
        """
        try:
            canonical = self._intent_chain.invoke({"query": query}).strip().lower()
            logger.debug(f"Intent: '{query}' → '{canonical}'")
            return canonical

        except Exception as e:
            logger.warning(f"Intent normalization failed: {e}")
            return query

    # ═══════════════════════════════════════════════════════
    # CACHE OPERATIONS
    # ═══════════════════════════════════════════════════════

    def get(self, query: str) -> tuple[str, str, list[Document]] | None:
        """
        Searches for a cached response by intent similarity.

        Returns:
            (canonical_intent, response, documents) if hit, None if miss.
        """
        self.stats["total"] += 1

        if not self.entries:
            self.stats["misses"] += 1
            return None

        try:
            # 1. Normalize intention
            canonical = self._get_canonical_intent(query)

            # 2. Vectorize the intention
            query_vector = np.array(self.embeddings.embed_query(canonical))

            # 3. Calculate similarities
            similarities = self._compute_similarities(query_vector)

            # 4. Find the best match
            best_idx = np.argmax(similarities)
            best_score = float(similarities[best_idx])

            # 5. Surpasses the threshold?
            if best_score >= self.threshold:
                self.stats["hits"] += 1
                entry = self.entries[best_idx]

                entry["last_accessed"] = time.time()
                entry["access_count"] = entry.get("access_count", 0) + 1

                logger.info(
                    f"🎯 Cache HIT (sim={best_score:.4f}) | "
                    f"Intent: '{canonical}' ≈ '{entry['canonical']}'"
                )

                print(
                    f"⚡ Cache HIT (sim={best_score:.3f}) | "
                    f"Saved ~2 sec and 3 LLM calls"
                )

                docs = self._deserialize_docs(entry["docs_data"])
                return entry["canonical"], entry["response"], docs

            # 6. Cache MISS
            self.stats["misses"] += 1
            logger.debug(
                f"Cache MISS | Intent: '{canonical}' | "
                f"Best match: '{self.entries[best_idx]['canonical']}' "
                f"(sim={best_score:.4f})"
            )
            return None

        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            self.stats["misses"] += 1
            return None

    def put(self, query: str, response: str, docs: list[Document]) -> None:
        """Stores a new entry in the cache."""
        try:
            # Re-calculate canonical intent to ensure consistency
            canonical = self._get_canonical_intent(query)
            query_vector = np.array(self.embeddings.embed_query(canonical))
            docs_data = self._serialize_docs(docs)

            entry = {
                "query": query,
                "canonical": canonical,
                "response": response,
                "docs_data": docs_data,
                "vector": query_vector,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 1,
            }

            if len(self.entries) >= self.max_cache_size:
                self._evict_least_used()

            self.entries.append(entry)
            self._rebuild_vector_matrix()

            if self.persist_path:
                self._save_to_disk()

            logger.debug(f"Cache stored: '{query}' as intent '{canonical}'")

        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    # ═══════════════════════════════════════════════════════
    # SIMILARITY COMPUTATION
    # ═══════════════════════════════════════════════════════

    def _compute_similarities(self, query_vector: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity against all cached vectors."""
        if self._vectors is None:
            self._rebuild_vector_matrix()

        # Add small epsilon to avoid division by zero
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)

        # Mypy/Numpy compatibility check
        assert self._vectors is not None

        similarities = self._vectors @ query_norm

        return similarities

    def _rebuild_vector_matrix(self) -> None:
        if not self.entries:
            self._vectors = None
            return

        vectors = np.array([e["vector"] for e in self.entries])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        self._vectors = vectors / norms

    # ═══════════════════════════════════════════════════════
    # EVICTION STRATEGY
    # ═══════════════════════════════════════════════════════

    def _evict_least_used(self) -> None:
        """Evicts the least used entry (LFU with temporal decay)."""
        now = time.time()
        scores = []
        for entry in self.entries:
            age_hours = (now - entry["created_at"]) / 3600 + 1
            score = entry.get("access_count", 1) / age_hours
            scores.append(score)

        worst_idx = np.argmin(scores)
        removed = self.entries.pop(worst_idx)

        logger.debug(
            f"Cache eviction: removed '{removed['query'][:40]}' "
            f"(score={scores[worst_idx]:.4f})"
        )

    # ═══════════════════════════════════════════════════════
    # SERIALIZATION / PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _serialize_docs(self, docs: list[Document]) -> list[dict]:
        """Converts Documents to dicts for persistence."""
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    def _deserialize_docs(self, docs_data: list[dict]) -> list[Document]:
        """Reconstructs Documents from dicts."""
        return [
            Document(
                page_content=d["page_content"],
                metadata=d["metadata"],
            )
            for d in docs_data
        ]

    def _save_to_disk(self) -> None:
        """Persists the cache to disk (JSON)."""
        if not self.persist_path:
            return

        try:
            data = []
            for entry in self.entries:
                # Convert numpy array to list for JSON serialization
                vector_list = (
                    entry["vector"].tolist()
                    if isinstance(entry["vector"], np.ndarray)
                    else entry["vector"]
                )

                data.append(
                    {
                        "query": entry["query"],
                        "canonical": entry.get("canonical", entry["query"]),
                        "response": entry["response"],
                        "docs_data": entry["docs_data"],
                        "vector": vector_list,
                        "created_at": entry["created_at"],
                        "last_accessed": entry["last_accessed"],
                        "access_count": entry["access_count"],
                    }
                )

            # Ensure parent directory exists
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Cache saved to {self.persist_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_disk(self) -> None:
        """Loads cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                item["vector"] = np.array(item["vector"])
                if "canonical" not in item:
                    item["canonical"] = item["query"]
                self.entries.append(item)

            self._rebuild_vector_matrix()

            logger.info(f"Cache loaded from disk: {len(self.entries)} entries")

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted cache file, starting fresh: {e}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    # ═══════════════════════════════════════════════════════
    # CACHE MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def invalidate(self) -> None:
        """Clears all cache entries."""
        old_size = len(self.entries)
        self.entries = []
        self._vectors = None
        self.stats = {"hits": 0, "misses": 0, "total": 0}

        if self.persist_path and self.persist_path.exists():
            try:
                self.persist_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to delete cache file: {e}")

        logger.info(f"Cache invalidated: {old_size} entries cleared")

    def get_stats(self) -> dict:
        """Performance statistics of the cache."""
        total = self.stats["total"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0

        return {
            "total_queries": total,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "cache_size": len(self.entries),
            "max_size": self.max_cache_size,
            "threshold": self.threshold,
            "most_asked": self._get_most_asked(top=5),
        }

    def _get_most_asked(self, top: int = 5) -> list[dict]:
        """Returns the most asked questions."""
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.get("access_count", 0),
            reverse=True,
        )
        return [
            {
                "query": e["query"][:60],
                "canonical": e.get("canonical", "N/A"),
                "times_asked": e.get("access_count", 0),
            }
            for e in sorted_entries[:top]
        ]
