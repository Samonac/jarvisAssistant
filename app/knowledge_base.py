"""Knowledge Base Manager for Jarvis Assistant.

Implements a per-user RAG (Retrieval-Augmented Generation) system:
- Documents are chunked and stored in SQLite
- Chunks are retrieved by keyword matching during conversations
- Jarvis can auto-create knowledge entries from conversations
- User isolation: each user's KB is private (admins see all)
"""

import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Chunk size in characters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class KnowledgeBaseManager:
    """Manages per-user knowledge bases with document chunking and retrieval.

    Attributes:
        db_manager: Database manager for persistent storage.
    """

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._initialize_tables()

    def _initialize_tables(self):
        """Create KB tables if they don't exist."""
        conn = self.db_manager._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                content TEXT NOT NULL,
                keywords TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES kb_documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_kb_chunks_username ON kb_chunks(username);
            CREATE INDEX IF NOT EXISTS idx_kb_chunks_keywords ON kb_chunks(keywords);
            CREATE INDEX IF NOT EXISTS idx_kb_documents_username ON kb_documents(username);
        """)
        conn.commit()

    # ── Document Management ────────────────────────────────────────────────

    def add_document(self, username: str, title: str, content: str, source: str = "manual") -> dict:
        """Add a document to the knowledge base, auto-chunking it.

        Args:
            username: Owner of the document.
            title: Document title.
            content: Full document text.
            source: Origin ('manual', 'upload', 'conversation', 'auto').

        Returns:
            Dict with document_id and chunk_count.
        """
        conn = self.db_manager._get_connection()

        # Insert document
        cursor = conn.execute(
            "INSERT INTO kb_documents (username, title, source, content) VALUES (?, ?, ?, ?)",
            (username, title, source, content),
        )
        doc_id = cursor.lastrowid

        # Chunk the content
        chunks = self._chunk_text(content)

        # Insert chunks with keywords
        for i, chunk in enumerate(chunks):
            keywords = self._extract_keywords(chunk)
            conn.execute(
                "INSERT INTO kb_chunks (document_id, username, chunk_index, content, keywords) VALUES (?, ?, ?, ?, ?)",
                (doc_id, username, i, chunk, ",".join(keywords)),
            )

        conn.commit()
        logger.info("KB document added: '%s' (%d chunks) for user '%s'", title, len(chunks), username)
        return {"document_id": doc_id, "chunk_count": len(chunks), "title": title}

    def get_documents(self, username: str, include_all: bool = False) -> list[dict]:
        """List documents. Regular users see their own; admins can see all."""
        conn = self.db_manager._get_connection()
        if include_all:
            cursor = conn.execute(
                "SELECT id, username, title, source, created_at, updated_at FROM kb_documents ORDER BY updated_at DESC"
            )
        else:
            cursor = conn.execute(
                "SELECT id, username, title, source, created_at, updated_at FROM kb_documents WHERE username = ? ORDER BY updated_at DESC",
                (username,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Get a document with its full content."""
        conn = self.db_manager._get_connection()
        row = conn.execute("SELECT * FROM kb_documents WHERE id = ?", (doc_id,)).fetchone()
        if row:
            return dict(row)
        return None

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all its chunks."""
        conn = self.db_manager._get_connection()
        conn.execute("DELETE FROM kb_chunks WHERE document_id = ?", (doc_id,))
        cursor = conn.execute("DELETE FROM kb_documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0

    def update_document(self, doc_id: int, title: str = None, content: str = None) -> bool:
        """Update a document (re-chunks if content changes)."""
        conn = self.db_manager._get_connection()
        doc = self.get_document(doc_id)
        if not doc:
            return False

        if title:
            conn.execute("UPDATE kb_documents SET title = ?, updated_at = ? WHERE id = ?",
                         (title, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), doc_id))

        if content:
            conn.execute("UPDATE kb_documents SET content = ?, updated_at = ? WHERE id = ?",
                         (content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), doc_id))
            # Re-chunk
            conn.execute("DELETE FROM kb_chunks WHERE document_id = ?", (doc_id,))
            chunks = self._chunk_text(content)
            for i, chunk in enumerate(chunks):
                keywords = self._extract_keywords(chunk)
                conn.execute(
                    "INSERT INTO kb_chunks (document_id, username, chunk_index, content, keywords) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, doc["username"], i, chunk, ",".join(keywords)),
                )

        conn.commit()
        return True

    # ── Chunk Management ───────────────────────────────────────────────────

    def get_chunks(self, doc_id: int) -> list[dict]:
        """Get all chunks for a document."""
        conn = self.db_manager._get_connection()
        cursor = conn.execute(
            "SELECT id, chunk_index, content, keywords FROM kb_chunks WHERE document_id = ? ORDER BY chunk_index",
            (doc_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_chunk(self, chunk_id: int, content: str) -> bool:
        """Update a specific chunk's content."""
        conn = self.db_manager._get_connection()
        keywords = self._extract_keywords(content)
        cursor = conn.execute(
            "UPDATE kb_chunks SET content = ?, keywords = ? WHERE id = ?",
            (content, ",".join(keywords), chunk_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_chunk(self, chunk_id: int) -> bool:
        """Delete a specific chunk."""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("DELETE FROM kb_chunks WHERE id = ?", (chunk_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ── Retrieval ──────────────────────────────────────────────────────────

    def search(self, username: str, query: str, max_results: int = 5) -> list[dict]:
        """Search the knowledge base for relevant chunks.

        Uses keyword matching against chunk keywords and content.
        Only returns chunks belonging to the specified user.

        Args:
            username: The user whose KB to search.
            query: The search query.
            max_results: Maximum chunks to return.

        Returns:
            List of relevant chunk dicts with content and source info.
        """
        conn = self.db_manager._get_connection()
        query_keywords = self._extract_keywords(query)

        if not query_keywords:
            return []

        # Search by keyword match and content LIKE
        results = []
        seen_ids = set()

        for keyword in query_keywords[:5]:  # Limit to 5 keywords
            cursor = conn.execute(
                """
                SELECT c.id, c.content, c.keywords, d.title, d.source
                FROM kb_chunks c
                JOIN kb_documents d ON c.document_id = d.id
                WHERE c.username = ? AND (c.keywords LIKE ? OR c.content LIKE ?)
                LIMIT ?
                """,
                (username, f"%{keyword}%", f"%{keyword}%", max_results),
            )
            for row in cursor.fetchall():
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    results.append({
                        "chunk_id": row["id"],
                        "content": row["content"],
                        "document_title": row["title"],
                        "source": row["source"],
                    })

        return results[:max_results]

    def get_context_for_message(self, username: str, message: str) -> str:
        """Get relevant KB context for a user message.

        Returns a formatted string to inject into the LLM context,
        or empty string if nothing relevant is found.
        """
        results = self.search(username, message, max_results=3)
        if not results:
            return ""

        parts = ["[KNOWLEDGE BASE — relevant information for this user]"]
        for r in results:
            parts.append(f"From '{r['document_title']}': {r['content']}")
        parts.append("[END KNOWLEDGE BASE]")
        return "\n".join(parts)

    # ── Auto-Learning ──────────────────────────────────────────────────────

    def learn_from_conversation(self, username: str, topic: str, content: str) -> dict:
        """Create a knowledge entry from a conversation insight.

        Called when Jarvis determines something worth remembering.
        """
        title = f"Learned: {topic[:50]}"
        return self.add_document(username, title, content, source="conversation")

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into semantic chunks that never break mid-sentence.

        Strategy:
        1. Split by paragraphs (double newlines) first
        2. If a paragraph is too long, split by sentences
        3. Merge small consecutive paragraphs into one chunk
        4. Target chunk size: 400-800 chars
        """
        if not text:
            return []

        MIN_CHUNK = 200
        MAX_CHUNK = 800

        # Step 1: Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Step 2: Split long paragraphs into sentences
        segments = []
        for para in paragraphs:
            if len(para) <= MAX_CHUNK:
                segments.append(para)
            else:
                # Split by sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= MAX_CHUNK:
                        current = (current + " " + sent).strip() if current else sent
                    else:
                        if current:
                            segments.append(current)
                        current = sent
                if current:
                    segments.append(current)

        # Step 3: Merge small consecutive segments
        chunks = []
        current_chunk = ""
        for seg in segments:
            if len(current_chunk) + len(seg) + 2 <= MAX_CHUNK:
                current_chunk = (current_chunk + "\n\n" + seg).strip() if current_chunk else seg
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = seg

        if current_chunk:
            chunks.append(current_chunk)

        # Step 4: Handle any remaining very small chunks by merging with previous
        final_chunks = []
        for chunk in chunks:
            if final_chunks and len(chunk) < MIN_CHUNK and len(final_chunks[-1]) + len(chunk) + 2 <= MAX_CHUNK:
                final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract significant keywords from text for indexing."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "because", "but", "and", "or", "if", "while", "that", "this",
            "these", "those", "it", "its", "i", "me", "my", "we", "our", "you",
            "your", "he", "him", "his", "she", "her", "they", "them", "their",
            "what", "which", "who", "whom", "sir", "please", "thank", "thanks",
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stop_words]

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        return unique[:20]  # Max 20 keywords per chunk
