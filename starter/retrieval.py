
"""LAB7: Dense, BM25 and Hybrid retrieval using LAB6 data."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import psycopg
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    chunk_id: str
    page: int
    content: str
    source: str
    section: str | None


class Retriever:
    def __init__(self):
        self.dsn = os.environ["DATABASE_URL"]
        self.conn = psycopg.connect(self.dsn)

        # Load the same Strategy B corpus for every method.
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT
                    chunk_id,
                    page,
                    content,
                    source,
                    metadata->>'section' AS section
                FROM public.lab_document_chunks
                WHERE strategy = 'B'
                  AND embedding IS NOT NULL
                ORDER BY page, chunk_id
            """)

            self.chunks = [
                Chunk(*row) for row in cur.fetchall()
            ]

        if not self.chunks:
            raise RuntimeError("No Strategy B chunks found")

        self.by_id = {
            chunk.chunk_id: chunk
            for chunk in self.chunks
        }

        # BM25 index built from the same 162 chunks.
        self.bm25 = BM25Okapi([
            tokenize(chunk.content)
            for chunk in self.chunks
        ])

        # Same embedding model as LAB6.
        self.model = SentenceTransformer("BAAI/bge-m3")

    def eligible(self, section=None):
        if section is None:
            return self.chunks

        return [
            c for c in self.chunks
            if c.section == str(section)
        ]

    @lru_cache(maxsize=256)
    def query_vector(self, question):
        vector = self.model.encode(
            question,
            normalize_embeddings=True
        )

        if len(vector) != 1024:
            raise RuntimeError(
                f"Expected 1024 dimensions, got {len(vector)}"
            )

        return "[" + ",".join(
            str(float(x)) for x in vector
        ) + "]"

    # METHOD 1: DENSE RETRIEVAL
    
    def dense(self, question, top_k=3, section=None):
        vector = self.query_vector(question)

        where = "strategy = 'B' AND embedding IS NOT NULL"
        params = [vector]

        if section is not None:
            where += " AND metadata->>'section' = %s"

        query = f"""
            SELECT
                chunk_id,
                1 - (embedding <=> %s::vector) AS score
            FROM public.lab_document_chunks
            WHERE {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        if section is not None:
            params.append(str(section))

        params.extend([vector, top_k])

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return [
                (self.by_id[cid], float(score))
                for cid, score in cur.fetchall()
            ]


    # METHOD 2: BM25 LEXICAL RETRIEVAL
    def keyword(self, question, top_k=3, section=None):
        tokens = tokenize(question)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            (
                (chunk, float(scores[i]))
                for i, chunk in enumerate(self.chunks)
                if section is None
                or chunk.section == str(section)
            ),
            key=lambda item: (
                -item[1],
                item[0].chunk_id
            )
        )

        return ranked[:top_k]

    # METHOD 3: HYBRID RETRIEVAL (RRF)
    def hybrid(
        self,
        question,
        top_k=3,
        section=None,
        rrf_k=60
    ):
        n = len(self.eligible(section))

        if n == 0:
            return []

        dense_results = self.dense(
            question, top_k=n, section=section
        )

        bm25_results = self.keyword(
            question, top_k=n, section=section
        )

        scores = {}

        for ranked in (dense_results, bm25_results):
            for rank, (chunk, _) in enumerate(ranked, 1):
                cid = chunk.chunk_id

                scores[cid] = (
                    scores.get(cid, 0.0)
                    + 1 / (rrf_k + rank)
                )

        items = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )[:top_k]

        return [
            (self.by_id[cid], score)
            for cid, score in items
        ]

    def search(
        self,
        method,
        question,
        top_k=3,
        section=None
    ):
        methods = {
            "dense": self.dense,
            "bm25": self.keyword,
            "hybrid": self.hybrid
        }

        return methods[method](
            question,
            top_k=top_k,
            section=section
        )

    def close(self):
        self.conn.close()
