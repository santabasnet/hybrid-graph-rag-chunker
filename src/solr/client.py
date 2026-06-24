from __future__ import annotations

from typing import Any

import httpx

from hybrid_graph_rag_chunker.constants import FIELD_EMBEDDING

DEFAULT_SOLR_BASE = "http://localhost:8983"
DEFAULT_COLLECTION = "hybrid_rag"
DEFAULT_BM25_ROWS = 20
DEFAULT_VECTOR_ROWS = 20
DEFAULT_TIMEOUT = 30.0


def _clean_val(v: Any) -> Any:
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v


def _clean_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: _clean_val(v) for k, v in doc.items()}


def _first_doc(response: dict[str, Any]) -> dict[str, Any] | None:
    docs = _docs(response)
    return docs[0] if docs else None


def _docs(response: dict[str, Any]) -> list[dict[str, Any]]:
    raw = response.get("response", {}).get("docs", [])
    return [_clean_doc(d) for d in raw]


def _ids_query(ids: list[str]) -> str:
    quoted = " ".join(f'"{i}"' for i in ids)
    return f"id:({quoted})"


class SolrClient:
    def __init__(
        self,
        base_url: str = DEFAULT_SOLR_BASE,
        collection: str = DEFAULT_COLLECTION,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.select_url = f"{base_url}/solr/{collection}/select"
        self.update_url = f"{base_url}/solr/{collection}/update"
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _post_select(self, params: dict[str, str]) -> list[dict[str, Any]]:
        resp = self._client.post(self.select_url, data=params)
        resp.raise_for_status()
        return _docs(resp.json())

    def search_bm25(
        self,
        query: str,
        rows: int = DEFAULT_BM25_ROWS,
        fl: str = "*,score",
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "q": query,
            "qf": "content^4 title^2",
            "defType": "edismax",
            "rows": str(rows),
            "fl": fl,
            "wt": "json",
        }
        return self._post_select(params)

    def search_vector(
        self,
        vector: list[float],
        rows: int = DEFAULT_VECTOR_ROWS,
        fl: str = "*,score",
    ) -> list[dict[str, Any]]:
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        params: dict[str, str] = {
            "q": f"{{!knn f={FIELD_EMBEDDING} topK={rows}}}{vector_str}",
            "rows": str(rows),
            "fl": fl,
            "wt": "json",
        }
        return self._post_select(params)

    def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        params: dict[str, str] = {
            "q": f'id:"{doc_id}"',
            "rows": "1",
            "fl": "*,score",
            "wt": "json",
        }
        resp = self._client.post(self.select_url, data=params)
        resp.raise_for_status()
        return _first_doc(resp.json())

    def get_by_ids(
        self, doc_ids: list[str], fl: str = "*,score"
    ) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        params: dict[str, str] = {
            "q": _ids_query(doc_ids),
            "rows": str(len(doc_ids)),
            "fl": fl,
            "wt": "json",
        }
        resp = self._client.post(self.select_url, data=params)
        resp.raise_for_status()
        return _docs(resp.json())

    def index_documents(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return
        payload: dict[str, Any] = {"add": docs}
        params = {"commitWithin": "5000", "wt": "json"}
        resp = self._client.post(self.update_url, json=payload, params=params)
        resp.raise_for_status()

    def delete_all(self) -> None:
        payload: dict[str, Any] = {"delete": {"query": "*:*"}}
        params = {"commit": "true", "wt": "json"}
        resp = self._client.post(self.update_url, json=payload, params=params)
        resp.raise_for_status()

    def commit(self) -> None:
        params = {"commit": "true", "wt": "json"}
        resp = self._client.post(self.update_url, json={}, params=params)
        resp.raise_for_status()
