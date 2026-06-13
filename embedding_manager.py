# -*- coding: utf-8 -*-
"""
embedding_manager.py
====================
RAG (Retrieval-Augmented Generation) hattının "Retrieval" katmanı.

Sorumluluklar (SRP):
    * Lokal embedding modelini yükleme (sentence-transformers)
      - Önce lokal klasörden okur (internetsiz), yoksa model adıyla indirir.
    * Metin parçalarını vektörleştirip kalıcı (persistent) ChromaDB'ye yazma.
    * Bir soru için benzerlik araması (similarity search) yaparak ilgili
      bağlam parçalarını döndürme.

Tasarım notu (ÖNEMLİ):
    ChromaDB 1.5.x, koleksiyona bir "embedding function" verildiğinde onun
    konfigürasyonunu kalıcı olarak saklamaya ve sonraki açılışlarda tutarlılık
    doğrulamasına çalışır. Özel fonksiyonlarda bu, "embedding function conflict"
    hatalarına yol açar. Bu yüzden embedding'leri DIŞARIDA (bu modülde) kendimiz
    hesaplayıp ChromaDB'ye doğrudan vektör olarak veriyoruz. Koleksiyon hiçbir
    embedding fonksiyonu bilmez; bu da sürümden bağımsız ve sağlam bir çözümdür.

%100 lokal çalışır; hiçbir bulut servisine bağlanmaz.
"""

from __future__ import annotations

import os
import re
import time
import math
import uuid
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# Tokenleştirme: Unicode kelime karakterleri (Latin/Türkçe/Korece/CJK dahil).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """BM25 için basit, script-bağımsız tokenleştirme (küçük harf)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


_NUM_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _norm_for_dedup(text: str) -> str:
    """
    Boilerplate tespiti için normalize: küçük harf, rakamlar (sayfa no vb.)
    silinir, noktalama boşluğa çevrilir, boşluklar teke iner. Böylece
    "All type Page 2/3 Normal Operation..." gibi yalnızca sayfa numarasıyla
    farklılaşan tekrarlayan başlıklar AYNI forma indirgenir. Tüm script'lerin
    harfleri (Latin/Türkçe/Korece) korunur (\\w, re.UNICODE).
    """
    t = _NUM_RE.sub("", text.lower())
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


class _BM25:
    """
    Minimal, harici bağımlılıksız BM25 (Okapi) — lexical (anahtar kelime) arama.
    Dense (anlamsal) retrieval'ın gürültülü/OCR'lı teknik metinde kaçırdığı tam
    terim eşleşmelerini yakalar. Ters indeks (postings) ile hızlı skorlar.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        df: Dict[str, int] = {}
        for i, toks in enumerate(corpus_tokens):
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, freq in tf.items():
                self.postings.setdefault(t, []).append((i, freq))
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def scores(self, query_tokens: List[str]) -> List[float]:
        sc = [0.0] * self.N
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, freq in self.postings[t]:
                dl = self.doc_len[i] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                sc[i] += idf * freq * (self.k1 + 1) / denom
        return sc


def resolve_device(device: str = "auto") -> str:
    """
    Kullanılacak cihazı belirler. 'auto' ise CUDA (GPU) varsa 'cuda', yoksa
    'cpu' döner. Böylece GPU'lu makinede (CUDA'lı torch kuruluysa) otomatik
    GPU, CPU makinede CPU kullanılır. Belirli bir değer verilirse aynen kullanılır.
    """
    if device and device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


@dataclass
class RetrievedContext:
    """Benzerlik aramasından dönen tek bir bağlam parçası."""
    text: str
    source: str
    page: int
    score: float  # Mesafe (düşük = daha benzer)


class LocalEmbedder:
    """
    sentence-transformers tabanlı lokal embedding modeli sarmalayıcısı.

    Modeli 'lazy' yükler (ilk ihtiyaçta), böylece uygulama açılışı hızlı kalır.
    e5 ailesi modelleri için önerilen 'query:' / 'passage:' ön ekleri otomatik
    uygulanır (retrieval kalitesini belirgin artırır).
    """

    def __init__(self, model_name_or_path: str, device: str = "auto") -> None:
        self._model_name_or_path = model_name_or_path
        self._device = resolve_device(device)
        if self._device != "cpu":
            logger.info("Embedding cihazı: %s", self._device)
        self._model = None
        self._lock = threading.Lock()
        # e5 modelleri özel ön ek ister; model adından otomatik tespit.
        self._use_e5_prefix = "e5" in os.path.basename(model_name_or_path).lower()

    def _ensure_model(self):
        """Modeli ilk ihtiyaçta, thread-safe biçimde yükler."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:  # Çift kontrol (double-checked locking).
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers kurulu değil. "
                    "'pip install sentence-transformers' çalıştırın."
                ) from exc

            target = self._model_name_or_path
            if os.path.isdir(target):
                logger.info("Embedding modeli lokal klasörden yükleniyor: %s", target)
            else:
                logger.info(
                    "Embedding modeli yükleniyor/indiriliyor: %s "
                    "(internet yoksa lokal cache kullanılır)", target
                )
            try:
                self._model = SentenceTransformer(target, device=self._device)
                return self._model
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Embedding modeli yüklenemedi ('{target}'). İnternet yoksa "
                    f"modelin lokalde mevcut olduğundan emin olun. Hata: {exc}"
                ) from exc

    def _encode(self, texts: List[str]) -> List[List[float]]:
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Kosinüs benzerliği için normalize.
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Doküman parçalarını vektörleştirir (e5 için 'passage:' ön eki)."""
        if self._use_e5_prefix:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Tek bir sorguyu vektörleştirir (e5 için 'query:' ön eki)."""
        prepared = f"query: {text}" if self._use_e5_prefix else text
        return self._encode([prepared])[0]


class EmbeddingManager:
    """
    Vektör veritabanı yönetimi ve RAG retrieval işlemlerini kapsar.

    Veriler 'persist_directory' altında diske yazılır; uygulama kapanıp
    açılsa bile gemideki bilgisayarda kalıcıdır.
    """

    def __init__(
        self,
        model_name_or_path: str = "intfloat/multilingual-e5-base",
        persist_directory: str = "./vector_store",
        collection_name: str = "gemi_dokumanlari",
        device: str = "auto",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embedder = LocalEmbedder(model_name_or_path, device=device)

        os.makedirs(self.persist_directory, exist_ok=True)

        self._client = None
        self._collection = None
        # Birden fazla thread'in aynı anda koleksiyon oluşturmasını engeller.
        # REENTRANT (RLock): clear(), kilidi tutarken _ensure_collection()'ı
        # çağırabilir; düz Lock burada deadlock'a yol açar (clear ilk DB işlemiyse).
        self._init_lock = threading.RLock()

        # Son add_chunks çağrısının süre ölçümleri (ingestion raporu için).
        self.last_embedding_time = 0.0
        self.last_chroma_write_time = 0.0

        # Lexical (BM25) indeks — hibrit retrieval için. Lazy kurulur, koleksiyon
        # her değiştiğinde (add/clear) geçersizleşir.
        self._bm25: Optional["_BM25"] = None
        self._lex_ids: List[str] = []
        self._lex_docs: List[str] = []
        self._lex_metas: List[dict] = []
        self._lex_dirty = True

        # Boilerplate dedup: bu oturumda eklenen chunk'ların normalize edilmiş
        # formları. Aynı form ikinci kez gelirse (tekrarlayan başlık/footer)
        # atlanır; benzersiz içerik daima korunur. Bulk ingest'te global çalışır.
        self._seen_norm: set = set()
        self.last_deduped = 0

    # ------------------------------------------------------------------ #
    #  Veritabanı Bağlantısı (Lazy + Thread-safe)
    # ------------------------------------------------------------------ #
    def _ensure_collection(self):
        """Kalıcı ChromaDB istemcisini ve koleksiyonunu hazırlar."""
        if self._collection is not None:
            return self._collection

        # Tüm başlatma tek seferde, tek thread tarafından yapılsın.
        with self._init_lock:
            if self._collection is not None:  # Çift kontrol.
                return self._collection

            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                raise ImportError(
                    "chromadb kurulu değil. 'pip install chromadb' çalıştırın."
                ) from exc

            # ChromaDB 1.5.x'in Rust binding'leri, süreçteki ilk başlatmada bazen
            # tam hazır olmadan çağrılır ("Could not connect to tenant ..."). Bu
            # aralıklı bir yarış durumudur; kısa beklemelerle tekrar denenince geçer.
            last_exc: Optional[Exception] = None
            for attempt in range(1, 4):  # En fazla 3 deneme
                try:
                    if self._client is None:
                        self._client = chromadb.PersistentClient(
                            path=self.persist_directory,
                            settings=Settings(
                                anonymized_telemetry=False, allow_reset=True
                            ),
                        )
                    # Embedding fonksiyonu VERMİYORUZ; vektörleri kendimiz veriyoruz.
                    self._collection = self._client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},  # Kosinüs benzerliği.
                    )
                    return self._collection
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    self._collection = None
                    logger.warning(
                        "Vektör veritabanı başlatma denemesi %d/3 başarısız: %s",
                        attempt, exc,
                    )
                    time.sleep(0.5 * attempt)  # Artan kısa bekleme.

            raise RuntimeError(
                f"Vektör veritabanı başlatılamadı (3 deneme): {last_exc}"
            ) from last_exc

    # ------------------------------------------------------------------ #
    #  Yazma
    # ------------------------------------------------------------------ #
    def add_chunks(self, chunks: List["object"]) -> int:
        """
        DocumentChunk listesini vektörleştirip veritabanına ekler.

        :param chunks: document_processor.DocumentChunk nesneleri
                       (text, source, page, chunk_index alanları beklenir).
        :return: Eklenen parça sayısı.
        """
        if not chunks:
            return 0

        # Boilerplate dedup: normalize formu daha önce görülen (tekrarlayan
        # başlık/footer) chunk'ları atla. Benzersiz içerik daima korunur.
        deduped_chunks = []
        skipped = 0
        for chunk in chunks:
            norm = _norm_for_dedup(getattr(chunk, "text", "") or "")
            if norm and norm in self._seen_norm:
                skipped += 1
                continue
            if norm:
                self._seen_norm.add(norm)
            deduped_chunks.append(chunk)
        self.last_deduped = skipped
        if skipped:
            logger.info("Dedup: %d tekrarlayan boilerplate chunk atlandı.", skipped)
        chunks = deduped_chunks
        if not chunks:
            return 0

        collection = self._ensure_collection()

        documents: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []

        for chunk in chunks:
            documents.append(chunk.text)
            source = getattr(chunk, "source", "bilinmiyor")
            metadatas.append(
                {
                    # 'source' geriye dönük uyumluluk (retrieval bunu okur);
                    # 'source_file' yeni, açık adlandırma (ikisi aynı değer).
                    "source": source,
                    "source_file": source,
                    "page": int(getattr(chunk, "page", 0)),
                    "chunk_index": int(getattr(chunk, "chunk_index", 0)),
                    "ocr_used": bool(getattr(chunk, "ocr_used", False)),
                    "rotation": int(getattr(chunk, "rotation", 0)),
                    "char_count": int(getattr(chunk, "char_count", 0) or len(chunk.text)),
                }
            )
            ids.append(str(uuid.uuid4()))

        embedding_time = 0.0
        write_time = 0.0
        try:
            # Büyük dosyalarda belleği korumak için partiler halinde ekle.
            batch_size = 64
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                batch_docs = documents[start:end]
                # Vektörleri biz hesaplıyoruz (ChromaDB'nin EF mekanizmasını kullanmadan).
                t0 = time.perf_counter()
                batch_embeddings = self._embedder.embed_documents(batch_docs)
                t1 = time.perf_counter()
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )
                t2 = time.perf_counter()
                embedding_time += t1 - t0
                write_time += t2 - t1
            self.last_embedding_time = round(embedding_time, 2)
            self.last_chroma_write_time = round(write_time, 2)
            self._lex_dirty = True  # lexical indeks yeniden kurulmalı
            logger.info("%d parça veritabanına eklendi.", len(documents))
            return len(documents)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Parçalar veritabanına eklenemedi: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Okuma / Arama
    # ------------------------------------------------------------------ #
    # Dense (kosinüs) mesafe eşiği. e5-base bu OCR'lı/çok-dilli korpusta mesafeleri
    # ~0.19-0.25 bandına sıkıştırıyor; bu yüzden eşik bu bandın altına çekilir.
    # Bir aday KORUNUR eğer: lexical (BM25) eşleşmesi varsa VEYA dense mesafesi
    # bu eşiğin altındaysa. Hiçbiri yoksa (alakasız soru) boş döner -> LLM
    # 'dokümanda bulunmuyor' demeli.
    DEFAULT_MAX_DISTANCE = 0.22
    RRF_K0 = 60  # Reciprocal Rank Fusion sabiti.
    # Lexical eşleşmede yalnızca AYIRT EDİCİ terimler sayılır. idf bu eşiğin
    # altındaki yaygın kelimeler (stopword: "ve", "nasıl", "sistem", "gemi"...)
    # her dokümanla eşleştiği için reddetme sinyalini bozar; bu yüzden elenir.
    IDF_FLOOR = 2.0

    def _ensure_lexical(self):
        """BM25 lexical indeksini (gerekirse) koleksiyondan kurar/yeniler."""
        if self._bm25 is not None and not self._lex_dirty:
            return self._bm25
        with self._init_lock:
            if self._bm25 is not None and not self._lex_dirty:
                return self._bm25
            collection = self._ensure_collection()
            data = collection.get(include=["documents", "metadatas"])
            self._lex_ids = data.get("ids") or []
            self._lex_docs = data.get("documents") or []
            self._lex_metas = data.get("metadatas") or []
            t0 = time.perf_counter()
            self._bm25 = _BM25([_tokenize(d) for d in self._lex_docs])
            self._lex_dirty = False
            logger.info("BM25 lexical indeks kuruldu: %d parça (%.1f sn).",
                        len(self._lex_docs), time.perf_counter() - t0)
            return self._bm25

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        max_distance: Optional[float] = None,
    ) -> List[RetrievedContext]:
        """
        HİBRİT retrieval: dense (anlamsal, e5) + lexical (BM25 anahtar kelime),
        Reciprocal Rank Fusion (RRF) ile birleştirilir. Gürültülü/çok-dilli OCR
        korpusunda dense tek başına ayrım yapamadığından, tam terim eşleşmesi
        (BM25) sıralamayı ve 'alakasız soruyu reddetme' sinyalini güçlendirir.

        Bir aday KORUNUR eğer: BM25 skoru > 0 (terim eşleşti) VEYA dense mesafesi
        <= eşik. İkisi de yoksa elenir. Hiç aday kalmazsa boş liste döner.
        """
        if not query or not query.strip():
            return []

        threshold = self.DEFAULT_MAX_DISTANCE if max_distance is None else max_distance
        collection = self._ensure_collection()
        try:
            count = collection.count()
            if count == 0:
                return []
            fetch_k = min(count, max(k * 6, 40))

            # --- Dense aday havuzu ---
            query_embedding = self._embedder.embed_query(query)
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
            d_ids = (res.get("ids") or [[]])[0]
            d_docs = (res.get("documents") or [[]])[0]
            d_metas = (res.get("metadatas") or [[]])[0]
            d_dists = (res.get("distances") or [[]])[0]

            dense_rank: Dict[str, int] = {}
            dist_by_id: Dict[str, float] = {}
            info_by_id: Dict[str, tuple] = {}
            for r, (cid, doc, meta, dist) in enumerate(
                    zip(d_ids, d_docs, d_metas, d_dists)):
                dense_rank[cid] = r
                dist_by_id[cid] = float(dist)
                info_by_id[cid] = (doc, meta)

            # --- Lexical (BM25) aday havuzu ---
            bm25 = self._ensure_lexical()
            bm_rank: Dict[str, int] = {}
            bm_score_by_id: Dict[str, float] = {}
            q_tokens = _tokenize(query)
            # Yalnızca ayırt edici (yüksek idf) sorgu terimlerini kullan; yaygın
            # kelimeler her şeyle eşleşip alakasız soruları "ilgili" gösterirdi.
            content = [t for t in q_tokens if bm25.idf.get(t, 0.0) >= self.IDF_FLOOR]
            if content and bm25.N:
                scores = bm25.scores(content)
                top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                r = 0
                for i in top[:fetch_k]:
                    if scores[i] <= 0:
                        break
                    cid = self._lex_ids[i]
                    bm_rank[cid] = r
                    bm_score_by_id[cid] = scores[i]
                    info_by_id.setdefault(cid, (self._lex_docs[i], self._lex_metas[i]))
                    r += 1
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Benzerlik araması başarısız: {exc}") from exc

        # --- RRF füzyonu + reddetme kapısı ---
        big = 10 * fetch_k
        candidates = set(dense_rank) | set(bm_rank)
        fused = {
            cid: 1.0 / (self.RRF_K0 + dense_rank.get(cid, big))
                 + 1.0 / (self.RRF_K0 + bm_rank.get(cid, big))
            for cid in candidates
        }
        ordered = sorted(candidates, key=lambda c: fused[c], reverse=True)

        contexts: List[RetrievedContext] = []
        for cid in ordered:
            dist = dist_by_id.get(cid)
            has_lex = bm_score_by_id.get(cid, 0.0) > 0
            close = dist is not None and dist <= threshold
            if not (has_lex or close):
                continue  # ne terim eşleşmesi ne yakın dense -> alakasız
            doc, meta = info_by_id[cid]
            contexts.append(RetrievedContext(
                text=doc,
                source=(meta or {}).get("source", "bilinmiyor"),
                page=(meta or {}).get("page", 0),
                score=dist if dist is not None else 1.0,
            ))
            if len(contexts) >= k:
                break
        return contexts

    # ------------------------------------------------------------------ #
    #  Yardımcı / Yönetim
    # ------------------------------------------------------------------ #
    def get_document_count(self) -> int:
        """Veritabanındaki toplam parça sayısını döndürür."""
        try:
            return self._ensure_collection().count()
        except Exception as exc:  # noqa: BLE001
            logger.error("Parça sayısı alınamadı: %s", exc)
            return 0

    def list_sources(self) -> List[str]:
        """Veritabanında kayıtlı benzersiz kaynak dosya adlarını döndürür."""
        try:
            collection = self._ensure_collection()
            if collection.count() == 0:
                return []
            data = collection.get(include=["metadatas"])
            sources = {
                (m or {}).get("source", "bilinmiyor")
                for m in (data.get("metadatas") or [])
            }
            return sorted(sources)
        except Exception as exc:  # noqa: BLE001
            logger.error("Kaynak listesi alınamadı: %s", exc)
            return []

    def clear(self) -> None:
        """Tüm vektör veritabanını sıfırlar (tüm dokümanları siler)."""
        with self._init_lock:
            try:
                if self._client is None:
                    self._ensure_collection()
                if self._client is not None:
                    self._client.delete_collection(self.collection_name)
                self._collection = None
                # Koleksiyonu yeniden oluştur.
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._bm25 = None
                self._lex_dirty = True
                self._seen_norm = set()  # dedup durumunu da sıfırla
                logger.info("Vektör veritabanı temizlendi.")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Veritabanı temizlenemedi: {exc}") from exc

    def warm_up(self) -> None:
        """
        Embedding modelini ve veritabanını önceden yükler.
        UI'da 'hazırlanıyor' aşamasında arka planda çağrılması önerilir.
        """
        self._ensure_collection()
        # Küçük bir kodlama ile modeli belleğe çek.
        self._embedder.embed_query("hazırlık")
