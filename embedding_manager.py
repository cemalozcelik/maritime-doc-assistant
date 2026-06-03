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
import time
import uuid
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


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

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        self._model_name_or_path = model_name_or_path
        self._device = device
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
        device: str = "cpu",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embedder = LocalEmbedder(model_name_or_path, device=device)

        os.makedirs(self.persist_directory, exist_ok=True)

        self._client = None
        self._collection = None
        # Birden fazla thread'in aynı anda koleksiyon oluşturmasını engeller.
        self._init_lock = threading.Lock()

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

        collection = self._ensure_collection()

        documents: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []

        for chunk in chunks:
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": getattr(chunk, "source", "bilinmiyor"),
                    "page": int(getattr(chunk, "page", 0)),
                    "chunk_index": int(getattr(chunk, "chunk_index", 0)),
                }
            )
            ids.append(str(uuid.uuid4()))

        try:
            # Büyük dosyalarda belleği korumak için partiler halinde ekle.
            batch_size = 64
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                batch_docs = documents[start:end]
                # Vektörleri biz hesaplıyoruz (ChromaDB'nin EF mekanizmasını kullanmadan).
                batch_embeddings = self._embedder.embed_documents(batch_docs)
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )
            logger.info("%d parça veritabanına eklendi.", len(documents))
            return len(documents)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Parçalar veritabanına eklenemedi: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Okuma / Arama
    # ------------------------------------------------------------------ #
    # Kosinüs mesafesi eşiği (0 = aynı, 2 = zıt). Bu değerden UZAK (büyük) parçalar
    # alakasız kabul edilip elenir. e5 modelinde ilgili parçalar tipik olarak
    # ~0.0-0.45 aralığında, alakasız/çöp (çizim OCR'ı vb.) ise daha büyük çıkar.
    # Çok düşük tutarsanız hiç sonuç gelmez; çok yüksek tutarsanız çöp sızar.
    DEFAULT_MAX_DISTANCE = 0.55

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        max_distance: Optional[float] = None,
    ) -> List[RetrievedContext]:
        """
        Soruya en benzer 'k' adet bağlam parçasını döndürür.

        Önce 'k'dan daha fazla aday çeker, ardından kosinüs mesafesi
        'max_distance' eşiğinden büyük (alakasız) olanları eler ve geriye
        kalanların en iyi 'k' tanesini döndürür. Böylece çizim OCR'ı gibi
        anlamsız parçalar bağlama girmez. Veritabanı boşsa boş liste döner.

        :param max_distance: None ise sınıf varsayılanı (DEFAULT_MAX_DISTANCE)
                             kullanılır. Eşiği geçen parça yoksa boş liste döner
                             (bu durumda LLM 'dokümanda bulunmuyor' demeli).
        """
        if not query or not query.strip():
            return []

        threshold = self.DEFAULT_MAX_DISTANCE if max_distance is None else max_distance
        collection = self._ensure_collection()

        try:
            count = collection.count()
            if count == 0:
                return []

            # Eşik elemesinin işe yaraması için 'k'dan fazla aday çek.
            fetch_k = min(count, max(k * 3, 20))
            query_embedding = self._embedder.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Benzerlik araması başarısız: {exc}") from exc

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        contexts: List[RetrievedContext] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            distance = float(dist)
            # Eşikten uzak (alakasız) parçaları ele.
            if distance > threshold:
                continue
            contexts.append(
                RetrievedContext(
                    text=doc,
                    source=(meta or {}).get("source", "bilinmiyor"),
                    page=(meta or {}).get("page", 0),
                    score=distance,
                )
            )

        # ChromaDB zaten mesafeye göre artan sıralı döndürür; en iyi 'k' tanesini al.
        return contexts[:k]

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
