# -*- coding: utf-8 -*-
"""
document_processor.py
=====================
PDF ve görsel dosyalarını işleyerek metin parçalarına (chunk) dönüştüren modül.

Sorumluluklar (Tek Sorumluluk Prensibi - SRP):
    * PDF'lerden metin çıkarma (PyMuPDF / fitz)
    * Metin içermeyen (taranmış/scanned) PDF sayfalarını görüntüye çevirip OCR ile okuma
    * Görsellerden lokal OCR (EasyOCR) ile metin çıkarma
    * Çıkarılan metni RecursiveCharacterTextSplitter ile anlamlı parçalara bölme

Bu modül LLM veya veritabanı hakkında HİÇBİR ŞEY bilmez. Sadece "dosya -> metin parçaları"
dönüşümünden sorumludur. Bu sayede embedding/LLM katmanlarından bağımsızdır.
"""

from __future__ import annotations

import os
import io
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

# PyMuPDF (fitz) -> PDF okuma ve sayfa render etme
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

# LangChain metin parçalayıcı
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Eski LangChain sürümleri için geri uyumluluk
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

# Pillow -> görsel açma / OCR ön-işleme
try:
    from PIL import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Veri Modelleri
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """Tek bir metin parçasını ve kaynağına dair üst veriyi (metadata) tutar."""
    text: str
    source: str           # Kaynak dosya adı
    page: int = 0         # PDF sayfa numarası (görseller için 0)
    chunk_index: int = 0  # Parçanın dosya içindeki sıra numarası


@dataclass
class ProcessResult:
    """Bir dosyanın işlenmesi sonucu üretilen tüm parçaları ve durumu kapsar."""
    chunks: List[DocumentChunk] = field(default_factory=list)
    success: bool = True
    message: str = ""


# ---------------------------------------------------------------------------
#  Ana İşleyici Sınıfı
# ---------------------------------------------------------------------------
class DocumentProcessor:
    """
    PDF ve görselleri işleyip metin parçaları üreten sınıf.

    OCR motoru (EasyOCR) yüklenmesi pahalı bir işlem olduğu için 'lazy loading'
    (gerektiğinde yükleme) yöntemiyle yalnızca ilk OCR ihtiyacında başlatılır.
    """

    # OCR uygulanmadan önce PDF'ten en az bu kadar karakter çıkmalı.
    # Daha azı çıkarsa sayfa "taranmış (scanned)" kabul edilip OCR'a yönlendirilir.
    MIN_TEXT_THRESHOLD = 20

    # --- Metin kalite filtresi eşikleri (OCR çöpünü ayıklamak için) ---
    # Taranmış teknik çizim/diyagramların OCR'ı çoğu zaman anlamsız karakter
    # çorbası üretir (ör. "MGONTRgE OYSTFGNE", "63620 r 1 ~ C) l\"T1"). Bu tür
    # parçalar veritabanını kirletip alakasız sonuçların en üste çıkmasına yol
    # açar. Aşağıdaki ölçütlerden biri bile sağlanmazsa parça "düşük kaliteli"
    # sayılıp atılır.
    MIN_CHUNK_CHARS = 15          # Bu uzunluğun altındaki parçalar atılır.
    MIN_ALPHA_RATIO = 0.50        # Harf / toplam karakter oranı en az.
    MIN_WORD_RATIO = 0.50         # Kelime-benzeri token / toplam token oranı en az.
    MIN_WORD_COUNT = 3            # En az bu kadar kelime-benzeri token olmalı.
    MIN_VOWEL_RATIO = 0.20        # Harfler içinde ünlü oranı en az (gibberish eler).
    _VOWELS = set("aeiouâîûöüıAEIOUÂÎÛÖÜI")
    _WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}")

    SUPPORTED_PDF = (".pdf",)
    SUPPORTED_IMAGE = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ocr_languages: Optional[List[str]] = None,
        enable_pdf_ocr_fallback: bool = True,
    ) -> None:
        """
        :param chunk_size: Her metin parçasının maksimum karakter uzunluğu.
        :param chunk_overlap: Bağlamı korumak için parçalar arası örtüşme miktarı.
        :param ocr_languages: OCR dilleri (varsayılan: Türkçe + İngilizce).
        :param enable_pdf_ocr_fallback: Metinsiz PDF sayfalarında OCR denensin mi?
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_languages = ocr_languages or ["tr", "en"]
        self.enable_pdf_ocr_fallback = enable_pdf_ocr_fallback

        # OCR motoru ilk kullanımda yüklenecek (lazy).
        self._ocr_reader = None

        # Metin parçalayıcıyı hazırla.
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters bulunamadı. "
                "Lütfen 'pip install langchain-text-splitters' çalıştırın."
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Türkçe metinlerde de iyi çalışan ayraç hiyerarşisi:
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------ #
    #  OCR Motoru (Lazy Loading)
    # ------------------------------------------------------------------ #
    def _get_ocr_reader(self):
        """EasyOCR okuyucusunu gerektiğinde (ilk OCR ihtiyacında) başlatır."""
        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            import easyocr  # Ağır bir import olduğu için fonksiyon içinde yapılıyor.
            logger.info("EasyOCR motoru yükleniyor (diller: %s)...", self.ocr_languages)
            # gpu=False -> Gemi bilgisayarlarında GPU olmayabilir, CPU güvenli seçim.
            self._ocr_reader = easyocr.Reader(self.ocr_languages, gpu=False)
            logger.info("EasyOCR motoru hazır.")
            return self._ocr_reader
        except Exception as exc:  # noqa: BLE001
            logger.error("EasyOCR başlatılamadı: %s", exc)
            raise RuntimeError(
                f"OCR motoru başlatılamadı. EasyOCR kurulu mu? Hata: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    #  Genel Giriş Noktası
    # ------------------------------------------------------------------ #
    def process_file(self, file_path: str) -> ProcessResult:
        """
        Dosya uzantısına göre uygun işleyiciye yönlendiren ana metot.
        UI katmanı yalnızca bu metodu çağırır; iç ayrımları bilmesine gerek yoktur.
        """
        if not os.path.isfile(file_path):
            return ProcessResult(success=False, message=f"Dosya bulunamadı: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in self.SUPPORTED_PDF:
                return self._process_pdf(file_path)
            elif ext in self.SUPPORTED_IMAGE:
                return self._process_image(file_path)
            else:
                return ProcessResult(
                    success=False,
                    message=f"Desteklenmeyen dosya türü: '{ext}'. "
                            f"Desteklenenler: PDF ve görseller.",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Dosya işlenirken beklenmeyen hata: %s", file_path)
            return ProcessResult(success=False, message=f"İşleme hatası: {exc}")

    def collect_supported_files(self, folder: str) -> List[str]:
        """
        Bir klasörü (ve tüm alt klasörlerini) tarayıp desteklenen
        (PDF/görsel) dosyaların tam yollarını sıralı biçimde döndürür.
        """
        found: List[str] = []
        if not os.path.isdir(folder):
            return found
        supported = set(self.SUPPORTED_PDF) | set(self.SUPPORTED_IMAGE)
        for root, _dirs, files in os.walk(folder):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported:
                    found.append(os.path.join(root, filename))
        return sorted(found)

    # ------------------------------------------------------------------ #
    #  PDF İşleme
    # ------------------------------------------------------------------ #
    def _process_pdf(self, file_path: str) -> ProcessResult:
        """PDF'ten metin çıkarır; metinsiz sayfalarda OCR'a düşer."""
        if fitz is None:
            return ProcessResult(
                success=False,
                message="PyMuPDF (fitz) kurulu değil. 'pip install PyMuPDF' gerekli.",
            )

        source = os.path.basename(file_path)
        all_chunks: List[DocumentChunk] = []
        ocr_used_pages = 0

        try:
            document = fitz.open(file_path)
        except Exception as exc:  # noqa: BLE001
            return ProcessResult(success=False, message=f"PDF açılamadı: {exc}")

        try:
            for page_number in range(len(document)):
                page = document[page_number]
                text = (page.get_text() or "").strip()

                # Sayfada anlamlı metin yoksa ve OCR açıksa -> taranmış sayfa, OCR uygula.
                if len(text) < self.MIN_TEXT_THRESHOLD and self.enable_pdf_ocr_fallback:
                    ocr_text = self._ocr_pdf_page(page)
                    if ocr_text:
                        text = ocr_text
                        ocr_used_pages += 1

                if not text:
                    continue  # Tamamen boş sayfayı atla.

                # Sayfa metnini parçalara böl ve metadata ekle.
                for idx, piece in enumerate(self._split_text(text)):
                    all_chunks.append(
                        DocumentChunk(
                            text=piece,
                            source=source,
                            page=page_number + 1,
                            chunk_index=idx,
                        )
                    )
        finally:
            document.close()

        if not all_chunks:
            return ProcessResult(
                success=False,
                message=f"'{source}' içinden okunabilir metin çıkarılamadı.",
            )

        msg = f"'{source}' işlendi: {len(all_chunks)} parça."
        if ocr_used_pages:
            msg += f" ({ocr_used_pages} sayfa OCR ile okundu.)"
        return ProcessResult(chunks=all_chunks, success=True, message=msg)

    def _ocr_pdf_page(self, page) -> str:
        """Tek bir PDF sayfasını görüntüye çevirip OCR ile metnini okur."""
        try:
            # 2x ölçek -> OCR doğruluğunu artırmak için daha yüksek çözünürlük.
            matrix = fitz.Matrix(2.0, 2.0)
            pixmap = page.get_pixmap(matrix=matrix)
            image_bytes = pixmap.tobytes("png")
            return self._run_ocr_on_bytes(image_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF sayfası OCR edilemedi: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    #  Görsel İşleme (OCR)
    # ------------------------------------------------------------------ #
    def _process_image(self, file_path: str) -> ProcessResult:
        """Bir görsel dosyasını lokal OCR ile okuyup metin parçaları üretir."""
        source = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as fh:
                image_bytes = fh.read()
            text = self._run_ocr_on_bytes(image_bytes).strip()
        except Exception as exc:  # noqa: BLE001
            return ProcessResult(success=False, message=f"Görsel OCR hatası: {exc}")

        if not text:
            return ProcessResult(
                success=False,
                message=f"'{source}' görselinde metin bulunamadı.",
            )

        chunks = [
            DocumentChunk(text=piece, source=source, page=0, chunk_index=idx)
            for idx, piece in enumerate(self._split_text(text))
        ]
        return ProcessResult(
            chunks=chunks,
            success=True,
            message=f"'{source}' OCR ile okundu: {len(chunks)} parça.",
        )

    def _run_ocr_on_bytes(self, image_bytes: bytes) -> str:
        """Ham görsel byte verisi üzerinde EasyOCR çalıştırır ve birleşik metin döner."""
        reader = self._get_ocr_reader()

        # EasyOCR numpy array veya bytes kabul eder; Pillow ile normalize ediyoruz.
        if Image is not None:
            try:
                import numpy as np
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                ocr_input = np.array(pil_image)
            except Exception:  # noqa: BLE001
                ocr_input = image_bytes  # Pillow başarısız olursa ham byte ile dene.
        else:
            ocr_input = image_bytes

        # detail=0 -> sadece metin listesi döner (koordinat/skor olmadan).
        results = reader.readtext(ocr_input, detail=0, paragraph=True)
        return "\n".join(results)

    # ------------------------------------------------------------------ #
    #  Metin Parçalama
    # ------------------------------------------------------------------ #
    def _split_text(self, text: str) -> List[str]:
        """
        Uzun metni, bağlamı koruyan örtüşmeli parçalara böler ve düşük kaliteli
        (OCR çöpü) parçaları ayıklar.
        """
        if not text:
            return []
        pieces = (chunk.strip() for chunk in self._splitter.split_text(text))
        return [p for p in pieces if p and not self._is_low_quality(p)]

    def _is_low_quality(self, text: str) -> bool:
        """
        Bir metin parçasının anlamsız OCR çöpü olup olmadığını sezgisel olarak
        değerlendirir. Gerçek (TR/EN) cümleleri korur; taranmış çizimlerden çıkan
        karakter çorbasını eler. Sözlük/kütüphane gerektirmez (tamamen lokal).
        """
        t = text.strip()
        if len(t) < self.MIN_CHUNK_CHARS:
            return True

        letters = sum(1 for c in t if c.isalpha())
        if letters == 0:
            return True

        # 1) Harf oranı: çoğunluk sembol/rakamsa muhtemelen çizim/tablo çöpü.
        if letters / len(t) < self.MIN_ALPHA_RATIO:
            return True

        tokens = t.split()
        words = self._WORD_RE.findall(t)
        # 2) Token'ların çoğu kelime-benzeri değilse (kopuk semboller) ele.
        if not tokens or len(words) / len(tokens) < self.MIN_WORD_RATIO:
            return True
        # 3) Anlamlı olabilmesi için en az birkaç gerçek kelime olmalı.
        if len(words) < self.MIN_WORD_COUNT:
            return True

        # 4) Ünlü oranı çok düşükse (ör. "MGONTRG ZXCV") gibberish kabul et.
        vowels = sum(1 for c in t if c in self._VOWELS)
        if vowels / letters < self.MIN_VOWEL_RATIO:
            return True

        return False
