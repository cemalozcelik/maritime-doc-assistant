# -*- coding: utf-8 -*-
"""
document_processor.py
=====================
PDF ve görsel dosyalarını işleyerek metin parçalarına (chunk) dönüştüren modül.

Sorumluluklar (Tek Sorumluluk Prensibi - SRP):
    * PDF'lerden metin çıkarma (PyMuPDF / fitz)
    * Metin içermeyen (taranmış/scanned) PDF sayfalarını yüksek DPI görüntüye
      çevirip OCR ile okuma (sayfa rotasyonunu otomatik tespit ederek)
    * Görsellerden lokal OCR (EasyOCR) ile metin çıkarma
    * OCR çıktısını güvenli (sürümden bağımsız) ayrıştırma
    * Çıkarılan metni temizleyip RecursiveCharacterTextSplitter ile parçalara bölme

Bu modül LLM veya veritabanı hakkında HİÇBİR ŞEY bilmez. Sadece "dosya -> metin
parçaları" dönüşümünden sorumludur.

Not: OCR motoru EasyOCR'dır. Aşağıdaki `extract_text_from_ocr_result` yardımcısı,
hem EasyOCR ((bbox, text, conf) / (bbox, text)) hem de PaddleOCR ([bbox, (text,
conf)]) çıktı biçimlerini güvenle ayrıştırır; tek bir indeks varsayımına güvenmez.
"""

from __future__ import annotations

import os
import io
import re
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

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


# ===========================================================================
#  OCR Çıktısı Ayrıştırma (sürümden bağımsız, güvenli)
# ===========================================================================
def _ocr_item_text(item: Any) -> str:
    """Tek bir OCR sonucu öğesinden metni güvenle çıkarır (biçim ne olursa olsun)."""
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "transcription", "label", "rec_text"):
            val = item.get(key)
            if isinstance(val, str):
                return val
        return ""
    if isinstance(item, (list, tuple)):
        # EasyOCR detail=1: (bbox, text, conf) -> item[1] str
        if len(item) >= 2 and isinstance(item[1], str):
            return item[1]
        # PaddleOCR: [bbox, (text, conf)] -> item[1][0] str
        if (len(item) >= 2 and isinstance(item[1], (list, tuple))
                and item[1] and isinstance(item[1][0], str)):
            return item[1][0]
        # Genel: ilk düz string elemanı (bbox'lar sayısal olduğu için atlanır)
        for el in item:
            if isinstance(el, str):
                return el
        # İç içe yapılar için özyinelemeli ara
        for el in item:
            txt = _ocr_item_text(el)
            if txt:
                return txt
    return ""


def _ocr_item_conf(item: Any) -> Optional[float]:
    """Bir OCR öğesinden güven (confidence) skorunu çıkarır; yoksa None."""
    if isinstance(item, (list, tuple)):
        # EasyOCR: (bbox, text, conf)
        if len(item) >= 3 and isinstance(item[2], (int, float)):
            return float(item[2])
        # PaddleOCR: [bbox, (text, conf)]
        if (len(item) >= 2 and isinstance(item[1], (list, tuple))
                and len(item[1]) >= 2 and isinstance(item[1][1], (int, float))):
            return float(item[1][1])
    if isinstance(item, dict):
        for key in ("confidence", "score", "conf"):
            val = item.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def extract_text_from_ocr_result(result: Any, debug_raw_path: Optional[str] = None) -> str:
    """
    OCR sonucundan birleşik metni güvenle çıkarır. Hangi OCR sürümü/biçimi olursa
    olsun (EasyOCR / PaddleOCR / düz string listesi) çalışır; tek bir indeks
    varsayımına (line[1][0] vb.) güvenmez.

    Bir Exception oluşursa pipeline ÇÖKMEZ: boş string döner ve (verilmişse)
    ham sonucu debug dosyasına yazar.
    """
    try:
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()
        lines: List[str] = []
        for item in result:
            txt = _ocr_item_text(item)
            if txt and txt.strip():
                lines.append(txt.strip())
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR sonucu ayrıştırılamadı (%s); ham çıktı debug'a yazılıyor.", exc)
        if debug_raw_path:
            try:
                os.makedirs(os.path.dirname(debug_raw_path), exist_ok=True)
                with open(debug_raw_path, "w", encoding="utf-8") as fh:
                    fh.write(repr(result))
            except Exception:  # noqa: BLE001
                pass
        return ""


# Kelime-benzeri token (en az 2 harf; Türkçe dahil). Gerçek metni OCR
# gürültüsünden (kopuk semboller, rastgele kısa parçalar) ayırmak için kullanılır.
_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}")


def meaningful_len(text: str) -> int:
    """
    'Anlamlı' karakter sayısı: en az 2 harften oluşan kelime-benzeri token'ların
    toplam uzunluğu. Yanlış rotasyondan / saf gürültüden gelen metinde bu değer
    çok düşük, gerçek metinde yüksektir. Rotasyon kabul/ret kararında kullanılır.
    """
    if not text:
        return 0
    return sum(len(w) for w in _WORD_RE.findall(text))


def _score_ocr_result(result: Any, text: str) -> float:
    """
    Bir OCR sonucunu puanlar (rotasyon seçimi için). Güven skorları varsa
    sum(len(metin) * güven) kullanılır; yoksa toplam metin uzunluğu.
    Daha yüksek = daha iyi (daha çok güvenilir metin).
    """
    try:
        total = 0.0
        for item in result or []:
            txt = _ocr_item_text(item)
            if not txt:
                continue
            conf = _ocr_item_conf(item)
            total += len(txt) * (conf if conf is not None else 1.0)
        if total > 0:
            return total
    except Exception:  # noqa: BLE001
        pass
    return float(len(text or ""))


# ===========================================================================
#  Metin Temizleme (nazik; Türkçe karakter ve sembolleri bozmaz)
# ===========================================================================
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INLINE_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def clean_ocr_text(text: str) -> str:
    """
    OCR metnini NAZİKÇE temizler:
      * Kontrol karakterlerini boşluğa çevirir.
      * Satır içi fazla boşlukları teke indirir.
      * 3+ boş satırı 2'ye indirir, satırları korur.
    Türkçe karakterleri (ç, ğ, ı, ö, ş, ü), teknik sembolleri ve formülleri SİLMEZ.
    """
    if not text:
        return ""
    text = _CONTROL_RE.sub(" ", text)
    lines = [_INLINE_WS_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    out = "\n".join(lines)
    out = _MULTI_NL_RE.sub("\n\n", out)
    return out.strip()


# ===========================================================================
#  İçe Aktarım Raporu
# ===========================================================================
def format_ingestion_report(stats: Dict[str, Any]) -> str:
    """Bir içe aktarım (ingestion) istatistik sözlüğünü okunaklı rapora çevirir."""
    failed = stats.get("failed_pages") or []
    lines = [
        "=== INGESTION RAPORU ===",
        f"  Kaynak              : {stats.get('source', '-')}",
        f"  Toplam sayfa        : {stats.get('total_pages', 0)}",
        f"  Metin cikan sayfa   : {stats.get('pages_with_text', 0)}",
        f"  OCR kullanilan sayfa: {stats.get('ocr_pages', 0)}",
        f"  Toplam karakter     : {stats.get('total_chars', 0)}",
        f"  Toplam chunk        : {stats.get('total_chunks', 0)}",
        f"  OCR suresi          : {stats.get('ocr_time', 0)} sn",
        f"  Embedding suresi    : {stats.get('embedding_time', 0)} sn",
        f"  Chroma yazma suresi : {stats.get('chroma_write_time', 0)} sn",
        f"  Basarisiz sayfa     : {len(failed)}" + (f" -> {failed}" if failed else ""),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Veri Modelleri
# ---------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """Tek bir metin parçasını ve kaynağına dair üst veriyi (metadata) tutar."""
    text: str
    source: str            # Kaynak dosya adı
    page: int = 0          # PDF sayfa numarası (görseller için 0)
    chunk_index: int = 0   # Parçanın dosya içindeki sıra numarası
    ocr_used: bool = False  # Bu parça OCR ile mi okundu?
    rotation: int = 0      # Sayfa için seçilen rotasyon (0/90/180/270)
    char_count: int = 0    # Parçadaki karakter sayısı


@dataclass
class ProcessResult:
    """Bir dosyanın işlenmesi sonucu üretilen tüm parçaları ve durumu kapsar."""
    chunks: List[DocumentChunk] = field(default_factory=list)
    success: bool = True
    message: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Ana İşleyici Sınıfı
# ---------------------------------------------------------------------------
class DocumentProcessor:
    """
    PDF ve görselleri işleyip metin parçaları üreten sınıf.

    OCR motoru (EasyOCR) yüklenmesi pahalı olduğu için 'lazy loading' ile yalnızca
    ilk OCR ihtiyacında başlatılır. Taranmış sayfalar yüksek DPI'da render edilir,
    sayfa rotasyonu (0/90/180/270) otomatik denenir ve en çok metin veren açı seçilir.
    """

    # OCR uygulanmadan önce PDF'ten en az bu kadar karakter çıkmalı.
    # Daha azı çıkarsa sayfa "taranmış (scanned)" kabul edilip OCR'a yönlendirilir.
    MIN_TEXT_THRESHOLD = 20

    # Rotasyon denemesinde İLK denenen açı (0 veya dominant) bu kadar metin
    # verdiyse diğer açıları DENEME (gereksiz 4x OCR'dan kaçınılır).
    ROT_FASTPATH_CHARS = 240
    # Dominant açı belge için zaten DOĞRU bilindiğinden, az da olsa gerçek metin
    # verdiyse ona güveniriz; diğer açılara yalnızca sonuç bu kadar bile metin
    # vermezse (sayfa yanlış yönde veya boş) düşeriz. Düşük eşik = bulk sayfalarda
    # tek OCR çağrısı = hız.
    ROT_ACCEPT_MIN = 25
    # Bir sayfanın dominant rotasyon TESPİTİNE katkı sayılması için (güvenilir
    # sinyal) gereken en az metin. Tespit eşiği kabul eşiğinden yüksek tutulur.
    ROT_PREFERRED_MIN = 80

    # Belge-geneli dominant rotasyon tespiti: ilk bu kadar METİNLİ sayfa tüm
    # açılarda taranır; sonra dominant açı belirlenip kalan sayfalarda önce o
    # denenir (zayıf çıkarsa diğerlerine fallback). Tüm sayfaları 4x taramaktan
    # çok daha hızlıdır.
    ROT_PROBE_PAGES = 5   # En fazla bu kadar sayfa tam taranır.
    ROT_PROBE_MIN = 3     # Bu kadar prob sonrası net çoğunluk varsa erken karar.

    # --- Chunk kalite filtresi (GEVŞETİLMİŞ) ---
    # Eski sürümdeki agresif gibberish filtresi gerçek teknik metni de eliyordu.
    # Artık yalnızca açıkça çöp olan (neredeyse hiç harf içermeyen) parçalar elenir.
    MIN_ALPHA_RATIO = 0.15        # Harf oranı bunun altındaysa (saf sembol/rakam) ele.

    SUPPORTED_PDF = (".pdf",)
    SUPPORTED_IMAGE = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ocr_languages: Optional[List[str]] = None,
        enable_pdf_ocr_fallback: bool = True,
        ocr_gpu: Optional[bool] = None,
        ocr_dpi: int = 300,
        min_chunk_chars: int = 60,
        try_rotations: bool = True,
        debug_dir: Optional[str] = None,
    ) -> None:
        """
        :param chunk_size: Her metin parçasının maksimum karakter uzunluğu.
        :param chunk_overlap: Bağlamı korumak için parçalar arası örtüşme miktarı.
        :param ocr_languages: OCR dilleri (varsayılan: Türkçe + İngilizce).
        :param enable_pdf_ocr_fallback: Metinsiz PDF sayfalarında OCR denensin mi?
        :param ocr_gpu: OCR için GPU? None ise otomatik (CUDA'lı torch varsa GPU).
        :param ocr_dpi: Taranmış sayfaların OCR için render DPI'ı (200-300 önerilir).
        :param min_chunk_chars: Bir parçanın korunması için en az karakter (50-100).
        :param try_rotations: Taranmış sayfalarda 0/90/180/270 rotasyon denensin mi?
        :param debug_dir: Verilirse, sayfa başına OCR metni 'debug_ocr/' altına,
                          boş sayfaların görüntüsü 'debug_failed_pages/' altına yazılır.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_languages = ocr_languages or ["tr", "en"]
        self.enable_pdf_ocr_fallback = enable_pdf_ocr_fallback
        self._ocr_gpu = ocr_gpu  # None = otomatik
        self.ocr_dpi = max(72, int(ocr_dpi))
        self.min_chunk_chars = max(1, int(min_chunk_chars))
        self.try_rotations = try_rotations
        self.debug_dir = debug_dir

        # OCR motoru ilk kullanımda yüklenecek (lazy).
        self._ocr_reader = None

        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters bulunamadı. "
                "Lütfen 'pip install langchain-text-splitters' çalıştırın."
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )

    # ------------------------------------------------------------------ #
    #  OCR Motoru (Lazy Loading)
    # ------------------------------------------------------------------ #
    def _detect_gpu(self) -> bool:
        """OCR için GPU kullanılıp kullanılmayacağını belirler (override veya otomatik)."""
        if self._ocr_gpu is not None:
            return self._ocr_gpu
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    def _get_ocr_reader(self):
        """EasyOCR okuyucusunu gerektiğinde (ilk OCR ihtiyacında) başlatır."""
        if self._ocr_reader is not None:
            return self._ocr_reader
        try:
            import easyocr  # Ağır bir import olduğu için fonksiyon içinde yapılıyor.
            use_gpu = self._detect_gpu()
            logger.info(
                "EasyOCR motoru yükleniyor (diller: %s, GPU: %s)...",
                self.ocr_languages, use_gpu,
            )
            self._ocr_reader = easyocr.Reader(self.ocr_languages, gpu=use_gpu)
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
        """Dosya uzantısına göre uygun işleyiciye yönlendiren ana metot."""
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
        """Bir klasörü (alt klasörler dahil) tarayıp desteklenen dosyaları döndürür."""
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
        """PDF'ten metin çıkarır; metinsiz sayfalarda rotasyon-bilinçli OCR'a düşer."""
        if fitz is None:
            return ProcessResult(
                success=False,
                message="PyMuPDF (fitz) kurulu değil. 'pip install PyMuPDF' gerekli.",
            )

        source = os.path.basename(file_path)
        all_chunks: List[DocumentChunk] = []
        ocr_pages = 0
        pages_with_text = 0
        total_chars = 0
        ocr_time = 0.0
        failed_pages: List[int] = []

        try:
            document = fitz.open(file_path)
        except Exception as exc:  # noqa: BLE001
            return ProcessResult(success=False, message=f"PDF açılamadı: {exc}")

        total_pages = len(document)
        # Belge-geneli dominant rotasyon: ilk birkaç metinli sayfadan belirlenir,
        # sonra kalan sayfalarda önce o denenir (hız için).
        probe_rotations: List[int] = []
        dominant_rotation: Optional[int] = None
        try:
            for page_number in range(total_pages):
                page = document[page_number]
                page_label = page_number + 1
                text_layer = (page.get_text() or "").strip()

                ocr_used = False
                rotation = 0
                rendered = None  # OCR'da render edilen numpy görüntü (debug için)

                if len(text_layer) >= self.MIN_TEXT_THRESHOLD:
                    page_text = clean_ocr_text(text_layer)
                elif self.enable_pdf_ocr_fallback:
                    t0 = time.perf_counter()
                    rendered = self._render_page_array(page)
                    best = self._ocr_image_best(
                        rendered, f"page_{page_label:03d}", preferred=dominant_rotation
                    )
                    ocr_time += time.perf_counter() - t0
                    page_text = clean_ocr_text(best["text"])
                    ocr_used = True
                    rotation = best["rotation"]
                    ocr_pages += 1
                    logger.info(
                        "page=%d best_rotation=%d text_len=%d score=%.0f%s",
                        page_label, rotation, len(page_text), best["score"],
                        " (dominant)" if dominant_rotation is not None else " (probe)",
                    )
                    # Dominant henüz belirlenmediyse, GÜVENİLİR metinli sayfalarla
                    # tespit et (anlamlı karakter yüksek eşik = sağlam sinyal).
                    if dominant_rotation is None and meaningful_len(best["text"]) >= self.ROT_PREFERRED_MIN:
                        probe_rotations.append(rotation)
                        if (len(probe_rotations) >= self.ROT_PROBE_PAGES
                                or self._rotation_decided(probe_rotations, self.ROT_PROBE_MIN)):
                            dominant_rotation = Counter(probe_rotations).most_common(1)[0][0]
                            logger.info(
                                "Dokuman dominant rotasyonu: %d derece (problar: %s)",
                                dominant_rotation, probe_rotations,
                            )
                else:
                    page_text = ""

                # Debug: sayfa metnini yaz (varsa); boşsa görüntüyü kaydet.
                self._write_page_debug(page_label, page_text, rendered)

                if page_text:
                    pages_with_text += 1
                    total_chars += len(page_text)
                    pieces = self._split_text(page_text)
                    if not pieces:
                        # OCR metin üretti ama chunker 0 üretti -> bug. Logla + kurtar.
                        logger.warning(
                            "WARNING: OCR produced text but chunker produced 0 chunks "
                            "(page=%d, chars=%d).", page_label, len(page_text)
                        )
                        pieces = [page_text]  # Veriyi kaybetme.
                    for idx, piece in enumerate(pieces):
                        all_chunks.append(DocumentChunk(
                            text=piece, source=source, page=page_label,
                            chunk_index=idx, ocr_used=ocr_used,
                            rotation=rotation, char_count=len(piece),
                        ))
                else:
                    failed_pages.append(page_label)
        finally:
            document.close()

        stats = {
            "source": source,
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "ocr_pages": ocr_pages,
            "total_chars": total_chars,
            "total_chunks": len(all_chunks),
            "ocr_time": round(ocr_time, 2),
            "failed_pages": failed_pages,
        }

        if total_chars > 0 and not all_chunks:
            # Bu noktaya normalde gelinmez (kurtarma var); yine de açıkça hata say.
            logger.warning(
                "WARNING: OCR produced text but chunker produced 0 chunks (%s).", source
            )

        if not all_chunks:
            if total_chars == 0:
                msg = (f"'{source}': hiçbir sayfadan metin çıkarılamadı "
                       f"({len(failed_pages)}/{total_pages} sayfa boş).")
                if self.debug_dir:
                    msg += f" Görseller '{self._failed_dir()}' altına kaydedildi."
            else:
                msg = f"'{source}': metin çıktı ama parça üretilemedi (chunker hatası)."
            return ProcessResult(success=False, message=msg, stats=stats)

        msg = f"'{source}' işlendi: {len(all_chunks)} parça."
        if ocr_pages:
            msg += f" ({ocr_pages} sayfa OCR ile okundu.)"
        if failed_pages:
            msg += f" {len(failed_pages)} sayfa boş kaldı."
        return ProcessResult(chunks=all_chunks, success=True, message=msg, stats=stats)

    def _render_page_array(self, page):
        """PDF sayfasını ocr_dpi çözünürlükte numpy RGB diziye render eder."""
        import numpy as np
        zoom = self.ocr_dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        if Image is not None:
            img = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            return np.array(img)
        # Pillow yoksa doğrudan pixmap tamponundan diziye çevir.
        data = np.frombuffer(pixmap.samples, dtype=np.uint8)
        return data.reshape(pixmap.height, pixmap.width, pixmap.n)

    # ------------------------------------------------------------------ #
    #  OCR Çekirdeği (rotasyon tespitli)
    # ------------------------------------------------------------------ #
    def _ocr_at_angle(self, arr, angle: int, label: str) -> Dict[str, Any]:
        """Görüntüyü tek bir açıda OCR eder: {text, rotation, score}."""
        import numpy as np
        k = (angle // 90) % 4
        rotated = arr if k == 0 else np.rot90(arr, k)
        rotated = np.ascontiguousarray(rotated)  # OCR için bellek-bitişik.
        try:
            result = self._get_ocr_reader().readtext(rotated, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR hata (%s, %d derece): %s", label, angle, exc)
            return {"text": "", "rotation": angle, "score": -1.0}
        dbg = None
        if self.debug_dir:
            dbg = os.path.join(self._ocr_dir(), f"{label}_rot{angle}_raw.txt")
        text = extract_text_from_ocr_result(result, debug_raw_path=dbg)
        return {"text": text, "rotation": angle, "score": _score_ocr_result(result, text)}

    def _ocr_image_best(self, arr, label: str,
                        preferred: Optional[int] = None) -> Dict[str, Any]:
        """
        Bir görüntüyü OCR eder ve en çok anlamlı metin veren rotasyonu seçer.

        * try_rotations kapalıysa yalnızca 0 derece denenir.
        * preferred (dominant açı) verilmişse ÖNCE o denenir; yeterli metin
          (>= ROT_PREFERRED_MIN) verirse diğer açılar DENENMEZ (hız). Zayıfsa
          kalan açılara fallback yapılır.
        * preferred yoksa 0,90,180,270 tam taranır (0 derece bol metin verirse
          erken durulur).
        """
        if not self.try_rotations:
            return self._ocr_at_angle(arr, 0, label)

        order = [0, 90, 180, 270]
        if preferred is not None:
            order = [preferred] + [a for a in order if a != preferred]

        best: Dict[str, Any] = {"text": "", "rotation": 0, "score": -1.0}
        for i, angle in enumerate(order):
            cur = self._ocr_at_angle(arr, angle, label)
            if cur["score"] > best["score"]:
                best = cur
            if i == 0:
                # İlk denenen açı (0 veya dominant) zaten güçlüyse: bitir.
                if meaningful_len(cur["text"]) >= self.ROT_FASTPATH_CHARS:
                    break
                # Dominant açı (doğru bilinen) az da olsa ANLAMLI metin verdiyse
                # güven; sadece neredeyse boş veya saf gürültüyse diğerlerine düş.
                if preferred is not None and meaningful_len(cur["text"]) >= self.ROT_ACCEPT_MIN:
                    break
        return best

    @staticmethod
    def _rotation_decided(probes: List[int], min_n: int) -> bool:
        """En az min_n prob varsa ve net çoğunluk (yarıdan fazla) oluştuysa True."""
        if len(probes) < min_n:
            return False
        _angle, count = Counter(probes).most_common(1)[0]
        return count > len(probes) / 2

    # ------------------------------------------------------------------ #
    #  Görsel İşleme (OCR)
    # ------------------------------------------------------------------ #
    def _process_image(self, file_path: str) -> ProcessResult:
        """Bir görsel dosyasını rotasyon-bilinçli OCR ile okuyup parçalar üretir."""
        import numpy as np
        source = os.path.basename(file_path)
        try:
            if Image is not None:
                arr = np.array(Image.open(file_path).convert("RGB"))
            else:
                with open(file_path, "rb") as fh:
                    arr = fh.read()
            best = self._ocr_image_best(arr, os.path.splitext(source)[0])
            text = clean_ocr_text(best["text"])
        except Exception as exc:  # noqa: BLE001
            return ProcessResult(success=False, message=f"Görsel OCR hatası: {exc}")

        if not text:
            return ProcessResult(
                success=False, message=f"'{source}' görselinde metin bulunamadı.",
                stats={"source": source, "total_pages": 1, "pages_with_text": 0,
                       "ocr_pages": 1, "total_chars": 0, "total_chunks": 0,
                       "failed_pages": [1]},
            )

        pieces = self._split_text(text) or [text]
        chunks = [
            DocumentChunk(text=piece, source=source, page=0, chunk_index=idx,
                          ocr_used=True, rotation=best["rotation"], char_count=len(piece))
            for idx, piece in enumerate(pieces)
        ]
        stats = {"source": source, "total_pages": 1, "pages_with_text": 1,
                 "ocr_pages": 1, "total_chars": len(text), "total_chunks": len(chunks),
                 "failed_pages": []}
        return ProcessResult(
            chunks=chunks, success=True,
            message=f"'{source}' OCR ile okundu: {len(chunks)} parça.", stats=stats,
        )

    # ------------------------------------------------------------------ #
    #  Metin Parçalama (GEVŞETİLMİŞ filtre)
    # ------------------------------------------------------------------ #
    def _split_text(self, text: str) -> List[str]:
        """
        Sayfa metnini örtüşmeli parçalara böler ve YALNIZCA açıkça çöp olan
        (neredeyse hiç harf içermeyen) parçaları eler. Çok kısa satırlar tek tek
        atılmaz; metin önce birleşik haldedir, sonra chunk'lanır. Filtre tüm
        parçaları elerse veri kaybını önlemek için en az birkaç harf içeren
        parçalar (gerekirse ham parçalar) korunur.
        """
        if not text or not text.strip():
            return []
        raw = [c.strip() for c in self._splitter.split_text(text)]
        raw = [c for c in raw if c]
        kept = [c for c in raw if not self._is_garbage(c)]
        if not kept and raw:
            # Filtre her şeyi eledi: en az 3 harf içerenleri kurtar, o da yoksa ham.
            kept = [c for c in raw if sum(ch.isalpha() for ch in c) >= 3] or raw
        return kept

    def _is_garbage(self, text: str) -> bool:
        """
        Sadece AÇIKÇA çöp parçaları eler (eski agresif filtre kaldırıldı).
        Çöp = çok kısa, ya da neredeyse hiç harf içermeyen (saf sembol/rakam).
        """
        t = text.strip()
        if len(t) < self.min_chunk_chars:
            return True
        letters = sum(1 for c in t if c.isalpha())
        if letters == 0:
            return True
        if letters / len(t) < self.MIN_ALPHA_RATIO:
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Debug Çıktısı
    # ------------------------------------------------------------------ #
    def _ocr_dir(self) -> str:
        return os.path.join(self.debug_dir, "debug_ocr")

    def _failed_dir(self) -> str:
        return os.path.join(self.debug_dir, "debug_failed_pages")

    def _write_page_debug(self, page_label: int, page_text: str, rendered) -> None:
        """debug_dir verilmişse: sayfa metnini txt'ye, boş sayfayı png'ye yazar."""
        if not self.debug_dir:
            return
        try:
            ocr_dir = self._ocr_dir()
            os.makedirs(ocr_dir, exist_ok=True)
            with open(os.path.join(ocr_dir, f"page_{page_label:03d}.txt"),
                      "w", encoding="utf-8") as fh:
                fh.write(page_text or "")
            if not page_text and rendered is not None and Image is not None:
                failed_dir = self._failed_dir()
                os.makedirs(failed_dir, exist_ok=True)
                Image.fromarray(rendered).save(
                    os.path.join(failed_dir, f"page_{page_label:03d}.png")
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Debug çıktısı yazılamadı (page=%d): %s", page_label, exc)
