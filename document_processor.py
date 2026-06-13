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
import json
import time
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# PyMuPDF (fitz) -> PDF okuma ve sayfa render etme
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

# LangChain metin parçalayıcı LAZY yüklenir (bkz. _get_splitter). Bu paketin
# import'u ~8 sn sürdüğünden, modül/uygulama açılışını bloklamaması için yalnızca
# ilk doküman işlemede (arka planda) yüklenir. Aksi halde arayüz geç açılır.

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


# Geriye dönük/açık ad: meaningful_score == meaningful_len (yalnızca rotasyon
# kararında kullanılır; Latin/Türkçe odaklıdır).
meaningful_score = meaningful_len


def is_meaningful_text(text: str, min_letters: int = 12,
                       min_alpha_ratio: float = 0.45) -> bool:
    """
    Bir metnin "gerçek içerik" olup olmadığını SCRIPT-BAĞIMSIZ değerlendirir.
    `str.isalpha()` Latin, Türkçe, Korece (Hangul), CJK vb. tüm harfleri sayar;
    bu yüzden Korece manuel metni KORUNUR, ama sembol/rakam ağırlıklı OCR
    gürültüsü ("@K-504 9 2 #X# Fa-07") elenir.

    Kalite kapısında (gömülü çizim tespiti, chunk filtresi) kullanılır.
    `meaningful_score`'tan farkı: o yalnızca Latin kelime-benzeri token sayar ve
    SADECE rotasyon kararında kullanılır.
    """
    t = (text or "").strip()
    if not t:
        return False
    letters = sum(1 for c in t if c.isalpha())
    if letters < min_letters:
        return False
    if letters / len(t) < min_alpha_ratio:
        return False
    return True


# ===========================================================================
#  İçe Aktarım Modları, Çizim Tespiti, Dosya İmzası, OCR Önbelleği
# ===========================================================================
# Mod -> varsayılan ayarlar. Kullanıcı ocr_dpi / skip_drawings ile override edebilir.
INGEST_MODES = {
    "fast":     {"ocr_dpi": 150, "skip_drawings": True},
    "balanced": {"ocr_dpi": 200, "skip_drawings": True},   # varsayılan
    "full":     {"ocr_dpi": 300, "skip_drawings": False},
}
DEFAULT_INGEST_MODE = "balanced"

# Teknik çizim/şema dosyalarını adından tanı (OCR'ı çoğu zaman gürültü üretir).
_DRAWING_RE = re.compile(
    r"\b(drawing|drawings|diagram|diagrams|schematic|schematics|piping|plan|"
    r"plans|blueprint|dwg|layout|wiring)\b",
    re.IGNORECASE,
)


def is_drawing_file(name: str) -> bool:
    """Dosya adı teknik çizim/şema anahtar kelimesi içeriyor mu?"""
    return bool(_DRAWING_RE.search(name or ""))


def file_signature(path: str) -> str:
    """
    Dosya içeriğine dayalı hızlı imza (boyut + baş/orta/son 64 KB'ın SHA-1'i).
    Tam hash kadar kesin değil ama pratikte çakışmaz; büyük PDF'lerde hızlıdır.
    Önbellek anahtarının 'file_hash' bileşenidir.
    """
    h = hashlib.sha1()
    try:
        size = os.path.getsize(path)
        h.update(str(size).encode())
        with open(path, "rb") as fh:
            for offset in (0, max(0, size // 2 - 32768), max(0, size - 65536)):
                fh.seek(offset)
                h.update(fh.read(65536))
    except OSError:
        h.update(os.path.basename(path).encode())
    return h.hexdigest()[:16]


class OcrCache:
    """
    Sayfa-bazlı OCR önbelleği. Anahtar: file_hash + page + dpi + engine +
    languages + rotation_policy. Önbellekte varsa OCR yeniden çalıştırılmaz.

    Her dosya için tek bir JSON dosyası ({cache_dir}/{file_hash}.json) kullanılır;
    dosya işlenirken belleğe alınır, sonunda diske yazılır.
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._mem: Dict[str, Dict[str, Any]] = {}

    def _path(self, file_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{file_hash}.json")

    def _file_map(self, file_hash: str) -> Dict[str, Any]:
        if file_hash in self._mem:
            return self._mem[file_hash]
        data: Dict[str, Any] = {}
        path = self._path(file_hash)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OCR önbelleği okunamadı (%s): %s", file_hash, exc)
                data = {}
        self._mem[file_hash] = data
        return data

    @staticmethod
    def page_key(page: int, dpi: int, engine: str, languages, policy: str) -> str:
        langs = ",".join(languages) if not isinstance(languages, str) else languages
        return f"p{page}|dpi{dpi}|{engine}|{langs}|{policy}"

    def get(self, file_hash: str, key: str) -> Optional[Dict[str, Any]]:
        return self._file_map(file_hash).get(key)

    def put(self, file_hash: str, key: str, value: Dict[str, Any]) -> None:
        self._file_map(file_hash)[key] = value

    def flush(self, file_hash: str) -> None:
        """Bir dosyanın önbelleğini diske yazar."""
        try:
            with open(self._path(file_hash), "w", encoding="utf-8") as fh:
                json.dump(self._mem.get(file_hash, {}), fh, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR önbelleği yazılamadı (%s): %s", file_hash, exc)


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
    def g(key, default=0):
        return stats.get(key, default)
    failed = g("failed_pages", []) or []
    # Per-dosya raporu 'dominant_rotation' (açı), toplulaştırılmış rapor
    # 'dominant_rotation_files' (kaç dosyada dominant belirlendi) taşır.
    dom = g("dominant_rotation", None)
    dom_files = g("dominant_rotation_files", None)
    lines = [
        "=== INGESTION RAPORU ===",
        f"  Kaynak                : {g('source', '-')}",
        f"  Toplam dosya          : {g('total_files', 1)}",
        f"  Toplam sayfa          : {g('total_pages')}",
        f"  Metin katmanli sayfa  : {g('text_layer_pages')}",
        f"  OCR gereken sayfa     : {g('ocr_required_pages')}",
        f"  OCR sonucu kullanilan : {g('ocr_used_pages')}  (cache dahil)",
        f"  Gercek OCR calisan    : {g('ocr_executed_pages')}  (cache haric)",
        f"  Onbellek (cache) hit  : {g('cache_hit_pages')}",
        f"  OCR atlanan sayfa     : {g('ocr_skipped_pages')}",
        f"  Cizim atlanan (ad)    : {g('drawing_skipped_pages')}",
        f"  Gomulu cizim (dusuk metin): {g('embedded_drawing_pages')}",
        f"  Fallback yapilan sayfa: {g('fallback_pages')}",
        f"  Toplam OCR cagrisi    : {g('total_ocr_calls')}",
        f"  Toplam chunk          : {g('total_chunks')}",
        f"  Metin cikarma suresi  : {g('text_time')} sn",
        f"  OCR suresi            : {g('ocr_time')} sn",
        f"  Embedding suresi      : {g('embedding_time')} sn",
        f"  Chroma yazma suresi   : {g('chroma_time')} sn",
        f"  Basarisiz sayfa       : {len(failed)}" + (f" -> {failed[:20]}" if failed else ""),
    ]
    if dom is not None:
        lines.append(f"  Dominant rotasyon     : {dom} derece")
    if dom_files is not None:
        lines.append(f"  Dominant rotasyonlu dosya: {dom_files}")
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

    # Rotasyon denemesinde İLK denenen açı (0 veya dominant) bu kadar ANLAMLI
    # metin verdiyse diğer açıları DENEME (gereksiz 4x OCR'dan kaçınılır).
    ROT_FASTPATH_CHARS = 240
    # Dominant açı belge için DOĞRU bilindiğinden, az da olsa anlamlı metin
    # (meaningful_score) verdiyse ona güveniriz; sadece neredeyse boş/gürültüyse
    # diğer açılara düşeriz. Düşük eşik = bulk sayfalarda tek OCR çağrısı = hız.
    ROT_ACCEPT_MIN = 20
    # Bir sayfanın dominant rotasyon TESPİTİNE katkı sayılması için (güvenilir
    # sinyal) gereken en az ANLAMLI metin. Kabul eşiğinden yüksek tutulur.
    ROT_PREFERRED_MIN = 60

    # Dominant rotasyon tespiti yalnızca OCR gereken sayfa sayısı bu eşiği
    # aşarsa yapılır (az sayfalı dosyada probe maliyeti anlamsız).
    DOMINANT_MIN_PAGES = 8

    # Gömülü çizim/şema tespiti: OCR sonrası sayfa metni script-bağımsız kalite
    # ölçütünü (is_meaningful_text: MIN_CHUNK_LETTERS + MIN_ALPHA_RATIO) geçemezse
    # sayfa "manuel içine gömülü çizim" sayılır -> Chroma'ya EKLENMEZ, index'e yazılır.

    # Belge-geneli dominant rotasyon tespiti: ilk bu kadar GÜVENİLİR metinli sayfa
    # tüm açılarda taranır; sonra dominant açı belirlenir.
    ROT_PROBE_PAGES = 5   # En fazla bu kadar sayfa tam taranır.
    ROT_PROBE_MIN = 3     # Bu kadar prob sonrası net çoğunluk varsa erken karar.

    # --- Chunk kalite filtresi ---
    # Sembol/rakam ağırlıklı OCR gürültüsünü (şema/diyagram parçaları) eler;
    # gerçek metni (Türkçe/Korece dahil) korur. Harf oranı script-bağımsızdır
    # (isalpha CJK'yi de sayar). Çok düşük tutmak çöpü içeri alır (retrieval'i
    # bozar), çok yüksek tutmak sayı-ağırlıklı gerçek tabloları eler -> 0.45 denge.
    MIN_ALPHA_RATIO = 0.45        # Harf oranı bunun altındaysa (saf sembol/rakam) ele.
    MIN_CHUNK_LETTERS = 12        # Bir parçada en az bu kadar harf olmalı.

    SUPPORTED_PDF = (".pdf",)
    SUPPORTED_IMAGE = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ocr_languages: Optional[List[str]] = None,
        enable_pdf_ocr_fallback: bool = True,
        ocr_gpu: Optional[bool] = None,
        ingest_mode: str = DEFAULT_INGEST_MODE,
        ocr_dpi: Optional[int] = None,
        skip_drawings: Optional[bool] = None,
        min_chunk_chars: int = 60,
        try_rotations: bool = True,
        cache_dir: Optional[str] = None,
        force_reocr: bool = False,
        debug_dir: Optional[str] = None,
        drawings_dir: Optional[str] = None,
    ) -> None:
        """
        :param ingest_mode: 'fast' | 'balanced' | 'full' (varsayılan balanced).
                            ocr_dpi ve skip_drawings için varsayılanları belirler.
        :param ocr_dpi: Render DPI'ı; None ise moda göre (fast 150 / balanced 200 /
                        full 300). Açıkça verilirse modu override eder.
        :param skip_drawings: Çizim/şema dosyalarında OCR atlansın mı? None ise
                              moda göre (fast/balanced True, full False).
        :param cache_dir: Verilirse sayfa-bazlı OCR önbelleği kullanılır (2. çalıştırma
                          çok hızlı). force_reocr ile bypass edilebilir.
        :param force_reocr: True ise önbellek okunmaz (yine de güncellenir).
        :param debug_dir: Sayfa OCR metni/boş sayfa görüntüsü buraya yazılır.
        :param drawings_dir: Atlanan çizim sayfalarının görüntüleri buraya kaydedilir.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_languages = ocr_languages or ["tr", "en"]
        self.enable_pdf_ocr_fallback = enable_pdf_ocr_fallback
        self._ocr_gpu = ocr_gpu  # None = otomatik

        mode = ingest_mode if ingest_mode in INGEST_MODES else DEFAULT_INGEST_MODE
        self.ingest_mode = mode
        defaults = INGEST_MODES[mode]
        self.ocr_dpi = max(72, int(ocr_dpi if ocr_dpi is not None else defaults["ocr_dpi"]))
        self.skip_drawings = (defaults["skip_drawings"] if skip_drawings is None
                              else bool(skip_drawings))

        self.min_chunk_chars = max(1, int(min_chunk_chars))
        self.try_rotations = try_rotations
        self.force_reocr = force_reocr
        self.debug_dir = debug_dir
        self.drawings_dir = drawings_dir
        self._cache = OcrCache(cache_dir) if cache_dir else None
        self.ocr_engine = "easyocr"

        # OCR motoru ve metin parçalayıcı ilk kullanımda yüklenecek (lazy).
        self._ocr_reader = None
        self._splitter = None  # bkz. _get_splitter (langchain lazy)

    def _get_splitter(self):
        """RecursiveCharacterTextSplitter'ı ilk ihtiyaçta (lazy) oluşturur.

        langchain import'u ~8 sn sürdüğü için açılışta DEĞİL, ilk doküman
        işlemede (arka plan thread'i) yüklenir; böylece arayüz hızlı açılır.
        """
        if self._splitter is not None:
            return self._splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:  # Eski LangChain sürümleri için geri uyumluluk
            try:
                from langchain.text_splitter import RecursiveCharacterTextSplitter
            except ImportError as exc:
                raise ImportError(
                    "langchain-text-splitters bulunamadı. "
                    "Lütfen 'pip install langchain-text-splitters' çalıştırın."
                ) from exc
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )
        return self._splitter

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
        """PDF'ten metin çıkarır; metinsiz sayfalarda rotasyon-bilinçli OCR'a düşer.

        Mod/cache/çizim-atlama kuralları:
          * Metin katmanı olan sayfalar doğrudan kullanılır (OCR yok).
          * Metinsiz sayfalar OCR'a gider. Çizim/şema dosyalarında (skip_drawings)
            OCR ATLANIR; sayfa görüntüsü/metadata saklanır ama Chroma'ya eklenmez.
          * cache_dir varsa OCR sonucu sayfa-bazlı önbelleğe alınır (2. çalıştırma hızlı).
          * Dominant rotasyon yalnızca OCR gereken sayfa >= DOMINANT_MIN_PAGES ise.
        """
        if fitz is None:
            return ProcessResult(
                success=False,
                message="PyMuPDF (fitz) kurulu değil. 'pip install PyMuPDF' gerekli.",
            )

        source = os.path.basename(file_path)
        # Çizim tespiti tüm yol (klasör adları dahil) üzerinden; ör. 'Diagrams/...'
        normalized_path = file_path.replace("\\", "/")
        is_drawing = self.skip_drawings and is_drawing_file(normalized_path)
        # file_hash her zaman hesaplanır (cache + çizim görüntü klasörü için).
        file_hash = file_signature(file_path)
        rotation_policy = ("smart" if self.try_rotations else "fixed0")

        all_chunks: List[DocumentChunk] = []
        st = {
            "source": source, "total_pages": 0, "text_layer_pages": 0,
            "ocr_required_pages": 0, "ocr_used_pages": 0, "ocr_executed_pages": 0,
            "ocr_skipped_pages": 0, "drawing_skipped_pages": 0,
            "embedded_drawing_pages": 0, "cache_hit_pages": 0,
            "failed_pages": [], "total_chunks": 0, "total_chars": 0,
            "text_time": 0.0, "ocr_time": 0.0, "total_ocr_calls": 0,
            "fallback_pages": 0, "dominant_rotation": None, "is_drawing": is_drawing,
        }

        try:
            document = fitz.open(file_path)
        except Exception as exc:  # noqa: BLE001
            return ProcessResult(success=False, message=f"PDF açılamadı: {exc}")

        total_pages = len(document)
        st["total_pages"] = total_pages

        # Ön tarama: hangi sayfalar metin katmanlı, kaçı OCR gerektiriyor?
        page_has_text: List[bool] = []
        for pn in range(total_pages):
            tl = (document[pn].get_text() or "").strip()
            page_has_text.append(len(tl) >= self.MIN_TEXT_THRESHOLD)
        st["ocr_required_pages"] = sum(1 for h in page_has_text if not h)
        # Dominant rotasyon yalnızca yeterince OCR sayfası varsa anlamlı.
        use_dominant = (self.try_rotations
                        and st["ocr_required_pages"] >= self.DOMINANT_MIN_PAGES)

        probe_rotations: List[int] = []
        dominant_rotation: Optional[int] = None
        probe_attempts = 0        # dominant kurmak için yapılan tam-arama sayısı
        give_up_rotation = False  # dominant kurulamadı -> rotasyon aramasını bırak
        try:
            for page_number in range(total_pages):
                page = document[page_number]
                page_label = page_number + 1
                ocr_used = False
                rotation = 0
                rendered = None
                page_text = ""

                if page_has_text[page_number]:
                    t0 = time.perf_counter()
                    page_text = clean_ocr_text((page.get_text() or "").strip())
                    st["text_layer_pages"] += 1
                    st["text_time"] += time.perf_counter() - t0
                elif not self.enable_pdf_ocr_fallback:
                    page_text = ""
                elif is_drawing:
                    # Çizim/şema: OCR'ı atla; görüntüyü (file_hash klasöründe) sakla,
                    # metadata'yı drawings_index.jsonl'a yaz, Chroma'ya EKLEME.
                    st["drawing_skipped_pages"] += 1
                    st["ocr_skipped_pages"] += 1
                    image_path = ""
                    if self.drawings_dir and Image is not None:
                        try:
                            rendered = self._render_page_array(page)
                            ddir = os.path.join(self.drawings_dir, file_hash)
                            os.makedirs(ddir, exist_ok=True)
                            image_path = os.path.join(ddir, f"page_{page_label:03d}.png")
                            Image.fromarray(rendered).save(image_path)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("Çizim görüntüsü kaydedilemedi: %s", exc)
                            image_path = ""
                    self._append_drawings_index({
                        "source_file": source, "page": page_label,
                        "page_type": "drawing", "ocr_skipped": True,
                        "reason": "drawing_filename_match",
                        "image_path": image_path,
                    })
                    continue  # bu sayfadan chunk üretme
                else:
                    # --- OCR (önbellek + rotasyon) ---
                    cache_key = None
                    cached = None
                    if self._cache and not self.force_reocr:
                        cache_key = OcrCache.page_key(
                            page_label, self.ocr_dpi, self.ocr_engine,
                            self.ocr_languages, rotation_policy)
                        cached = self._cache.get(file_hash, cache_key)

                    if cached is not None:
                        page_text = cached.get("text", "")
                        rotation = int(cached.get("rotation", 0))
                        ocr_used = True
                        st["cache_hit_pages"] += 1
                        st["ocr_used_pages"] += 1  # OCR sonucu kullanıldı (cache'ten)
                        # Önbellekten gelen sayfa da dominant rotasyon kararına katkı verir.
                        if (use_dominant and dominant_rotation is None
                                and meaningful_score(page_text) >= self.ROT_PREFERRED_MIN):
                            probe_rotations.append(rotation)
                            if (len(probe_rotations) >= self.ROT_PROBE_PAGES
                                    or self._rotation_decided(probe_rotations, self.ROT_PROBE_MIN)):
                                dominant_rotation = Counter(probe_rotations).most_common(1)[0][0]
                                st["dominant_rotation"] = dominant_rotation
                                logger.info(
                                    "Dokuman dominant rotasyonu (cache): %d derece (problar: %s)",
                                    dominant_rotation, probe_rotations)
                    else:
                        # Rotasyon stratejisi:
                        #  * dominant kuruldu -> sadece dominant açı (fallback yok):
                        #    çizim sayfaları boşuna 4x taranmaz.
                        #  * dominant kurulamadı (give_up) -> sadece 0 derece.
                        #  * hâlâ probe -> tam arama (dominant bulmak için).
                        if use_dominant and dominant_rotation is not None:
                            pref, allow_fb = dominant_rotation, False
                        elif use_dominant and give_up_rotation:
                            pref, allow_fb = 0, False
                        else:
                            pref, allow_fb = (None if not use_dominant else dominant_rotation), True

                        t0 = time.perf_counter()
                        rendered = self._render_page_array(page)
                        best = self._ocr_image_best(
                            rendered, f"page_{page_label:03d}",
                            preferred=pref, allow_fallback=allow_fb)
                        st["ocr_time"] += time.perf_counter() - t0
                        page_text = clean_ocr_text(best["text"])
                        rotation = best["rotation"]
                        ocr_used = True
                        st["ocr_used_pages"] += 1       # OCR sonucu kullanıldı
                        st["ocr_executed_pages"] += 1   # gerçekten OCR çalıştı (cache değil)
                        st["total_ocr_calls"] += best.get("calls", 1)
                        if best.get("fell_back"):
                            st["fallback_pages"] += 1
                        logger.info(
                            "page=%d best_rotation=%d text_len=%d score=%.0f%s",
                            page_label, rotation, len(page_text), best["score"],
                            " (dominant)" if (use_dominant and dominant_rotation is not None)
                            else (" (give-up)" if give_up_rotation else " (probe)"),
                        )
                        # Önbelleğe yaz (force_reocr olsa da güncelle).
                        if self._cache:
                            ck = cache_key or OcrCache.page_key(
                                page_label, self.ocr_dpi, self.ocr_engine,
                                self.ocr_languages, rotation_policy)
                            self._cache.put(file_hash, ck,
                                            {"text": page_text, "rotation": rotation})
                        # Dominant rotasyon tespiti (güvenilir metinli sayfalarla).
                        if use_dominant and dominant_rotation is None and not give_up_rotation:
                            probe_attempts += 1
                            if meaningful_score(best["text"]) >= self.ROT_PREFERRED_MIN:
                                probe_rotations.append(rotation)
                                if (len(probe_rotations) >= self.ROT_PROBE_PAGES
                                        or self._rotation_decided(probe_rotations, self.ROT_PROBE_MIN)):
                                    dominant_rotation = Counter(probe_rotations).most_common(1)[0][0]
                                    st["dominant_rotation"] = dominant_rotation
                                    logger.info(
                                        "Dokuman dominant rotasyonu: %d derece (problar: %s)",
                                        dominant_rotation, probe_rotations)
                            # Yeterince denedik ama dominant kurulamadı: okunur metin
                            # yok -> kalan sayfalarda rotasyon aramasını bırak (hız).
                            if dominant_rotation is None and probe_attempts >= self.ROT_PROBE_PAGES:
                                give_up_rotation = True
                                logger.info(
                                    "Dominant kurulamadi (%d denemede okunur metin yok); "
                                    "kalan sayfalarda rotasyon aramasi durduruldu (%s).",
                                    probe_attempts, source)

                # OCR'lı bir sayfa en iyi açıda bile anlamlı metin veremiyorsa
                # "manuel içine gömülü çizim" say: Chroma'ya ekleme, index'e yaz.
                # (Metin katmanlı sayfalar bu kurala tabi değildir; korunur.)
                if ocr_used and not is_meaningful_text(
                        page_text, min_letters=self.MIN_CHUNK_LETTERS,
                        min_alpha_ratio=self.MIN_ALPHA_RATIO):
                    # Gömülü çizim/şema: "başarısız" değil, beklenen bir durum.
                    st["embedded_drawing_pages"] += 1
                    self._append_drawings_index({
                        "source_file": source, "page": page_label,
                        "page_type": "embedded_drawing", "ocr_skipped": False,
                        "reason": "low_text_after_ocr", "image_path": "",
                    })
                    self._write_page_debug(page_label, "", rendered)
                    continue

                # Debug çıktısı (OCR'lı/boş sayfalar için).
                self._write_page_debug(page_label, page_text, rendered)

                if page_text:
                    st["total_chars"] += len(page_text)
                    pieces = self._split_text(page_text)
                    if not pieces:
                        logger.warning(
                            "WARNING: OCR produced text but chunker produced 0 chunks "
                            "(page=%d, chars=%d).", page_label, len(page_text))
                        pieces = [page_text]
                    for idx, piece in enumerate(pieces):
                        all_chunks.append(DocumentChunk(
                            text=piece, source=source, page=page_label,
                            chunk_index=idx, ocr_used=ocr_used,
                            rotation=rotation, char_count=len(piece)))
                elif ocr_used:
                    st["failed_pages"].append(page_label)
        finally:
            document.close()
            if self._cache and file_hash:
                self._cache.flush(file_hash)

        st["total_chunks"] = len(all_chunks)
        st["text_time"] = round(st["text_time"], 2)
        st["ocr_time"] = round(st["ocr_time"], 2)

        if st["total_chars"] > 0 and not all_chunks:
            logger.warning(
                "WARNING: OCR produced text but chunker produced 0 chunks (%s).", source)

        if not all_chunks:
            if is_drawing:
                msg = (f"'{source}': çizim/şema dosyası — OCR atlandı "
                       f"({st['drawing_skipped_pages']} sayfa).")
                return ProcessResult(success=True, message=msg, stats=st)  # hata değil
            if st["total_chars"] == 0:
                msg = (f"'{source}': hiçbir sayfadan metin çıkarılamadı "
                       f"({len(st['failed_pages'])}/{total_pages} sayfa boş).")
            else:
                msg = f"'{source}': metin çıktı ama parça üretilemedi (chunker hatası)."
            return ProcessResult(success=False, message=msg, stats=st)

        msg = f"'{source}' işlendi: {len(all_chunks)} parça."
        if st["ocr_used_pages"]:
            msg += (f" ({st['ocr_used_pages']} OCR"
                    + (f", {st['cache_hit_pages']} önbellek" if st['cache_hit_pages'] else "")
                    + " sayfa.)")
        if st["failed_pages"]:
            msg += f" {len(st['failed_pages'])} sayfa boş."
        return ProcessResult(chunks=all_chunks, success=True, message=msg, stats=st)

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
                        preferred: Optional[int] = None,
                        allow_fallback: bool = True) -> Dict[str, Any]:
        """
        Bir görüntüyü OCR eder ve en çok anlamlı metin veren rotasyonu seçer.

        * try_rotations kapalıysa yalnızca 0 derece denenir.
        * preferred (dominant açı) verilmişse ÖNCE o denenir; yeterli metin
          (>= ROT_PREFERRED_MIN) verirse diğer açılar DENENMEZ (hız).
        * allow_fallback False ise (dominant kararlı / give-up) preferred açıda
          tek tarama yapılır, zayıf olsa bile diğer açılara GİDİLMEZ. Bu, çizim
          sayfalarının boşuna 4x taranmasını önler.
        * preferred yoksa 0,90,180,270 tam taranır (0 derece bol metin verirse
          erken durulur).
        """
        if not self.try_rotations:
            res = self._ocr_at_angle(arr, 0, label)
            res["calls"] = 1
            res["fell_back"] = False
            return res

        order = [0, 90, 180, 270]
        if preferred is not None:
            order = [preferred] + [a for a in order if a != preferred]

        best: Dict[str, Any] = {"text": "", "rotation": 0, "score": -1.0}
        calls = 0
        for i, angle in enumerate(order):
            cur = self._ocr_at_angle(arr, angle, label)
            calls += 1
            if cur["score"] > best["score"]:
                best = cur
            if i == 0:
                # İlk denenen açı (0 veya dominant) zaten güçlüyse: bitir.
                if meaningful_score(cur["text"]) >= self.ROT_FASTPATH_CHARS:
                    break
                # Dominant açı (doğru bilinen) az da olsa ANLAMLI metin verdiyse
                # güven; sadece neredeyse boş veya saf gürültüyse diğerlerine düş.
                if preferred is not None and meaningful_score(cur["text"]) >= self.ROT_ACCEPT_MIN:
                    break
                # Fallback kapalıysa (dominant kararlı / give-up): tek tarama yeter.
                if preferred is not None and not allow_fallback:
                    break
        best["calls"] = calls
        # Tercih edilen açı verildiyse ve birden fazla açı denendiyse: fallback oldu.
        best["fell_back"] = preferred is not None and calls > 1
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
        """
        Bir görsel dosyasını OCR ile okuyup parçalar üretir.

        Şema/diyagram görselleri (ad ile veya OCR sonrası anlamlı metin çok
        düşükse) "çizim" sayılır: ana retrieval'a (Chroma'ya) EKLENMEZ, yalnızca
        drawings_index'e yazılır. Böylece "@K-504 9 2 #X#" gibi parçalı OCR
        gürültüsü arama kalitesini bozmaz.
        """
        import numpy as np
        source = os.path.basename(file_path)
        normalized_path = file_path.replace("\\", "/")
        file_hash = file_signature(file_path)
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

        base = {"source": source, "total_pages": 1, "text_layer_pages": 0,
                "ocr_required_pages": 1, "ocr_used_pages": 1, "ocr_executed_pages": 1,
                "ocr_skipped_pages": 0, "drawing_skipped_pages": 0,
                "embedded_drawing_pages": 0, "cache_hit_pages": 0,
                "total_ocr_calls": best.get("calls", 1),
                "fallback_pages": 1 if best.get("fell_back") else 0,
                "ocr_time": 0.0, "text_time": 0.0, "dominant_rotation": None,
                "total_chars": len(text), "total_chunks": 0, "failed_pages": []}

        named_drawing = self.skip_drawings and is_drawing_file(normalized_path)
        low_text = not is_meaningful_text(
            text, min_letters=self.MIN_CHUNK_LETTERS, min_alpha_ratio=self.MIN_ALPHA_RATIO)

        if named_drawing or low_text:
            # Çizim/şema görseli: ana retrieval'a girmez; index'e ve (varsa) diske yaz.
            base["embedded_drawing_pages"] = 1
            image_path = ""
            if self.drawings_dir and Image is not None:
                try:
                    ddir = os.path.join(self.drawings_dir, file_hash)
                    os.makedirs(ddir, exist_ok=True)
                    image_path = os.path.join(ddir, "page_001.png")
                    Image.fromarray(arr).save(image_path)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Çizim görüntüsü kaydedilemedi: %s", exc)
                    image_path = ""
            self._append_drawings_index({
                "source_file": source, "page": 1, "page_type": "drawing_image",
                "ocr_skipped": named_drawing,
                "reason": "drawing_filename_match" if named_drawing else "low_text_after_ocr",
                "image_path": image_path,
            })
            return ProcessResult(
                success=True,
                message=f"'{source}': şema/çizim görseli — ana aramaya eklenmedi.",
                stats=base,
            )

        pieces = self._split_text(text) or [text]
        chunks = [
            DocumentChunk(text=piece, source=source, page=0, chunk_index=idx,
                          ocr_used=True, rotation=best["rotation"], char_count=len(piece))
            for idx, piece in enumerate(pieces)
        ]
        base["total_chunks"] = len(chunks)
        return ProcessResult(
            chunks=chunks, success=True,
            message=f"'{source}' OCR ile okundu: {len(chunks)} parça.", stats=base,
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
        raw = [c.strip() for c in self._get_splitter().split_text(text)]
        raw = [c for c in raw if c]
        kept = [c for c in raw if not self._is_garbage(c)]
        if not kept and raw:
            # Filtre her şeyi eledi: en az 3 harf içerenleri kurtar, o da yoksa ham.
            kept = [c for c in raw if sum(ch.isalpha() for ch in c) >= 3] or raw
        return kept

    def _is_garbage(self, text: str) -> bool:
        """
        Açıkça çöp parçaları eler: çok kısa, ya da sembol/rakam ağırlıklı
        (script-bağımsız harf oranı düşük) OCR gürültüsü. Gerçek metin (Türkçe/
        Korece dahil) korunur.
        """
        t = text.strip()
        if len(t) < self.min_chunk_chars:
            return True
        return not is_meaningful_text(
            t, min_letters=self.MIN_CHUNK_LETTERS, min_alpha_ratio=self.MIN_ALPHA_RATIO)

    # ------------------------------------------------------------------ #
    #  Çizim Atlama İndeksi
    # ------------------------------------------------------------------ #
    def _drawings_index_path(self) -> str:
        """drawings_index.jsonl için yazılacak yol (uygun bir taban klasör seçer)."""
        base = self.drawings_dir or self.debug_dir
        if not base and self._cache:
            base = os.path.dirname(os.path.normpath(self._cache.cache_dir))
        if not base:
            base = os.getcwd()
        return os.path.join(base, "drawings_index.jsonl")

    def _append_drawings_index(self, entry: Dict[str, Any]) -> None:
        """Atlanan bir çizim sayfasının metadata'sını drawings_index.jsonl'a ekler."""
        try:
            path = self._drawings_index_path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug("drawings_index yazılamadı: %s", exc)

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
