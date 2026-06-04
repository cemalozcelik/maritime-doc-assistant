# -*- coding: utf-8 -*-
"""
main.py
=======
Gemi Teknik Doküman Asistanı - Ana giriş noktası ve koordinatör.

Sorumluluklar:
    * CustomTkinter ana penceresini başlatmak.
    * Modülleri (DocumentProcessor, EmbeddingManager, LLMConnector) bir araya
      getirmek ve aralarındaki iş akışını yönetmek (orkestrasyon).
    * UI'ı dondurmamak için ağır işlemleri (OCR, embedding, LLM) arka plan
      thread'lerinde çalıştırmak ve sonuçları thread-safe biçimde UI'a aktarmak.

Mimari Not:
    Tkinter thread-safe değildir. Arka plan thread'leri UI'ı doğrudan
    güncellemez; bunun yerine bir kuyruğa (queue) iş bırakır ve ana thread
    'after' döngüsüyle bu kuyruğu işleyerek arayüzü günceller.
"""

from __future__ import annotations

import os
import sys
import time
import queue
import logging
import threading

# --- Loglama ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# Kullanılan embedding modeli. Çevrimdışı kontrolü, ağır kütüphaneler
# (transformers/huggingface_hub) import EDİLMEDEN ÖNCE yapılmalı; bu yüzden bu
# sabit ve yardımcılar dosyanın en başında tanımlanır.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def resource_path(relative: str) -> str:
    """
    Salt-okunur paketli kaynaklara (örn. gömülü embedding modeli) erişim.
    PyInstaller --onefile modunda dosyaları geçici '_MEIPASS' klasörüne açar.
    """
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, relative)


def enable_hf_offline_if_available(model_name: str) -> None:
    """
    Embedding modeli lokalde (projedeki 'models/' klasöründe veya Hugging Face
    cache'inde) zaten mevcutsa, Hugging Face'i TAMAMEN çevrimdışı moda alır.
    Böylece model yüklenirken internete (Hub) hiç gidilmez.

    Önemli: Model HENÜZ indirilmemişse offline'a ALINMAZ; ilk çalıştırmada
    internetten indirilebilsin.

    KRİTİK: Bu fonksiyon huggingface_hub'ı import ETMEZ ve ağır importlardan
    ÖNCE çağrılır. Çünkü huggingface_hub import anında HF_HUB_OFFLINE'ı okuyup
    sabitler; import'tan sonra set etmek etkisiz kalır.
    """
    def _go_offline(reason: str) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        logger.info("Hugging Face çevrimdışı moda alındı (%s).", reason)

    # 1) Projeye gömülü 'models/<isim>' klasörü var mı?
    safe_name = model_name.replace("/", "_")
    if os.path.isdir(resource_path(os.path.join("models", safe_name))):
        _go_offline("gömülü model")
        return

    # 2) Hugging Face cache'inde indirilmiş mi? (HF import etmeden yolu hesapla)
    hub_cache = os.environ.get("HF_HUB_CACHE")
    hf_home = os.environ.get("HF_HOME")
    if hub_cache:
        base = hub_cache
    elif hf_home:
        base = os.path.join(hf_home, "hub")
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    repo_dir = "models--" + model_name.replace("/", "--")
    snapshots = os.path.join(base, repo_dir, "snapshots")
    try:
        if os.path.isdir(snapshots) and os.listdir(snapshots):
            _go_offline("cache'te mevcut")
    except OSError as exc:
        logger.debug("HF cache kontrolü atlandı: %s", exc)


# KRİTİK SIRA: Ağır kütüphaneler import EDİLMEDEN ÖNCE çevrimdışı modu ayarla.
# (document_processor -> langchain -> huggingface_hub zinciri HF'i import eder.)
enable_hf_offline_if_available(EMBEDDING_MODEL)

# --- Ağır importlar (çevrimdışı mod ayarlandıktan SONRA) ---
from tkinter import filedialog, messagebox  # noqa: E402

import customtkinter as ctk  # noqa: E402

from ui_components import Sidebar, ChatArea  # noqa: E402
from document_processor import DocumentProcessor  # noqa: E402
from embedding_manager import EmbeddingManager  # noqa: E402
from llm_connector import LLMConnector  # noqa: E402

# Sürükle-bırak (opsiyonel). Kütüphane yoksa özellik sessizce devre dışı kalır;
# dosya/klasör seçme butonları her durumda çalışmaya devam eder.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # noqa: E402
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False


# ---------------------------------------------------------------------------
#  Yol Yardımcıları (PyInstaller uyumlu)
# ---------------------------------------------------------------------------
def writable_data_dir() -> str:
    """
    Yazılabilir kalıcı veri klasörü (vektör veritabanı için).
    .exe'nin yanındaki 'data' klasörünü kullanır; oluşturulamazsa kullanıcı
    profilindeki bir klasöre düşer. _MEIPASS yazılabilir OLMADIĞI için asla
    oraya yazmayız.
    """
    if getattr(sys, "frozen", False):
        # Paketlenmiş .exe -> exe'nin bulunduğu dizin.
        base = os.path.dirname(sys.executable)
    else:
        # Geliştirme ortamı -> proje klasörü.
        base = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base, "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
        # Yazma testi.
        test = os.path.join(data_dir, ".write_test")
        with open(test, "w") as fh:
            fh.write("ok")
        os.remove(test)
        return data_dir
    except Exception:  # noqa: BLE001
        # Yedek: kullanıcı profili.
        fallback = os.path.join(
            os.path.expanduser("~"), ".gemi_asistani", "data"
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


def local_embedding_model_path(model_name: str) -> str:
    """
    Embedding modeli için önce paketli/lokal klasörü, yoksa model adını döndürür.
    Paketleme sırasında modeli 'models/<isim>' altına koyabilirsiniz (bkz. README).
    """
    safe_name = model_name.replace("/", "_")
    candidate = resource_path(os.path.join("models", safe_name))
    if os.path.isdir(candidate):
        logger.info("Lokal embedding modeli bulundu: %s", candidate)
        return candidate
    # Lokal yoksa model adı: ilk çalışmada indirilir, sonraki çalışmalarda cache'ten.
    return model_name


# ---------------------------------------------------------------------------
#  Ana Uygulama
# ---------------------------------------------------------------------------
# Sürükle-bırak metotları (drop_target_register, dnd_bind) TkinterDnD'nin
# DnDWrapper mixin'inden gelir. Kütüphane varsa onu da temel sınıf olarak ekleriz;
# yoksa yalnızca ctk.CTk'den miras alınır (özellik sessizce kapalı kalır).
if _DND_AVAILABLE:
    _APP_BASES = (ctk.CTk, TkinterDnD.DnDWrapper)
else:
    _APP_BASES = (ctk.CTk,)


class GemiAsistaniApp(*_APP_BASES):
    """Uygulamanın ana penceresi ve koordinatörü."""

    EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
    TOP_K = 8  # Her soruda kaç bağlam parçası getirilsin (eşik elemesinden sonra).

    def __init__(self) -> None:
        super().__init__()

        self.title("Gemi Teknik Doküman Asistanı")
        self.geometry("1100x720")
        self.minsize(900, 600)

        # (Çevrimdışı mod, modül yüklenirken ağır importlardan önce ayarlandı.)
        # --- İş modüllerini hazırla (modeller lazy yüklenir) ---
        self.processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.embedder = EmbeddingManager(
            model_name_or_path=local_embedding_model_path(self.EMBEDDING_MODEL),
            persist_directory=os.path.join(writable_data_dir(), "vector_store"),
            collection_name="gemi_dokumanlari",
        )
        self.llm = LLMConnector()

        # --- Thread <-> UI iletişimi için kuyruk ---
        self._ui_queue: "queue.Queue" = queue.Queue()
        self._busy = False  # Aynı anda tek ağır iş.

        # --- Arayüz yerleşimi ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(
            self,
            on_provider_change=self._on_provider_change,
            on_upload=self._on_upload,
            on_upload_folder=self._on_upload_folder,
            on_refresh_ollama=self._on_refresh_ollama,
            on_clear_db=self._on_clear_db,
            dnd_available=_DND_AVAILABLE,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.chat = ChatArea(self, on_send=self._on_send)
        self.chat.grid(row=0, column=1, sticky="nsew")

        # Sürükle-bırak desteğini etkinleştir (kütüphane varsa).
        self._setup_dnd()

        # UI kuyruğunu işlemeye başla.
        self.after(100, self._process_ui_queue)

        # Açılışta arka planda hazırlık (model + DB ısındırma).
        self._run_in_background(self._warmup_task, on_done=self._warmup_done)

        # İlk durum bilgisi.
        self.sidebar.set_documents(self.embedder.list_sources())

        # Pencere kapatma olayı.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================== #
    #  Arka Plan İş Altyapısı
    # ================================================================== #
    def _run_in_background(self, target, on_done=None, *args, **kwargs) -> None:
        """
        Verilen fonksiyonu ayrı bir thread'de çalıştırır. Fonksiyonun dönüşü
        (veya hatası) UI kuyruğu üzerinden 'on_done(result, error)' ile ana
        thread'e iletilir.
        """
        def worker():
            try:
                result = target(*args, **kwargs)
                self._ui_queue.put(("done", on_done, result, None))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Arka plan işi hatası")
                self._ui_queue.put(("done", on_done, None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _post_ui(self, func) -> None:
        """Arka plan thread'inin UI'da çalıştırmak istediği kısa işleri kuyruğa koyar."""
        self._ui_queue.put(("call", func, None, None))

    def _process_ui_queue(self) -> None:
        """Ana thread: kuyruktaki UI işlerini güvenle uygular."""
        try:
            while True:
                kind, func, result, error = self._ui_queue.get_nowait()
                if kind == "call" and callable(func):
                    func()
                elif kind == "done" and callable(func):
                    func(result, error)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_ui_queue)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        """Meşguliyet durumunu ayarlar; ilgili kontrolleri kilitler/açar."""
        self._busy = busy
        self.sidebar.set_controls_enabled(not busy)
        self.chat.set_input_enabled(not busy)
        if status:
            self.sidebar.set_status(status, "orange" if busy else "lightgreen")

    # ================================================================== #
    #  Açılış Hazırlığı
    # ================================================================== #
    def _warmup_task(self):
        """Arka plan: embedding modelini ve veritabanını ısındırır."""
        self._post_ui(lambda: self.sidebar.set_status("Model yükleniyor...", "orange"))
        self.embedder.warm_up()
        return True

    def _warmup_done(self, result, error) -> None:
        if error:
            self.sidebar.set_status("Model yüklenemedi!", "red")
            messagebox.showwarning(
                "Uyarı",
                "Embedding modeli yüklenemedi. İnternet yoksa modelin lokalde "
                f"mevcut olduğundan emin olun.\n\nDetay: {error}",
            )
        else:
            count = self.embedder.get_document_count()
            self.sidebar.set_status(f"Hazır ({count} parça)", "lightgreen")
        # Açılışta Ollama'yı sessizce yokla.
        self._on_refresh_ollama(silent=True)

    # ================================================================== #
    #  Olay İşleyiciler (UI Callback'leri)
    # ================================================================== #
    def _on_provider_change(self, value: str) -> None:
        """Model sağlayıcı değiştiğinde durum çubuğunu bilgilendirir."""
        # Sidebar kendi __init__'i sırasında bu callback'i tetikleyebilir; o anda
        # 'self.sidebar' henüz atanmamış olur. Hazır değilse sessizce çık.
        if not hasattr(self, "sidebar"):
            return
        if value.startswith("Ollama"):
            self.sidebar.set_status("Ollama seçildi (çevrimdışı)", "gray70")
        else:
            self.sidebar.set_status("Gemini seçildi (çevrimiçi)", "gray70")

    def _on_refresh_ollama(self, silent: bool = False) -> None:
        """Ollama sunucusunu kontrol edip mevcut modelleri listeler."""
        def task():
            if not LLMConnector.check_ollama():
                return []
            return LLMConnector.get_ollama_models()

        def done(models, error):
            if error or not models:
                self.sidebar.set_ollama_models([])
                if not silent:
                    messagebox.showinfo(
                        "Ollama",
                        "Ollama sunucusu bulunamadı veya hiç model yok.\n\n"
                        "1) 'ollama serve' çalışıyor mu?\n"
                        "2) 'ollama pull llama3' ile model indirdiniz mi?",
                    )
            else:
                self.sidebar.set_ollama_models(models)

        self._run_in_background(task, on_done=done)

    def _on_upload(self) -> None:
        """Tek tek dosya seçtirip işler."""
        if self._busy:
            return
        paths = filedialog.askopenfilenames(
            title="Doküman veya görsel seçin",
            filetypes=[
                ("Desteklenen dosyalar", "*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("PDF dosyaları", "*.pdf"),
                ("Görseller", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                ("Tüm dosyalar", "*.*"),
            ],
        )
        if paths:
            self._process_paths(list(paths))

    def _on_upload_folder(self) -> None:
        """Bir klasör seçtirip içindeki ve alt klasörlerdeki tüm desteklenen dosyaları işler."""
        if self._busy:
            return
        folder = filedialog.askdirectory(title="İçinde doküman bulunan klasörü seçin")
        if not folder:
            return
        files = self.processor.collect_supported_files(folder)
        if not files:
            messagebox.showinfo(
                "Boş Klasör",
                "Seçilen klasörde (ve alt klasörlerinde) desteklenen dosya "
                "(PDF/görsel) bulunamadı.",
            )
            return
        self._process_paths(files)

    def _expand_paths(self, paths: List[str]) -> List[str]:
        """Yol listesindeki klasörleri içindeki desteklenen dosyalarla genişletir."""
        expanded: List[str] = []
        for path in paths:
            if os.path.isdir(path):
                expanded.extend(self.processor.collect_supported_files(path))
            elif os.path.isfile(path):
                expanded.append(path)
        return expanded

    def _process_paths(self, paths: List[str]) -> None:
        """Verilen dosya yollarını arka planda işleyip veritabanına ekler (ortak akış)."""
        if self._busy or not paths:
            return

        self._set_busy(True, "Dokümanlar işleniyor...")
        total = len(paths)

        def task():
            total_chunks = 0
            skipped = 0
            messages = []
            # Veritabanında zaten kayıtlı kaynak adları (dosya adı bazında).
            # Aynı klasörü tekrar seçince mevcut dosyalar yeniden işlenmesin.
            existing = set(self.embedder.list_sources())
            for index, path in enumerate(paths, start=1):
                name = os.path.basename(path)
                # Zaten yüklüyse (veya aynı seçimde mükerrer geldiyse) atla.
                if name in existing:
                    skipped += 1
                    messages.append(f"[ATLANDI] {name}: zaten yüklü")
                    self._post_ui(
                        lambda n=name, i=index: self.sidebar.set_status(
                            f"Atlanıyor ({i}/{total}): {n} (zaten yüklü)", "orange"
                        )
                    )
                    continue
                self._post_ui(
                    lambda n=name, i=index: self.sidebar.set_status(
                        f"İşleniyor ({i}/{total}): {n}", "orange"
                    )
                )
                result = self.processor.process_file(path)
                if result.success and result.chunks:
                    added = self.embedder.add_chunks(result.chunks)
                    total_chunks += added
                    # Bu oturumda işlendi; aynı ad tekrar gelirse bir daha ekleme.
                    existing.add(name)
                    messages.append(f"[OK] {result.message}")
                else:
                    messages.append(f"[HATA] {result.message}")
            return total_chunks, skipped, messages

        def done(result, error):
            self._set_busy(False)
            if error:
                self.sidebar.set_status("İşleme hatası!", "red")
                messagebox.showerror("Hata", f"Doküman işlenirken hata oluştu:\n{error}")
                return
            _total_chunks, skipped, messages = result
            self.sidebar.set_documents(self.embedder.list_sources())
            count = self.embedder.get_document_count()
            self.sidebar.set_status(f"Hazır ({count} parça)", "lightgreen")
            # Çok dosyada sohbeti boğmamak için listeyi kısalt.
            if len(messages) > 15:
                shown = messages[:15] + [f"... ve {len(messages) - 15} dosya daha"]
            else:
                shown = messages
            ozet = "Yükleme tamamlandı"
            if skipped:
                ozet += f" ({skipped} dosya zaten yüklüydü, atlandı)"
            self.chat.add_message(
                "Sistem", ozet + ":\n" + "\n".join(shown), is_user=False
            )

        self._run_in_background(task, on_done=done)

    # ------------------------------------------------------------------ #
    #  Sürükle-Bırak
    # ------------------------------------------------------------------ #
    def _setup_dnd(self) -> None:
        """Pencereye dosya/klasör sürükle-bırak desteği ekler (kütüphane varsa)."""
        if not _DND_AVAILABLE:
            logger.info("tkinterdnd2 yok; sürükle-bırak devre dışı.")
            return
        try:
            # CustomTkinter'ın oluşturduğu mevcut Tk yorumlayıcısına tkdnd'yi yükle.
            self.TkdndVersion = TkinterDnD._require(self)
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
            logger.info("Sürükle-bırak etkin.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sürükle-bırak etkinleştirilemedi: %s", exc)

    def _on_drop(self, event) -> None:
        """Pencereye bırakılan dosya/klasörleri işler."""
        if self._busy:
            self.chat.add_message(
                "Sistem", "Şu an meşgulüm; işlem bitince tekrar deneyin.", is_user=False
            )
            return
        try:
            # tk.splitlist; boşluk içeren ve {} ile sarmalanmış yolları doğru ayırır.
            raw_paths = list(self.tk.splitlist(event.data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sürüklenen veriler ayrıştırılamadı: %s", exc)
            return
        expanded = self._expand_paths(raw_paths)
        if not expanded:
            messagebox.showinfo(
                "Desteklenmeyen",
                "Bırakılan öğelerde desteklenen dosya (PDF/görsel) bulunamadı.",
            )
            return
        self._process_paths(expanded)

    def _on_clear_db(self) -> None:
        """Onay alıp veritabanını temizler."""
        if self._busy:
            return
        if not messagebox.askyesno(
            "Onay",
            "Tüm yüklü dokümanlar veritabanından silinecek. Emin misiniz?",
        ):
            return

        self._set_busy(True, "Veritabanı temizleniyor...")

        def task():
            self.embedder.clear()
            return True

        def done(result, error):
            self._set_busy(False)
            if error:
                messagebox.showerror("Hata", f"Veritabanı temizlenemedi:\n{error}")
                self.sidebar.set_status("Temizleme hatası!", "red")
            else:
                self.sidebar.set_documents([])
                self.sidebar.set_status("Hazır (0 parça)", "lightgreen")
                self.chat.add_message("Sistem", "Veritabanı temizlendi.", is_user=False)

        self._run_in_background(task, on_done=done)

    # ================================================================== #
    #  Soru-Cevap Akışı (RAG)
    # ================================================================== #
    def _on_send(self, question: str) -> None:
        """Kullanıcı sorusunu alır, sağlayıcıyı yapılandırır ve cevabı üretir."""
        if self._busy:
            return

        # 1) Sağlayıcıyı UI seçimine göre ayarla.
        if not self._configure_provider():
            return

        # 2) Kullanıcı mesajını göster ve "Yazıyor..." balonu ekle.
        self.chat.add_message("Siz", question, is_user=True)
        thinking_label = self.chat.add_message("Asistan", "Düşünüyor...", is_user=False)

        self._set_busy(True, "Cevap üretiliyor...")

        def task():
            from perf_monitor import ResourceSampler, format_perf_block

            # Soru sorulduğundan cevap gelene kadarki tüm süreci ölç.
            with ResourceSampler() as sampler:
                t0 = time.perf_counter()
                # 3) İlgili bağlamı getir (retrieval).
                contexts = self.embedder.similarity_search(question, k=self.TOP_K)
                t1 = time.perf_counter()
                # 4) LLM'den cevap üret (generation).
                response = self.llm.ask(question, contexts)
                t2 = time.perf_counter()

            timings = {
                "retrieval_s": round(t1 - t0, 2),
                "generation_s": round(t2 - t1, 2),
                "total_s": round(t2 - t0, 2),
            }
            perf_text = format_perf_block(timings, response.meta, sampler.summary())
            return response, contexts, perf_text

        def done(result, error):
            self._set_busy(False)
            self.sidebar.set_status(
                f"Hazır ({self.embedder.get_document_count()} parça)", "lightgreen"
            )
            if error:
                self.chat.update_message(thinking_label, f"Hata: {error}")
                return

            response, contexts, perf_text = result
            if not response.success:
                self.chat.update_message(thinking_label, f"Hata: {response.error}")
                return

            # Kaynak dipnotu ekle.
            answer = response.text
            if contexts:
                sources = sorted({
                    f"{c.source}" + (f" (s.{c.page})" if c.page else "")
                    for c in contexts
                })
                answer += "\n\nKaynaklar: " + ", ".join(sources)
            # Performans ölçümünü en sona ekle.
            answer += "\n\n" + perf_text
            self.chat.update_message(thinking_label, answer)

        self._run_in_background(task, on_done=done)

    def _configure_provider(self) -> bool:
        """UI seçimine göre LLM sağlayıcıyı kurar. Başarılıysa True döner."""
        provider = self.sidebar.get_provider()

        if provider.startswith("Gemini"):
            api_key = self.sidebar.get_api_key()
            if not api_key:
                messagebox.showwarning(
                    "Eksik Bilgi",
                    "Gemini kullanmak için API anahtarı girmelisiniz.\n"
                    "İnternet yoksa 'Ollama (Çevrimdışı)' seçeneğini kullanın.",
                )
                return False
            if not LLMConnector.check_ollama() and not self._has_internet():
                # Gemini seçili ama internet yok -> kullanıcıyı uyar.
                messagebox.showwarning(
                    "İnternet Yok",
                    "Gemini için internet bağlantısı gerekir. Çevrimdışı çalışmak "
                    "için 'Ollama (Çevrimdışı)' seçeneğine geçin.",
                )
                return False
            self.llm.use_gemini(api_key=api_key)
            return True

        # Ollama
        model = self.sidebar.get_ollama_model()
        if not model or model.startswith("("):
            messagebox.showwarning(
                "Model Yok",
                "Geçerli bir Ollama modeli bulunamadı.\n"
                "'ollama pull llama3' ile model indirip 'Yenile'ye basın.",
            )
            return False
        if not LLMConnector.check_ollama():
            messagebox.showerror(
                "Ollama Çalışmıyor",
                "Ollama sunucusuna bağlanılamadı. Lütfen 'ollama serve' çalıştırın.",
            )
            return False
        self.llm.use_ollama(model_name=model)
        return True

    @staticmethod
    def _has_internet() -> bool:
        """Basit internet kontrolü."""
        try:
            import socket
            socket.setdefaulttimeout(3)
            socket.create_connection(("8.8.8.8", 53))
            return True
        except OSError:
            return False

    # ================================================================== #
    #  Kapanış
    # ================================================================== #
    def _on_close(self) -> None:
        if self._busy:
            if not messagebox.askyesno(
                "Çıkış", "Bir işlem sürüyor. Yine de çıkmak istiyor musunuz?"
            ):
                return
        self.destroy()


def main() -> None:
    """Uygulamayı başlatır."""
    try:
        app = GemiAsistaniApp()
        app.mainloop()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Uygulama başlatılamadı")
        try:
            messagebox.showerror("Kritik Hata", f"Uygulama başlatılamadı:\n{exc}")
        except Exception:  # noqa: BLE001
            print(f"Kritik Hata: {exc}")


if __name__ == "__main__":
    main()
