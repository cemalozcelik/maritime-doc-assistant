# -*- coding: utf-8 -*-
"""
main.py
=======
Gemi Teknik Doküman Asistanı - Ana giriş noktası ve koordinatör.

Sorumluluklar:
    * CustomTkinter ana penceresini başlatmak.
    * Modülleri (DocumentProcessor, EmbeddingManager, LLMConnector, ModelManager,
      ChatStore) bir araya getirmek ve aralarındaki iş akışını yönetmek.
    * UI'ı dondurmamak için ağır işlemleri (OCR, embedding, LLM, model indirme)
      arka plan thread'lerinde çalıştırmak ve sonuçları thread-safe biçimde aktarmak.

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
from typing import List, Optional

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

from ui_components import (  # noqa: E402
    ChatHistoryRail, ChatArea, ModelsView, SettingsView,
)
from document_processor import DocumentProcessor  # noqa: E402
from embedding_manager import EmbeddingManager  # noqa: E402
from llm_connector import LLMConnector  # noqa: E402
from model_manager import ModelManager  # noqa: E402
from chat_store import ChatStore, ROLE_USER, ROLE_ASSISTANT  # noqa: E402

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
    Yazılabilir kalıcı veri klasörü (vektör veritabanı, modeller, sohbetler için).
    .exe'nin yanındaki 'data' klasörünü kullanır; oluşturulamazsa kullanıcı
    profilindeki bir klasöre düşer. _MEIPASS yazılabilir OLMADIĞI için asla
    oraya yazmayız.
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(base, "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
        test = os.path.join(data_dir, ".write_test")
        with open(test, "w") as fh:
            fh.write("ok")
        os.remove(test)
        return data_dir
    except Exception:  # noqa: BLE001
        fallback = os.path.join(os.path.expanduser("~"), ".gemi_asistani", "data")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def local_embedding_model_path(model_name: str) -> str:
    """
    Embedding modeli için önce paketli/lokal klasörü, yoksa model adını döndürür.
    """
    safe_name = model_name.replace("/", "_")
    candidate = resource_path(os.path.join("models", safe_name))
    if os.path.isdir(candidate):
        logger.info("Lokal embedding modeli bulundu: %s", candidate)
        return candidate
    return model_name


# ---------------------------------------------------------------------------
#  Ana Uygulama
# ---------------------------------------------------------------------------
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
        self.geometry("1200x760")
        self.minsize(1000, 640)

        data_dir = writable_data_dir()

        # --- İş modüllerini hazırla (modeller lazy yüklenir) ---
        self.processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        self.embedder = EmbeddingManager(
            model_name_or_path=local_embedding_model_path(self.EMBEDDING_MODEL),
            persist_directory=os.path.join(data_dir, "vector_store"),
            collection_name="gemi_dokumanlari",
        )
        self.llm = LLMConnector()
        self.model_manager = ModelManager(
            models_dir=os.path.join(data_dir, "models_gguf")
        )
        self.chat_store = ChatStore(base_dir=os.path.join(data_dir, "chats"))

        # --- Thread <-> UI iletişimi için kuyruk ---
        self._ui_queue: "queue.Queue" = queue.Queue()
        self._busy = False  # Aynı anda tek ağır iş.
        self._download_cancel: Optional[threading.Event] = None

        # --- Oturum durumu ---
        self.current_session = self.chat_store.create(
            provider=LLMConnector.PROVIDER_LOCAL
        )  # Bellekte; ilk mesajda diske yazılır.

        # --- Arayüz yerleşimi ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.rail = ChatHistoryRail(
            self,
            on_new_chat=self._on_new_chat,
            on_select_chat=self._on_select_chat,
            on_delete_chat=self._on_delete_chat,
            on_show_view=self._show_view,
        )
        self.rail.grid(row=0, column=0, sticky="nsew")

        # İçerik konteyneri: üç görünüm aynı hücrede; aktif olan öne alınır.
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.chat = ChatArea(self.content, on_send=self._on_send)
        self.models_view = ModelsView(
            self.content,
            on_provider_change=self._on_provider_change,
            on_select_model=self._on_select_model,
            on_download=self._on_download_model,
            on_cancel_download=self._on_cancel_download,
            on_delete_model=self._on_delete_model,
        )
        self.settings_view = SettingsView(
            self.content,
            on_upload=self._on_upload,
            on_upload_folder=self._on_upload_folder,
            on_clear_db=self._on_clear_db,
            dnd_available=_DND_AVAILABLE,
        )
        self._views = {
            "chat": self.chat,
            "models": self.models_view,
            "settings": self.settings_view,
        }
        for view in self._views.values():
            view.grid(row=0, column=0, sticky="nsew")
        self._show_view("chat")

        # Sürükle-bırak desteğini etkinleştir (kütüphane varsa).
        self._setup_dnd()

        # UI kuyruğunu işlemeye başla.
        self.after(100, self._process_ui_queue)

        # Açılışta arka planda hazırlık (model + DB ısındırma).
        self._run_in_background(self._warmup_task, on_done=self._warmup_done)

        # İlk durum bilgisi.
        self.settings_view.set_documents(self.embedder.list_sources())
        self._refresh_model_lists()
        self._refresh_sessions()

        # Pencere kapatma olayı.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================== #
    #  Görünüm Yönetimi
    # ================================================================== #
    def _show_view(self, name: str) -> None:
        """İlgili görünümü öne alır ve ray düğmesini vurgular."""
        view = self._views.get(name)
        if view is None:
            return
        view.tkraise()
        self.rail.set_active_view(name)

    # ================================================================== #
    #  Arka Plan İş Altyapısı
    # ================================================================== #
    def _run_in_background(self, target, on_done=None, *args, **kwargs) -> None:
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
        self.rail.set_controls_enabled(not busy)
        self.settings_view.set_controls_enabled(not busy)
        self.models_view.set_controls_enabled(not busy)
        self.chat.set_input_enabled(not busy)
        if status:
            self.rail.set_status(status, "orange" if busy else "lightgreen")

    # ================================================================== #
    #  Açılış Hazırlığı
    # ================================================================== #
    def _warmup_task(self):
        """Arka plan: embedding modelini ve veritabanını ısındırır."""
        self._post_ui(lambda: self.rail.set_status("Model yükleniyor...", "orange"))
        self.embedder.warm_up()
        return True

    def _warmup_done(self, result, error) -> None:
        if error:
            self.rail.set_status("Model yüklenemedi!", "red")
            messagebox.showwarning(
                "Uyarı",
                "Embedding modeli yüklenemedi. İnternet yoksa modelin lokalde "
                f"mevcut olduğundan emin olun.\n\nDetay: {error}",
            )
        else:
            count = self.embedder.get_document_count()
            self.rail.set_status(f"Hazır ({count} parça)", "lightgreen")

    # ================================================================== #
    #  Sohbet Oturumları (Kalıcı Geçmiş)
    # ================================================================== #
    def _refresh_sessions(self) -> None:
        """Ray'deki geçmiş sohbet listesini diskteki oturumlarla günceller."""
        sessions = self.chat_store.list_sessions()
        self.rail.set_sessions(sessions, active_id=self.current_session.get("id"))

    def _persist_current(self) -> None:
        """Aktif oturumu (mesaj içeriyorsa) diske yazar."""
        if not ChatStore.is_empty(self.current_session):
            self.chat_store.save(self.current_session)

    def _on_new_chat(self) -> None:
        """Yeni bir boş sohbet başlatır (mevcut boşsa yeniden kullanır)."""
        if self._busy:
            return
        if ChatStore.is_empty(self.current_session):
            # Zaten boş bir oturumdayız; sadece sohbeti göster.
            self.chat.show_greeting()
            self._show_view("chat")
            return
        self._persist_current()
        self.current_session = self.chat_store.create(
            provider=self.models_view.get_provider()
        )
        self.chat.show_greeting()
        self._show_view("chat")
        self._refresh_sessions()

    def _on_select_chat(self, session_id: str) -> None:
        """Geçmiş bir sohbeti yükler."""
        if self._busy:
            return
        if session_id == self.current_session.get("id"):
            self._show_view("chat")
            return
        self._persist_current()
        loaded = self.chat_store.load(session_id)
        if not loaded:
            self._refresh_sessions()
            return
        self.current_session = loaded
        self.chat.load_messages(loaded.get("messages") or [])
        self._show_view("chat")
        self._refresh_sessions()

    def _on_delete_chat(self, session_id: str) -> None:
        """Bir sohbeti siler; aktif sohbet silindiyse yeni bir boş sohbete geçer."""
        if self._busy:
            return
        self.chat_store.delete(session_id)
        if session_id == self.current_session.get("id"):
            self.current_session = self.chat_store.create(
                provider=self.models_view.get_provider()
            )
            self.chat.show_greeting()
        self._refresh_sessions()

    # ================================================================== #
    #  Sağlayıcı / Model Yönetimi
    # ================================================================== #
    def _on_provider_change(self, value: str) -> None:
        """Model sağlayıcı değiştiğinde durum çubuğunu bilgilendirir."""
        if not hasattr(self, "rail"):
            return
        if value.startswith("Yerel"):
            self.rail.set_status("Yerel model seçildi (çevrimdışı)", "gray70")
        else:
            self.rail.set_status("Gemini seçildi (çevrimiçi)", "gray70")

    def _on_select_model(self, filename: str) -> None:
        """Aktif yerel modeli ayarlar (henüz yüklemez; ilk soruda yüklenir)."""
        self.rail.set_status(f"Aktif model: {filename}", "gray70")

    def _refresh_model_lists(self) -> None:
        """Modeller görünümündeki indirilmiş ve indirilebilir listeleri tazeler."""
        downloaded = self.model_manager.list_downloaded()
        active = self.models_view.get_active_model()
        self.models_view.set_downloaded_models(downloaded, active=active)
        self.models_view.set_curated(self.model_manager.curated(), downloaded)

    def _on_download_model(self, model: dict) -> None:
        """Bir GGUF modelini arka planda indirir (ilerleme + iptal)."""
        if self._busy:
            return
        self._download_cancel = threading.Event()
        self.models_view.show_progress(True)
        self.models_view.set_progress(
            0, model.get("approx_mb", 0) * 1024 * 1024,
            f"İndiriliyor: {model['label']}",
        )
        self._set_busy(True, f"İndiriliyor: {model['filename']}")

        cancel = self._download_cancel
        # İlerlemeyi her ~16 MB'da bir UI'a gönder (kuyruğu boğmamak için).
        last = {"sent": 0}

        def task():
            def cb(done, total):
                if done - last["sent"] >= 16 * 1024 * 1024 or done >= total:
                    last["sent"] = done
                    self._post_ui(lambda d=done, t=total: self.models_view.set_progress(d, t))
            return self.model_manager.download(
                model["repo_id"], model["filename"],
                progress_cb=cb, cancel_event=cancel,
            )

        def done(result, error):
            self._set_busy(False)
            self.models_view.show_progress(False)
            if error:
                msg = str(error)
                if "iptal" in msg.lower():
                    self.rail.set_status("İndirme iptal edildi", "orange")
                else:
                    self.rail.set_status("İndirme hatası!", "red")
                    messagebox.showerror(
                        "İndirme Hatası",
                        f"Model indirilemedi:\n{error}\n\n"
                        "İnternet bağlantınızı kontrol edip tekrar deneyin.",
                    )
            else:
                self.rail.set_status(f"İndirildi: {model['filename']}", "lightgreen")
            # Listeyi her durumda tazele ve indirileni aktif yap.
            self._refresh_model_lists()
            if not error:
                self.models_view.set_downloaded_models(
                    self.model_manager.list_downloaded(), active=model["filename"]
                )

        self._run_in_background(task, on_done=done)

    def _on_cancel_download(self) -> None:
        """Süren indirmeyi iptal eder."""
        if self._download_cancel is not None:
            self._download_cancel.set()
            self.rail.set_status("İndirme iptal ediliyor...", "orange")

    def _on_delete_model(self, filename: str) -> None:
        """İndirilmiş bir modeli siler."""
        if self._busy:
            return
        if not messagebox.askyesno(
            "Onay", f"'{filename}' silinecek. Emin misiniz?"
        ):
            return
        self.model_manager.delete(filename)
        self.rail.set_status(f"Silindi: {filename}", "gray70")
        self._refresh_model_lists()

    # ================================================================== #
    #  Doküman Yükleme
    # ================================================================== #
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
        """Bir klasör seçtirip içindeki tüm desteklenen dosyaları işler."""
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
        """Verilen dosya yollarını arka planda işleyip veritabanına ekler."""
        if self._busy or not paths:
            return

        self._set_busy(True, "Dokümanlar işleniyor...")
        total = len(paths)

        def task():
            total_chunks = 0
            skipped = 0
            messages = []
            existing = set(self.embedder.list_sources())
            for index, path in enumerate(paths, start=1):
                name = os.path.basename(path)
                if name in existing:
                    skipped += 1
                    messages.append(f"[ATLANDI] {name}: zaten yüklü")
                    self._post_ui(
                        lambda n=name, i=index: self.rail.set_status(
                            f"Atlanıyor ({i}/{total}): {n} (zaten yüklü)", "orange"
                        )
                    )
                    continue
                self._post_ui(
                    lambda n=name, i=index: self.rail.set_status(
                        f"İşleniyor ({i}/{total}): {n}", "orange"
                    )
                )
                result = self.processor.process_file(path)
                if result.success and result.chunks:
                    added = self.embedder.add_chunks(result.chunks)
                    total_chunks += added
                    existing.add(name)
                    messages.append(f"[OK] {result.message}")
                else:
                    messages.append(f"[HATA] {result.message}")
            return total_chunks, skipped, messages

        def done(result, error):
            self._set_busy(False)
            if error:
                self.rail.set_status("İşleme hatası!", "red")
                messagebox.showerror("Hata", f"Doküman işlenirken hata oluştu:\n{error}")
                return
            _total_chunks, skipped, messages = result
            self.settings_view.set_documents(self.embedder.list_sources())
            count = self.embedder.get_document_count()
            self.rail.set_status(f"Hazır ({count} parça)", "lightgreen")
            if len(messages) > 15:
                shown = messages[:15] + [f"... ve {len(messages) - 15} dosya daha"]
            else:
                shown = messages
            ozet = "Yükleme tamamlandı"
            if skipped:
                ozet += f" ({skipped} dosya zaten yüklüydü, atlandı)"
            self.chat.add_message("Sistem", ozet + ":\n" + "\n".join(shown), is_user=False)

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
            "Onay", "Tüm yüklü dokümanlar veritabanından silinecek. Emin misiniz?",
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
                self.rail.set_status("Temizleme hatası!", "red")
            else:
                self.settings_view.set_documents([])
                self.rail.set_status("Hazır (0 parça)", "lightgreen")
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

        # 2) Kullanıcı mesajını göster ve oturuma ekle.
        self.chat.add_message("Siz", question, is_user=True)
        ChatStore.add_message(self.current_session, ROLE_USER, question)
        self._persist_current()
        self._refresh_sessions()

        thinking_label = self.chat.add_message("Asistan", "Düşünüyor...", is_user=False)
        self._set_busy(True, "Cevap üretiliyor...")

        def task():
            from perf_monitor import ResourceSampler, format_perf_block

            with ResourceSampler() as sampler:
                t0 = time.perf_counter()
                contexts = self.embedder.similarity_search(question, k=self.TOP_K)
                t1 = time.perf_counter()
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
            self.rail.set_status(
                f"Hazır ({self.embedder.get_document_count()} parça)", "lightgreen"
            )
            if error:
                self.chat.update_message(thinking_label, f"Hata: {error}")
                return

            response, contexts, perf_text = result
            if not response.success:
                self.chat.update_message(thinking_label, f"Hata: {response.error}")
                return

            answer = response.text
            if contexts:
                sources = sorted({
                    f"{c.source}" + (f" (s.{c.page})" if c.page else "")
                    for c in contexts
                })
                answer += "\n\nKaynaklar: " + ", ".join(sources)
            answer += "\n\n" + perf_text
            self.chat.update_message(thinking_label, answer)

            # Asistan cevabını oturuma ekle ve kaydet.
            ChatStore.add_message(self.current_session, ROLE_ASSISTANT, answer)
            self.current_session["model"] = (response.meta or {}).get("model", "")
            self._persist_current()
            self._refresh_sessions()

        self._run_in_background(task, on_done=done)

    def _configure_provider(self) -> bool:
        """UI seçimine göre LLM sağlayıcıyı kurar. Başarılıysa True döner."""
        provider = self.models_view.get_provider()

        if provider.startswith("Gemini"):
            api_key = self.models_view.get_api_key()
            if not api_key:
                messagebox.showwarning(
                    "Eksik Bilgi",
                    "Gemini kullanmak için 'Modeller' sekmesinden API anahtarı "
                    "girmelisiniz.\nİnternet yoksa 'Yerel Model' seçeneğini kullanın.",
                )
                self._show_view("models")
                return False
            if not self._has_internet():
                messagebox.showwarning(
                    "İnternet Yok",
                    "Gemini için internet bağlantısı gerekir. Çevrimdışı çalışmak "
                    "için 'Yerel Model' seçeneğine geçin.",
                )
                return False
            self.llm.use_gemini(api_key=api_key)
            return True

        # Yerel Model (gömülü llama.cpp)
        if not LLMConnector.is_local_engine_available():
            messagebox.showerror(
                "Motor Bulunamadı",
                "Gömülü model motoru (llama-cpp-python) yüklenemedi.\n"
                "Kurulum: pip install llama-cpp-python",
            )
            return False
        model = self.models_view.get_active_model()
        if not model or model.startswith("("):
            messagebox.showwarning(
                "Model Yok",
                "Önce 'Modeller' sekmesinden bir model indirip seçin.",
            )
            self._show_view("models")
            return False
        path = self.model_manager.path_of(model)
        if not os.path.isfile(path):
            messagebox.showwarning(
                "Model Bulunamadı",
                f"Seçili model dosyası bulunamadı:\n{model}\nLütfen tekrar indirin.",
            )
            self._refresh_model_lists()
            return False
        self.llm.use_local(path)
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
        # Aktif sohbeti kaydet.
        try:
            self._persist_current()
        except Exception:  # noqa: BLE001
            pass
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
