# -*- coding: utf-8 -*-
"""
ui_components.py
================
CustomTkinter ile modern, tamamen Türkçe, Gemini benzeri arayüz bileşenleri.

Yerleşim:
    * ChatHistoryRail : en solda, geçmiş sohbetler + gezinme (Sohbet/Modeller/Ayarlar).
    * ChatArea        : sohbet geçmişi ve soru giriş kutusu.
    * ModelsView      : sağlayıcı seçimi, yerel model indirme/seçme/silme.
    * SettingsView    : doküman yönetimi ve durum.

Sorumluluklar (SRP):
    * Görsel bileşenleri oluşturmak.
    * Kullanıcı etkileşimlerini, dışarıdan verilen 'callback' fonksiyonlarına
      iletmek (iş mantığı içermez; sadece sunum katmanıdır).

Bu modül; doküman, embedding veya LLM modüllerini import ETMEZ. Tamamen
ayrıştırılmıştır (DIP). main.py bu bileşenleri oluşturur ve geri çağırma
fonksiyonlarını bağlar.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import customtkinter as ctk

# Genel tema ayarları
ctk.set_appearance_mode("dark")          # "dark" / "light" / "system"
ctk.set_default_color_theme("blue")

# Renk paleti (deniz/teknik tema)
COLOR_USER_BUBBLE = "#1f6aa5"
COLOR_BOT_BUBBLE = "#343638"
COLOR_INFO = "#2b2b2b"
COLOR_RAIL = "#202123"
COLOR_RAIL_ITEM = "#2a2b32"
COLOR_RAIL_ACTIVE = "#1f6aa5"


# ---------------------------------------------------------------------------
#  Sol Ray (Geçmiş Sohbetler + Gezinme)
# ---------------------------------------------------------------------------
class ChatHistoryRail(ctk.CTkFrame):
    """
    En soldaki dar panel: yeni sohbet, geçmiş sohbet listesi (seçme/silme) ve
    görünümler arası gezinme (Sohbet / Modeller / Ayarlar) düğmeleri.
    """

    def __init__(
        self,
        master,
        on_new_chat: Callable[[], None],
        on_select_chat: Callable[[str], None],
        on_delete_chat: Callable[[str], None],
        on_show_view: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(master, width=250, corner_radius=0, fg_color=COLOR_RAIL, **kwargs)
        self.grid_propagate(False)

        self._on_select_chat = on_select_chat
        self._on_delete_chat = on_delete_chat
        self._on_show_view = on_show_view
        self._active_id: Optional[str] = None
        self._row_widgets: List[ctk.CTkBaseClass] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)  # Sohbet listesi büyüsün.

        # --- Başlık ---
        self.title_label = ctk.CTkLabel(
            self, text="Gemi Asistanı",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))

        # --- Yeni sohbet ---
        self.new_chat_btn = ctk.CTkButton(
            self, text="+  Yeni Sohbet", height=40,
            command=on_new_chat, font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.new_chat_btn.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        # --- Geçmiş sohbetler (kaydırılabilir) ---
        self.list_frame = ctk.CTkScrollableFrame(
            self, label_text="Geçmiş Sohbetler", fg_color="transparent"
        )
        self.list_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.list_frame.grid_columnconfigure(0, weight=1)

        # --- Gezinme düğmeleri (alt, 2x2 ızgara) ---
        nav = ctk.CTkFrame(self, fg_color="transparent")
        nav.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 6))
        nav.grid_columnconfigure((0, 1), weight=1)
        self._nav_btns: Dict[str, ctk.CTkButton] = {}
        nav_items = (
            ("chat", "Sohbet"), ("models", "Modeller"),
            ("downloads", "İndirilenler"), ("settings", "Ayarlar"),
        )
        for i, (key, text) in enumerate(nav_items):
            btn = ctk.CTkButton(
                nav, text=text, height=32,
                font=ctk.CTkFont(size=12),
                command=lambda k=key: self._on_show_view(k),
            )
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            self._nav_btns[key] = btn

        # --- Durum çubuğu (en altta) ---
        self.status_label = ctk.CTkLabel(
            self, text="● Hazırlanıyor...",
            font=ctk.CTkFont(size=11), text_color="gray70", anchor="w",
            wraplength=220, justify="left",
        )
        self.status_label.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 12))

        self.set_active_view("chat")

    # -- Geçmiş listesi ---------------------------------------------------- #
    def set_sessions(self, sessions: List[Dict], active_id: Optional[str] = None) -> None:
        """Geçmiş sohbet listesini yeniden çizer."""
        self._active_id = active_id
        for w in self._row_widgets:
            w.destroy()
        self._row_widgets.clear()

        if not sessions:
            empty = ctk.CTkLabel(
                self.list_frame, text="Henüz sohbet yok.",
                font=ctk.CTkFont(size=11), text_color="gray60",
            )
            empty.grid(row=0, column=0, sticky="ew", pady=6)
            self._row_widgets.append(empty)
            return

        for i, s in enumerate(sessions):
            sid = s.get("id")
            is_active = sid == active_id
            row = ctk.CTkFrame(
                self.list_frame,
                fg_color=COLOR_RAIL_ACTIVE if is_active else COLOR_RAIL_ITEM,
                corner_radius=8,
            )
            row.grid(row=i, column=0, sticky="ew", pady=3)
            row.grid_columnconfigure(0, weight=1)

            title = s.get("title") or "Yeni sohbet"
            open_btn = ctk.CTkButton(
                row, text=title, anchor="w", height=34,
                fg_color="transparent", hover_color="#3a3b42",
                font=ctk.CTkFont(size=12),
                command=lambda sid=sid: self._on_select_chat(sid),
            )
            open_btn.grid(row=0, column=0, sticky="ew", padx=(6, 0), pady=2)

            del_btn = ctk.CTkButton(
                row, text="×", width=28, height=28,
                fg_color="transparent", hover_color="#8a2c2c",
                font=ctk.CTkFont(size=16),
                command=lambda sid=sid: self._on_delete_chat(sid),
            )
            del_btn.grid(row=0, column=1, padx=(0, 4), pady=2)
            self._row_widgets.append(row)

    # -- Gezinme ----------------------------------------------------------- #
    def set_active_view(self, name: str) -> None:
        """Aktif görünüm düğmesini vurgular."""
        for key, btn in self._nav_btns.items():
            if key == name:
                btn.configure(fg_color=COLOR_RAIL_ACTIVE)
            else:
                btn.configure(fg_color="transparent")

    # -- Durum / kontrol --------------------------------------------------- #
    def set_status(self, text: str, color: str = "gray70") -> None:
        self.status_label.configure(text=f"● {text}", text_color=color)

    def set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.new_chat_btn.configure(state=state)


# ---------------------------------------------------------------------------
#  Sohbet Alanı
# ---------------------------------------------------------------------------
class ChatArea(ctk.CTkFrame):
    """Sohbet geçmişi, soru giriş kutusu ve gönder butonunu içeren ana alan."""

    GREETING = (
        "Merhaba! Ben gemi teknik doküman asistanınızım. 'Ayarlar' sekmesinden "
        "doküman yükleyip, 'Modeller' sekmesinden bir model seçtikten sonra "
        "sorularınızı sorabilirsiniz."
    )

    def __init__(self, master, on_send: Callable[[str], None], **kwargs) -> None:
        super().__init__(master, corner_radius=0, **kwargs)
        self._on_send = on_send

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Kaydırılabilir sohbet geçmişi ---
        self.history = ctk.CTkScrollableFrame(self, label_text="Sohbet")
        self.history.grid(row=0, column=0, sticky="nsew", padx=15, pady=(15, 5))
        self.history.grid_columnconfigure(0, weight=1)

        # --- Giriş satırı ---
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(5, 15))
        input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Dokümanlar hakkında bir soru yazın...",
            height=45, font=ctk.CTkFont(size=14),
        )
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", lambda _event: self._handle_send())

        self.send_btn = ctk.CTkButton(
            input_frame, text="Gönder", width=110, height=45,
            command=self._handle_send, font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.send_btn.grid(row=0, column=1)

        self._row = 0
        self._containers: List[ctk.CTkBaseClass] = []
        self.show_greeting()

    # -- İç olaylar -------------------------------------------------------- #
    def _handle_send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        if self._on_send:
            self._on_send(text)

    # -- Geçmiş yönetimi --------------------------------------------------- #
    def clear(self) -> None:
        """Tüm mesaj balonlarını kaldırır."""
        for c in self._containers:
            try:
                c.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._containers.clear()
        self._row = 0

    def show_greeting(self) -> None:
        """Boş bir sohbet için karşılama mesajını gösterir."""
        self.clear()
        self.add_message("Asistan", self.GREETING, is_user=False)

    def load_messages(self, messages: List[Dict]) -> None:
        """Kaydedilmiş bir oturumun mesajlarını yükler."""
        self.clear()
        role_to_sender = {"user": "Siz", "assistant": "Asistan", "system": "Sistem"}
        for m in messages or []:
            role = m.get("role", "assistant")
            sender = role_to_sender.get(role, "Asistan")
            self.add_message(sender, m.get("text", ""), is_user=(role == "user"))
        if not messages:
            self.show_greeting()

    # -- Dışarıya açık yardımcılar ---------------------------------------- #
    def add_message(self, sender: str, text: str, is_user: bool = False) -> ctk.CTkLabel:
        """Sohbete bir mesaj balonu ekler ve eklenen etiketi döndürür."""
        anchor = "e" if is_user else "w"
        bubble_color = COLOR_USER_BUBBLE if is_user else COLOR_BOT_BUBBLE

        container = ctk.CTkFrame(self.history, fg_color="transparent")
        container.grid(row=self._row, column=0, sticky="ew", pady=4)
        container.grid_columnconfigure(0, weight=1)

        bubble = ctk.CTkFrame(container, fg_color=bubble_color, corner_radius=12)
        bubble.grid(row=0, column=0, sticky=anchor, padx=8)

        ctk.CTkLabel(
            bubble, text=sender, font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray85", anchor="w",
        ).pack(anchor="w", padx=12, pady=(8, 0))

        msg_label = ctk.CTkLabel(
            bubble, text=text, font=ctk.CTkFont(size=14),
            justify="left", anchor="w", wraplength=620,
        )
        msg_label.pack(anchor="w", padx=12, pady=(2, 4))

        # Mesaj metnini panoya kopyalama butonu.
        copy_btn = ctk.CTkButton(
            bubble, text="Kopyala", width=90, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="transparent", border_width=1, hover_color="#4a4d50",
        )
        copy_btn.configure(
            command=lambda lbl=msg_label, btn=copy_btn: self._copy_to_clipboard(lbl, btn)
        )
        copy_btn.pack(anchor="e", padx=12, pady=(0, 8))

        self._containers.append(container)
        self._row += 1
        # Yeni mesaja otomatik kaydır.
        self.after(50, self._scroll_to_bottom)
        return msg_label

    def _copy_to_clipboard(self, label: ctk.CTkLabel, button: ctk.CTkButton) -> None:
        """İlgili mesaj balonunun o anki metnini panoya kopyalar."""
        try:
            text = label.cget("text")
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_idletasks()  # Panoya yazmanın tamamlanmasını garanti et.
        except Exception:  # noqa: BLE001
            return
        # Kısa görsel geri bildirim: "Kopyalandı" -> 1.5 sn sonra eski metne dön.
        try:
            button.configure(text="Kopyalandı")
            self.after(1500, lambda: self._reset_copy_button(button))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _reset_copy_button(button: ctk.CTkButton) -> None:
        """Kopyala butonunu eski metnine döndürür (widget hâlâ varsa)."""
        try:
            button.configure(text="Kopyala")
        except Exception:  # noqa: BLE001
            pass

    def update_message(self, label: ctk.CTkLabel, text: str) -> None:
        """Mevcut bir mesaj balonunun metnini günceller (örn. 'Yazıyor...' -> cevap)."""
        try:
            label.configure(text=text)
            self.after(50, self._scroll_to_bottom)
        except Exception:  # noqa: BLE001
            pass  # Widget yok edilmişse sessizce geç.

    def _scroll_to_bottom(self) -> None:
        try:
            self.history._parent_canvas.yview_moveto(1.0)
        except Exception:  # noqa: BLE001
            pass

    def set_input_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.send_btn.configure(state=state)


# ---------------------------------------------------------------------------
#  Modeller Görünümü (sağlayıcı + yerel model indirme/seçme)
# ---------------------------------------------------------------------------
class ModelsView(ctk.CTkFrame):
    """
    Sağlayıcı seçimi (Gemini / Yerel) ve aktif yerel modelin seçimi. Gemini için
    API anahtarı girişi de buradadır. Model indirme işlemi ayrı 'İndirilenler'
    görünümündedir (DownloadsView).
    """

    PROVIDER_GEMINI = "Gemini (Çevrimiçi)"
    PROVIDER_LOCAL = "Yerel Model (Çevrimdışı)"

    def __init__(
        self,
        master,
        on_provider_change: Callable[[str], None],
        on_select_model: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(master, corner_radius=0, **kwargs)
        self._on_provider_change = on_provider_change
        self._on_select_model = on_select_model

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            self, text="Modeller ve Sağlayıcı",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        # --- Sağlayıcı seçimi ---
        prov_frame = ctk.CTkFrame(self, fg_color="transparent")
        prov_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))
        ctk.CTkLabel(
            prov_frame, text="Yapay Zeka Sağlayıcısı:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.provider_seg = ctk.CTkSegmentedButton(
            prov_frame,
            values=[self.PROVIDER_LOCAL, self.PROVIDER_GEMINI],
            command=self._handle_provider_change,
        )
        self.provider_seg.set(self.PROVIDER_LOCAL)
        self.provider_seg.pack(fill="x")

        # --- Gemini API anahtarı (Gemini seçilince) ---
        self.gemini_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(
            self.gemini_frame, text="Gemini API Anahtarı:",
            font=ctk.CTkFont(size=12), anchor="w",
        ).pack(fill="x", pady=(0, 2))
        self.api_key_entry = ctk.CTkEntry(
            self.gemini_frame, placeholder_text="Gemini API Anahtarı", show="*",
        )
        self.api_key_entry.pack(fill="x")
        ctk.CTkLabel(
            self.gemini_frame,
            text="Gemini için internet bağlantısı ve geçerli bir anahtar gerekir.",
            font=ctk.CTkFont(size=11), text_color="gray60", anchor="w",
            justify="left", wraplength=560,
        ).pack(fill="x", pady=(4, 0))

        # --- Yerel model alanı (Yerel seçilince) ---
        self.local_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.local_frame.grid_columnconfigure(0, weight=1)

        active_row = ctk.CTkFrame(self.local_frame, fg_color="transparent")
        active_row.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            active_row, text="Aktif yerel model:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 10))
        self.active_menu = ctk.CTkOptionMenu(
            active_row, values=["(model indirilmedi)"], command=self._handle_select,
            width=320,
        )
        self.active_menu.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            self.local_frame,
            text="Yeni model indirmek veya silmek için sol menüden 'İndirilenler' "
                 "sekmesini kullanın. İndirilen modeller burada listelenir.",
            font=ctk.CTkFont(size=11), text_color="gray60", anchor="w",
            justify="left", wraplength=560,
        ).pack(fill="x", pady=(2, 0))

        # Yerel/gemini başlangıç görünürlüğü.
        self._handle_provider_change(self.PROVIDER_LOCAL)

    # -- Sağlayıcı --------------------------------------------------------- #
    def _handle_provider_change(self, value: str) -> None:
        self.gemini_frame.grid_forget()
        self.local_frame.grid_forget()
        if value.startswith("Gemini"):
            self.gemini_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(4, 10))
        else:
            self.local_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(4, 10))
        if self._on_provider_change:
            self._on_provider_change(value)

    def get_provider(self) -> str:
        return self.provider_seg.get()

    def set_provider(self, value: str) -> None:
        self.provider_seg.set(value)
        self._handle_provider_change(value)

    def get_api_key(self) -> str:
        return self.api_key_entry.get().strip()

    # -- Aktif model ------------------------------------------------------- #
    def _handle_select(self, value: str) -> None:
        if value and not value.startswith("(") and self._on_select_model:
            self._on_select_model(value)

    def set_downloaded_models(self, models: List[str], active: Optional[str] = None) -> None:
        """İndirilmiş model listesini (aktif seçim menüsü) günceller."""
        if models:
            self.active_menu.configure(values=models, state="normal")
            self.active_menu.set(active if active in models else models[0])
        else:
            self.active_menu.configure(values=["(model indirilmedi)"], state="disabled")
            self.active_menu.set("(model indirilmedi)")

    def get_active_model(self) -> str:
        return self.active_menu.get()

    def set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.provider_seg.configure(state=state)
        self.active_menu.configure(state=state)


# ---------------------------------------------------------------------------
#  İndirilenler Görünümü (Model kataloğu + indirme)
# ---------------------------------------------------------------------------
class DownloadsView(ctk.CTkFrame):
    """
    Model indirme merkezi. Üç yol sunar:
      * Önerilenler: elle seçilmiş, Türkçe-yetkin hızlı seçim listesi.
      * Arama: Hugging Face'te GGUF modellerini isimle arayıp repodaki bir
        quantization dosyasını seçerek indirme (tüm güncel modellere erişim).
      * Manuel: repo_id ve dosya adını doğrudan yazarak indirme.
    İndirilen modeller çevrimdışı kullanılabilir ve 'Modeller' sekmesinden aktif
    model olarak seçilir. Sağlayıcı seçiminden bağımsız her zaman erişilebilir.
    """

    def __init__(
        self,
        master,
        on_download: Callable[[Dict], None],
        on_cancel_download: Callable[[], None],
        on_delete_model: Callable[[str], None],
        on_search: Callable[[str], None],
        on_select_repo: Callable[[str], None],
        **kwargs,
    ) -> None:
        super().__init__(master, corner_radius=0, **kwargs)
        self._on_download = on_download
        self._on_cancel_download = on_cancel_download
        self._on_delete_model = on_delete_model
        self._on_search = on_search
        self._on_select_repo = on_select_repo

        # Görünüm durumu ve önbellek.
        self._mode = "curated"        # "curated" | "search" | "files"
        self._loading = False
        self._curated: List[Dict] = []
        self._downloaded: List[str] = []
        self._search_results: List[Dict] = []
        self._search_title = "Arama sonuçları (Hugging Face)"
        self._repo_files: List[Dict] = []
        self._current_repo: Optional[str] = None
        self._rows: List[ctk.CTkBaseClass] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            self, text="İndirilenler",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 4))

        ctk.CTkLabel(
            self,
            text="Önerilen modelleri tek tıkla indirin ya da Hugging Face'te "
                 "arayarak/elle girerek tüm güncel modellere erişin. İndirme "
                 "anında internet gerekir; indirildikten sonra kullanım "
                 "çevrimdışıdır. '.gguf' dosyaları başka bir bilgisayarın aynı "
                 "klasörüne kopyalanarak da taşınabilir.",
            font=ctk.CTkFont(size=12), text_color="gray60", anchor="w",
            justify="left", wraplength=620,
        ).grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))

        # --- Arama satırı ---
        search_frame = ctk.CTkFrame(self, fg_color="transparent")
        search_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 6))
        search_frame.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Hugging Face'te model ara (ör. qwen3, llama 3.3, phi-4)",
            height=38,
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.search_entry.bind("<Return>", lambda _e: self._handle_search())
        self.search_btn = ctk.CTkButton(
            search_frame, text="Ara", width=80, height=38, command=self._handle_search,
        )
        self.search_btn.grid(row=0, column=1, padx=(0, 6))
        self.browse_btn = ctk.CTkButton(
            search_frame, text="Tüm modeller", width=120, height=38,
            command=self._handle_browse,
        )
        self.browse_btn.grid(row=0, column=2, padx=(0, 6))
        self.curated_btn = ctk.CTkButton(
            search_frame, text="Önerilenler", width=110, height=38,
            fg_color="transparent", border_width=1, command=self.show_curated,
        )
        self.curated_btn.grid(row=0, column=3)

        # --- Bağlam başlığı + geri düğmesi ---
        head_frame = ctk.CTkFrame(self, fg_color="transparent")
        head_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 2))
        head_frame.grid_columnconfigure(0, weight=1)
        self.context_label = ctk.CTkLabel(
            head_frame, text="", font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        )
        self.context_label.grid(row=0, column=0, sticky="ew")
        self.back_btn = ctk.CTkButton(
            head_frame, text="← Geri", width=90, height=28,
            fg_color="transparent", border_width=1,
            command=self._handle_back,
        )
        # back_btn yalnızca repo dosya listesindeyken görünür.

        # --- Sonuç listesi (curated/arama/dosyalar ortak alan) ---
        self.list_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.list_frame.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 6))
        self.list_frame.grid_columnconfigure(0, weight=1)

        # --- Manuel giriş ---
        manual = ctk.CTkFrame(self, fg_color=COLOR_INFO, corner_radius=8)
        manual.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 6))
        manual.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkLabel(
            manual, text="Manuel indirme (repo kimliği ve dosya adı):",
            font=ctk.CTkFont(size=12, weight="bold"), anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(8, 2))
        self.manual_repo = ctk.CTkEntry(
            manual, placeholder_text="repo_id (ör. bartowski/Qwen2.5-7B-Instruct-GGUF)",
        )
        self.manual_repo.grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 8))
        self.manual_file = ctk.CTkEntry(
            manual, placeholder_text="dosya adı (ör. Qwen2.5-7B-Instruct-Q4_K_M.gguf)",
        )
        self.manual_file.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 8))
        self.manual_btn = ctk.CTkButton(
            manual, text="İndir", width=90, command=self._handle_manual,
        )
        self.manual_btn.grid(row=1, column=2, padx=(4, 10), pady=(0, 8))

        # --- İndirme ilerleme alanı ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_label = ctk.CTkLabel(
            self.progress_frame, text="", font=ctk.CTkFont(size=12), anchor="w",
        )
        self.progress_label.pack(fill="x")
        prog_row = ctk.CTkFrame(self.progress_frame, fg_color="transparent")
        prog_row.pack(fill="x", pady=(2, 0))
        prog_row.grid_columnconfigure(0, weight=1)
        self.progress_bar = ctk.CTkProgressBar(prog_row)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.cancel_btn = ctk.CTkButton(
            prog_row, text="İptal", width=80, command=self._on_cancel_download,
            fg_color="#8a2c2c", hover_color="#a83232",
        )
        self.cancel_btn.grid(row=0, column=1)

        self._render()

    # -- Yardımcılar ------------------------------------------------------- #
    @staticmethod
    def _basename(path: str) -> str:
        return path.replace("\\", "/").rsplit("/", 1)[-1]

    def _is_downloaded(self, filename: str) -> bool:
        return self._basename(filename) in self._downloaded

    @staticmethod
    def _fmt_size(num_bytes: Optional[int]) -> str:
        if not num_bytes:
            return "boyut bilinmiyor"
        gb = num_bytes / (1024 ** 3)
        if gb >= 1:
            return f"~{gb:.1f} GB"
        return f"~{num_bytes / (1024 ** 2):.0f} MB"

    def _clear_rows(self) -> None:
        for w in self._rows:
            w.destroy()
        self._rows.clear()

    def _info_label(self, text: str) -> None:
        lbl = ctk.CTkLabel(
            self.list_frame, text=text, font=ctk.CTkFont(size=12),
            text_color="gray60", anchor="w", justify="left", wraplength=560,
        )
        lbl.grid(row=0, column=0, sticky="ew", pady=8, padx=4)
        self._rows.append(lbl)

    # -- Olay işleyiciler -------------------------------------------------- #
    def _handle_search(self) -> None:
        query = self.search_entry.get().strip()
        if query and self._on_search:
            self._on_search(query)

    def _handle_browse(self) -> None:
        """Arama terimi olmadan en popüler GGUF modellerini listeler."""
        self.search_entry.delete(0, "end")
        if self._on_search:
            self._on_search("")

    def _handle_back(self) -> None:
        # Repo dosya listesinden arama sonuçlarına dön.
        self._mode = "search"
        self._loading = False
        self._render()

    def _handle_manual(self) -> None:
        repo_id = self.manual_repo.get().strip()
        filename = self.manual_file.get().strip()
        if not repo_id or not filename:
            return
        self._on_download({
            "repo_id": repo_id, "filename": filename,
            "label": filename, "approx_mb": 0,
        })

    # -- Görünüm geçişleri (main.py çağırır) ------------------------------ #
    def set_curated(self, curated: List[Dict], downloaded: List[str]) -> None:
        """Önerilen liste ve indirilmiş model durumunu günceller."""
        self._curated = curated
        self._downloaded = downloaded
        self._render()  # mevcut moddaki indirilmiş/indirilmedi durumlarını tazeler.

    def show_curated(self) -> None:
        self._mode = "curated"
        self._loading = False
        self._render()

    def begin_search(self, title: str = "Arama sonuçları (Hugging Face)") -> None:
        self._mode = "search"
        self._loading = True
        self._search_results = []
        self._search_title = title
        self._render()

    def set_search_results(self, results: List[Dict],
                           downloaded: Optional[List[str]] = None) -> None:
        self._mode = "search"
        self._loading = False
        self._search_results = results
        if downloaded is not None:
            self._downloaded = downloaded
        self._render()

    def begin_repo_files(self, repo_id: str) -> None:
        self._mode = "files"
        self._loading = True
        self._current_repo = repo_id
        self._repo_files = []
        self._render()

    def set_repo_files(self, repo_id: str, files: List[Dict],
                       downloaded: Optional[List[str]] = None) -> None:
        self._mode = "files"
        self._loading = False
        self._current_repo = repo_id
        self._repo_files = files
        if downloaded is not None:
            self._downloaded = downloaded
        self._render()

    # -- Çizim ------------------------------------------------------------- #
    def _render(self) -> None:
        self._clear_rows()
        self.back_btn.grid_forget()

        if self._mode == "curated":
            self.context_label.configure(text="Önerilen modeller")
            self._render_curated()
        elif self._mode == "search":
            self.context_label.configure(text=self._search_title)
            if self._loading:
                self._info_label("Aranıyor...")
            elif not self._search_results:
                self._info_label("Sonuç yok. Farklı bir arama deneyin "
                                 "ya da manuel indirme bölümünü kullanın.")
            else:
                self._render_search()
        elif self._mode == "files":
            self.context_label.configure(text=f"Repo: {self._current_repo}")
            self.back_btn.grid(row=0, column=1, sticky="e")
            if self._loading:
                self._info_label("Dosyalar getiriliyor...")
            elif not self._repo_files:
                self._info_label("Bu repoda .gguf dosyası bulunamadı.")
            else:
                self._render_files()

    def _render_curated(self) -> None:
        for i, m in enumerate(self._curated):
            is_down = self._is_downloaded(m["filename"])
            durum = "İndirildi" if is_down else f"~{m['approx_mb'] / 1024:.1f} GB"
            ram = m.get("min_ram")
            ram_txt = f" · En az {ram} RAM/VRAM" if ram else ""
            info = (f"{m['label']}\n{m['params']} · {m['quant']} · "
                    f"{durum}{ram_txt} — {m['note']}")
            self._add_row(info, m["filename"], download_payload=m)

    def _render_search(self) -> None:
        for i, r in enumerate(self._search_results):
            repo_id = r["repo_id"]
            info = (f"{repo_id}\n{r.get('downloads', 0):,} indirme · "
                    f"{r.get('likes', 0)} beğeni").replace(",", ".")
            row = ctk.CTkFrame(self.list_frame, fg_color=COLOR_INFO, corner_radius=8)
            row.grid(row=i, column=0, sticky="ew", pady=3)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                row, text=info, justify="left", anchor="w",
                font=ctk.CTkFont(size=12), wraplength=480,
            ).grid(row=0, column=0, sticky="ew", padx=10, pady=8)
            ctk.CTkButton(
                row, text="Dosyalar", width=90,
                command=lambda rid=repo_id: self._on_select_repo(rid),
            ).grid(row=0, column=1, padx=10, pady=8)
            self._rows.append(row)

    def _render_files(self) -> None:
        # Yalnızca tek-dosya, indirilebilir modelleri göster; çok parçalı (split)
        # ve görsel eki (mmproj) dosyaları gizle.
        models = [f for f in self._repo_files if f.get("kind", "model") == "model"]
        hidden = len(self._repo_files) - len(models)

        if not models:
            self._info_label(
                "Bu repoda doğrudan indirilebilir tek-dosya model bulunamadı "
                "(yalnızca çok parçalı veya görsel eki dosyalar var). Farklı bir "
                "repo deneyin."
            )
            return

        for f in models:
            fname = f["filename"]
            size = f.get("size")
            is_down = self._is_downloaded(fname)
            durum = "İndirildi" if is_down else self._fmt_size(size)
            quant = f.get("quant", "?")
            quality = f.get("quality", "")
            info = f"{self._basename(fname)}\n{quant} · {quality} · {durum}"
            mb = int(size / (1024 * 1024)) if size else 0
            payload = {
                "repo_id": self._current_repo, "filename": fname,
                "label": self._basename(fname), "approx_mb": mb,
            }
            self._add_row(info, fname, download_payload=payload)

        if hidden:
            note = ctk.CTkLabel(
                self.list_frame,
                text=f"{hidden} ek dosya gizlendi (çok parçalı model veya görsel "
                     f"ekleri — uygulamada kullanılamaz).",
                font=ctk.CTkFont(size=11), text_color="gray55",
                anchor="w", justify="left", wraplength=540,
            )
            note.grid(row=len(self._rows), column=0, sticky="ew", pady=(6, 2), padx=4)
            self._rows.append(note)

    def _add_row(self, info: str, filename: str, download_payload: Dict) -> None:
        """İndir/Sil düğmeli ortak satır (curated ve dosya listeleri için)."""
        i = len(self._rows)
        is_down = self._is_downloaded(filename)
        row = ctk.CTkFrame(self.list_frame, fg_color=COLOR_INFO, corner_radius=8)
        row.grid(row=i, column=0, sticky="ew", pady=3)
        row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            row, text=info, justify="left", anchor="w",
            font=ctk.CTkFont(size=12), wraplength=480,
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        if is_down:
            btn = ctk.CTkButton(
                row, text="Sil", width=90,
                fg_color="#8a2c2c", hover_color="#a83232",
                command=lambda fn=self._basename(filename): self._on_delete_model(fn),
            )
        else:
            btn = ctk.CTkButton(
                row, text="İndir", width=90,
                command=lambda mm=download_payload: self._on_download(mm),
            )
        btn.grid(row=0, column=1, padx=10, pady=8)
        self._rows.append(row)

    # -- İndirme ilerleme -------------------------------------------------- #
    def show_progress(self, visible: bool) -> None:
        if visible:
            self.progress_frame.grid(row=6, column=0, sticky="ew", padx=20, pady=(0, 16))
        else:
            self.progress_frame.grid_forget()

    def set_progress(self, done: int, total: int, label: str = "") -> None:
        frac = (done / total) if total else 0
        self.progress_bar.set(frac)
        if label:
            self.progress_label.configure(text=label)
        elif total:
            self.progress_label.configure(
                text=f"İndiriliyor: %{int(frac*100)} "
                     f"({done//(1024*1024)}/{total//(1024*1024)} MB)"
            )

    def set_controls_enabled(self, enabled: bool) -> None:
        """Meşgulken arama ve manuel indirme kontrollerini kilitler."""
        state = "normal" if enabled else "disabled"
        for w in (self.search_entry, self.search_btn, self.curated_btn,
                  self.manual_repo, self.manual_file, self.manual_btn):
            w.configure(state=state)


# ---------------------------------------------------------------------------
#  Ayarlar Görünümü (Doküman Yönetimi)
# ---------------------------------------------------------------------------
class SettingsView(ctk.CTkFrame):
    """Doküman yükleme/temizleme ve yüklü doküman listesi."""

    def __init__(
        self,
        master,
        on_upload: Callable[[], None],
        on_upload_folder: Callable[[], None],
        on_clear_db: Callable[[], None],
        dnd_available: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(master, corner_radius=0, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(
            self, text="Doküman Yönetimi",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 6))
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        self.upload_btn = ctk.CTkButton(
            btn_frame, text="Dosya Yükle", command=on_upload, height=40,
        )
        self.upload_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.upload_folder_btn = ctk.CTkButton(
            btn_frame, text="Klasör Yükle", command=on_upload_folder, height=40,
        )
        self.upload_folder_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        if dnd_available:
            ctk.CTkLabel(
                self,
                text="İpucu: Dosya veya klasörü pencereye sürükleyip bırakabilirsiniz.",
                font=ctk.CTkFont(size=11), text_color="gray60",
                anchor="w", justify="left",
            ).grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 6))

        self.clear_btn = ctk.CTkButton(
            self, text="Veritabanını Temizle",
            fg_color="#8a2c2c", hover_color="#a83232", command=on_clear_db,
        )
        self.clear_btn.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            self, text="Yüklü Dokümanlar:",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).grid(row=4, column=0, sticky="ew", padx=20, pady=(6, 2))

        self.doc_list = ctk.CTkTextbox(self, state="disabled")
        self.doc_list.grid(row=5, column=0, sticky="nsew", padx=20, pady=(0, 20))

    def set_documents(self, sources: List[str]) -> None:
        self.doc_list.configure(state="normal")
        self.doc_list.delete("1.0", "end")
        if sources:
            for src in sources:
                self.doc_list.insert("end", f"• {src}\n")
        else:
            self.doc_list.insert("end", "Henüz doküman yüklenmedi.")
        self.doc_list.configure(state="disabled")

    def set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for w in (self.upload_btn, self.upload_folder_btn, self.clear_btn):
            w.configure(state=state)
