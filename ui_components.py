# -*- coding: utf-8 -*-
"""
ui_components.py
================
CustomTkinter ile modern, tamamen Türkçe arayüz bileşenleri.

Sorumluluklar (SRP):
    * Görsel bileşenleri (sol panel, sohbet alanı, giriş kutusu) oluşturmak.
    * Kullanıcı etkileşimlerini, dışarıdan verilen 'callback' fonksiyonlarına
      iletmek (iş mantığı içermez; sadece sunum katmanıdır).

Bu modül; doküman, embedding veya LLM modüllerini import ETMEZ. Tamamen
ayrıştırılmıştır (Bağımlılığı Tersine Çevirme - DIP). main.py bu bileşenleri
oluşturur ve geri çağırma fonksiyonlarını bağlar.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import customtkinter as ctk

# Genel tema ayarları
ctk.set_appearance_mode("dark")          # "dark" / "light" / "system"
ctk.set_default_color_theme("blue")

# Renk paleti (deniz/teknik tema)
COLOR_USER_BUBBLE = "#1f6aa5"
COLOR_BOT_BUBBLE = "#343638"
COLOR_INFO = "#2b2b2b"


# ---------------------------------------------------------------------------
#  Sol Panel (Kontrol Paneli)
# ---------------------------------------------------------------------------
class Sidebar(ctk.CTkFrame):
    """Model seçimi, dosya yükleme ve veritabanı kontrollerini içeren sol panel."""

    def __init__(
        self,
        master,
        on_provider_change: Callable[[str], None],
        on_upload: Callable[[], None],
        on_upload_folder: Callable[[], None],
        on_refresh_ollama: Callable[[], None],
        on_clear_db: Callable[[], None],
        dnd_available: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(master, width=300, corner_radius=0, **kwargs)
        self.grid_propagate(False)

        self._on_provider_change = on_provider_change

        # --- Başlık ---
        self.title_label = ctk.CTkLabel(
            self,
            text="⚓ Gemi Doküman\nAsistanı",
            font=ctk.CTkFont(size=22, weight="bold"),
            justify="center",
        )
        self.title_label.pack(pady=(20, 10), padx=20)

        self.subtitle = ctk.CTkLabel(
            self,
            text="Çevrimdışı Teknik Yardımcı",
            font=ctk.CTkFont(size=12),
            text_color="gray70",
        )
        self.subtitle.pack(pady=(0, 20))

        # --- Model Sağlayıcı Seçimi ---
        ctk.CTkLabel(
            self, text="Yapay Zeka Modeli:",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(10, 2))

        self.provider_menu = ctk.CTkOptionMenu(
            self,
            values=["Gemini (Çevrimiçi)", "Ollama (Çevrimdışı)"],
            command=self._handle_provider_change,
        )
        self.provider_menu.pack(fill="x", padx=20, pady=(0, 10))

        # --- Gemini API Anahtarı (Gemini seçilince görünür) ---
        self.api_key_entry = ctk.CTkEntry(
            self, placeholder_text="Gemini API Anahtarı", show="*",
        )

        # --- Ollama Model Seçimi (Ollama seçilince görünür) ---
        self.ollama_menu = ctk.CTkOptionMenu(self, values=["(model bulunamadı)"])
        self.ollama_refresh_btn = ctk.CTkButton(
            self, text="🔄 Ollama Modellerini Yenile",
            fg_color="transparent", border_width=1, command=on_refresh_ollama,
        )

        # --- Doküman Yükleme ---
        ctk.CTkLabel(
            self, text="Doküman Yönetimi:",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(20, 2))

        self.upload_btn = ctk.CTkButton(
            self, text="📄 Dosya Yükle", command=on_upload, height=40,
        )
        self.upload_btn.pack(fill="x", padx=20, pady=(0, 6))

        self.upload_folder_btn = ctk.CTkButton(
            self, text="📁 Klasör Yükle", command=on_upload_folder, height=40,
        )
        self.upload_folder_btn.pack(fill="x", padx=20, pady=(0, 6))

        # Sürükle-bırak ipucu (yalnızca destekleniyorsa göster).
        if dnd_available:
            self.dnd_hint = ctk.CTkLabel(
                self,
                text="↪ İpucu: Dosya veya klasörü\npencereye sürükleyip bırakabilirsiniz.",
                font=ctk.CTkFont(size=11),
                text_color="gray60",
                justify="left",
            )
            self.dnd_hint.pack(fill="x", padx=20, pady=(0, 8))

        self.clear_btn = ctk.CTkButton(
            self, text="🗑️ Veritabanını Temizle",
            fg_color="#8a2c2c", hover_color="#a83232", command=on_clear_db,
        )
        self.clear_btn.pack(fill="x", padx=20, pady=(0, 10))

        # --- Yüklü Doküman Listesi ---
        ctk.CTkLabel(
            self, text="Yüklü Dokümanlar:",
            font=ctk.CTkFont(size=13, weight="bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(10, 2))

        self.doc_list = ctk.CTkTextbox(self, height=140, state="disabled")
        self.doc_list.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # --- Durum Çubuğu (en altta) ---
        self.status_label = ctk.CTkLabel(
            self, text="● Hazırlanıyor...",
            font=ctk.CTkFont(size=11), text_color="gray70", anchor="w",
        )
        self.status_label.pack(fill="x", padx=20, pady=(0, 15))

        # Başlangıçta sağlayıcıya uygun alanları göster.
        self._handle_provider_change(self.provider_menu.get())

    # -- İç olaylar -------------------------------------------------------- #
    def _handle_provider_change(self, value: str) -> None:
        """Sağlayıcı değişince ilgili giriş alanlarını gösterir/gizler."""
        # Önce hepsini gizle.
        self.api_key_entry.pack_forget()
        self.ollama_menu.pack_forget()
        self.ollama_refresh_btn.pack_forget()

        if value.startswith("Gemini"):
            self.api_key_entry.pack(
                fill="x", padx=20, pady=(0, 10),
                after=self.provider_menu,
            )
        else:  # Ollama
            self.ollama_menu.pack(
                fill="x", padx=20, pady=(0, 4), after=self.provider_menu,
            )
            self.ollama_refresh_btn.pack(
                fill="x", padx=20, pady=(0, 10), after=self.ollama_menu,
            )

        if self._on_provider_change:
            self._on_provider_change(value)

    # -- Dışarıya açık yardımcılar ---------------------------------------- #
    def get_provider(self) -> str:
        return self.provider_menu.get()

    def get_api_key(self) -> str:
        return self.api_key_entry.get().strip()

    def get_ollama_model(self) -> str:
        return self.ollama_menu.get()

    def set_ollama_models(self, models: List[str]) -> None:
        """Ollama model listesini günceller."""
        if models:
            self.ollama_menu.configure(values=models)
            self.ollama_menu.set(models[0])
        else:
            self.ollama_menu.configure(values=["(model bulunamadı)"])
            self.ollama_menu.set("(model bulunamadı)")

    def set_status(self, text: str, color: str = "gray70") -> None:
        """Alt durum çubuğunu günceller (thread-safe değildir; main 'after' ile çağırmalı)."""
        self.status_label.configure(text=f"● {text}", text_color=color)

    def set_documents(self, sources: List[str]) -> None:
        """Yüklü doküman listesini günceller."""
        self.doc_list.configure(state="normal")
        self.doc_list.delete("1.0", "end")
        if sources:
            for src in sources:
                self.doc_list.insert("end", f"• {src}\n")
        else:
            self.doc_list.insert("end", "Henüz doküman yüklenmedi.")
        self.doc_list.configure(state="disabled")

    def set_controls_enabled(self, enabled: bool) -> None:
        """Uzun işlemler sırasında butonları kilitler/açar."""
        state = "normal" if enabled else "disabled"
        for widget in (
            self.upload_btn, self.upload_folder_btn, self.clear_btn, self.provider_menu
        ):
            widget.configure(state=state)


# ---------------------------------------------------------------------------
#  Sohbet Alanı
# ---------------------------------------------------------------------------
class ChatArea(ctk.CTkFrame):
    """Sohbet geçmişi, soru giriş kutusu ve gönder butonunu içeren ana alan."""

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
            input_frame, text="Gönder ➤", width=110, height=45,
            command=self._handle_send, font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.send_btn.grid(row=0, column=1)

        self._row = 0
        self.add_message(
            "Asistan",
            "Merhaba! Ben gemi teknik doküman asistanınızım. "
            "Sol panelden doküman yükleyip bana sorularınızı sorabilirsiniz.",
            is_user=False,
        )

    # -- İç olaylar -------------------------------------------------------- #
    def _handle_send(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        if self._on_send:
            self._on_send(text)

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
            bubble, text="📋 Kopyala", width=90, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="transparent", border_width=1, hover_color="#4a4d50",
        )
        copy_btn.configure(
            command=lambda lbl=msg_label, btn=copy_btn: self._copy_to_clipboard(lbl, btn)
        )
        copy_btn.pack(anchor="e", padx=12, pady=(0, 8))

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
        # Kısa görsel geri bildirim: "✓ Kopyalandı" -> 1.5 sn sonra eski metne dön.
        try:
            button.configure(text="✓ Kopyalandı")
            self.after(1500, lambda: self._reset_copy_button(button))
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _reset_copy_button(button: ctk.CTkButton) -> None:
        """Kopyala butonunu eski metnine döndürür (widget hâlâ varsa)."""
        try:
            button.configure(text="📋 Kopyala")
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
