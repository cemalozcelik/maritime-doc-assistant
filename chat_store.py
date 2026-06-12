# -*- coding: utf-8 -*-
"""
chat_store.py
=============
Kalıcı sohbet geçmişi deposu. Her sohbet, diske bir JSON dosyası olarak yazılır;
uygulama kapatılıp açıldığında geçmiş sohbetler korunur (Gemini benzeri davranış).

Sorumluluklar (SRP):
    * Sohbet oturumlarını oluşturmak, listelemek, yüklemek, kaydetmek ve silmek.
    * İş mantığı veya arayüzle ilgilenmez; yalnızca kalıcılık katmanıdır.

Oturum şeması (JSON):
    {
      "id": "20260610-153012-ab12",
      "title": "Yakıt seperatörü verim düşüklüğü",
      "created_at": 1718031012.0,
      "updated_at": 1718031120.0,
      "provider": "Yerel Model (Çevrimdışı)",
      "model": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
      "messages": [
        {"role": "user", "text": "...", "ts": 1718031012.0},
        {"role": "assistant", "text": "...", "ts": 1718031020.0}
      ]
    }
"""

from __future__ import annotations

import os
import json
import time
import uuid
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Geçerli roller (UI tarafındaki "Siz"/"Asistan"/"Sistem" ile eşlenir).
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

_TITLE_MAX = 40


class ChatStore:
    """Sohbet oturumlarını bir klasördeki JSON dosyalarında saklar."""

    def __init__(self, base_dir: str) -> None:
        """
        :param base_dir: Sohbet JSON dosyalarının yazılacağı klasör
                         (genelde writable_data_dir()/chats).
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # -- Yol yardımcıları -------------------------------------------------- #
    def _path(self, session_id: str) -> str:
        return os.path.join(self.base_dir, f"{session_id}.json")

    # -- Oluşturma --------------------------------------------------------- #
    def create(
        self,
        provider: str = "",
        model: str = "",
        title: str = "Yeni sohbet",
    ) -> Dict:
        """Boş bir oturum nesnesi döndürür (henüz diske yazmaz)."""
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        session = {
            "id": f"{stamp}-{uuid.uuid4().hex[:4]}",
            "title": title,
            "created_at": now,
            "updated_at": now,
            "provider": provider,
            "model": model,
            "messages": [],
        }
        return session

    # -- Mesaj ekleme ------------------------------------------------------ #
    @staticmethod
    def add_message(session: Dict, role: str, text: str) -> None:
        """Oturuma bir mesaj ekler; başlık boşsa ilk kullanıcı mesajından üretir."""
        session.setdefault("messages", []).append(
            {"role": role, "text": text, "ts": time.time()}
        )
        session["updated_at"] = time.time()
        # İlk anlamlı kullanıcı mesajından otomatik başlık üret.
        if role == ROLE_USER and (
            not session.get("title") or session["title"] in ("", "Yeni sohbet")
        ):
            session["title"] = ChatStore.auto_title(text)

    @staticmethod
    def auto_title(text: str) -> str:
        """Bir metinden kısa, tek satırlık bir başlık üretir."""
        clean = " ".join((text or "").split())
        if not clean:
            return "Yeni sohbet"
        if len(clean) <= _TITLE_MAX:
            return clean
        return clean[:_TITLE_MAX].rstrip() + "..."

    # -- Kaydetme ---------------------------------------------------------- #
    def save(self, session: Dict) -> None:
        """Oturumu diske (atomik olarak) yazar."""
        if not session or not session.get("id"):
            return
        session["updated_at"] = time.time()
        path = self._path(session["id"])
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(session, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Sohbet kaydedilemedi (%s): %s", session.get("id"), exc)

    # -- Yükleme ----------------------------------------------------------- #
    def load(self, session_id: str) -> Optional[Dict]:
        """Tek bir oturumu diskten okur; bulunamazsa None döner."""
        path = self._path(session_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("Sohbet okunamadı (%s): %s", session_id, exc)
            return None

    # -- Listeleme --------------------------------------------------------- #
    def list_sessions(self) -> List[Dict]:
        """
        Tüm oturumların özetini (mesaj içeriği olmadan) en yeni güncellenen
        en üstte olacak şekilde döndürür: {id, title, updated_at, message_count}.
        """
        out: List[Dict] = []
        try:
            names = [n for n in os.listdir(self.base_dir) if n.endswith(".json")]
        except OSError:
            return out
        for name in names:
            path = os.path.join(self.base_dir, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                continue
            out.append({
                "id": data.get("id") or name[:-5],
                "title": data.get("title") or "Yeni sohbet",
                "updated_at": data.get("updated_at") or data.get("created_at") or 0,
                "message_count": len(data.get("messages") or []),
            })
        out.sort(key=lambda s: s.get("updated_at") or 0, reverse=True)
        return out

    # -- Silme ------------------------------------------------------------- #
    def delete(self, session_id: str) -> None:
        """Bir oturumu diskten siler (yoksa sessizce geçer)."""
        path = self._path(session_id)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            logger.warning("Sohbet silinemedi (%s): %s", session_id, exc)

    @staticmethod
    def is_empty(session: Optional[Dict]) -> bool:
        """Oturum hiç mesaj içermiyorsa True (boş 'Yeni sohbet' tekrar tekrar
        diske yazılmasın / listede birikmesin diye kullanılır)."""
        return not session or not (session.get("messages") or [])
