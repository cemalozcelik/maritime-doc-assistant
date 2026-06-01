# -*- coding: utf-8 -*-
"""
llm_connector.py
================
RAG hattının "Generation" katmanı. Kullanıcının seçimine göre:
    * Gemini API (internet varken)  -> google-generativeai
    * Ollama (internet yokken)      -> localhost REST API (llama3, mistral, gemma...)

Sorumluluklar (SRP):
    * Sağlayıcılara bağlanma ve cevap üretme.
    * Soru + bağlam parçalarından gemicilik odaklı bir prompt oluşturma.
    * Ollama erişilebilirliğini kontrol etme ve mevcut modelleri listeleme.

Tasarım: Strateji deseni (Strategy Pattern). Her sağlayıcı ortak bir arayüzü
(BaseLLMConnector) uygular; LLMConnector cephesi (facade) doğru stratejiyi seçer.
Bu, Açık/Kapalı Prensibi'ne (OCP) uygundur: yeni sağlayıcı eklemek mevcut kodu bozmaz.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# Ollama varsayılan adresi (lokal).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Gemicilik bağlamına özel sistem yönergesi (system prompt).
SYSTEM_PROMPT = (
    "Sen bir geminin teknik dokümanlarına hakim, kıdemli bir gemi mühendisi "
    "asistanısın. Sana verilen BAĞLAM (doküman alıntıları) bilgisine dayanarak "
    "kullanıcının sorusunu Türkçe, açık ve teknik olarak doğru biçimde yanıtla.\n"
    "Kurallar:\n"
    "1. Cevabı yalnızca verilen bağlama dayandır. Bağlamda yoksa uydurma.\n"
    "2. Bilgi bağlamda yoksa açıkça 'Verilen dokümanlarda bu bilgi bulunmuyor.' de.\n"
    "3. Mümkünse hangi kaynaktan/sayfadan aldığını belirt.\n"
    "4. Güvenlikle ilgili konularda dikkatli ve net ol."
)


# ---------------------------------------------------------------------------
#  Veri Modeli
# ---------------------------------------------------------------------------
class LLMResponse:
    """LLM cevabını ve durum bilgisini taşıyan basit kap."""

    def __init__(self, text: str, success: bool = True, error: str = "") -> None:
        self.text = text
        self.success = success
        self.error = error


# ---------------------------------------------------------------------------
#  Ortak Arayüz (Strateji)
# ---------------------------------------------------------------------------
class BaseLLMConnector(ABC):
    """Tüm LLM sağlayıcılarının uygulaması gereken ortak arayüz."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Verilen tam prompt için bir cevap üretir."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Sağlayıcı şu an kullanılabilir mi?"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  Gemini (Bulut)
# ---------------------------------------------------------------------------
class GeminiConnector(BaseLLMConnector):
    """Google Gemini API bağlantısı (internet gerektirir)."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai kurulu değil. "
                "'pip install google-generativeai' çalıştırın."
            ) from exc

        if not self.api_key:
            raise ValueError("Gemini API anahtarı boş olamaz.")

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def generate(self, prompt: str) -> LLMResponse:
        try:
            model = self._ensure_model()
            response = model.generate_content(prompt)
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return LLMResponse("", success=False, error="Gemini boş cevap döndürdü.")
            return LLMResponse(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini hatası: %s", exc)
            return LLMResponse(
                "",
                success=False,
                error=f"Gemini bağlantı hatası (internet/anahtar kontrol edin): {exc}",
            )

    def is_available(self) -> bool:
        """API anahtarı var mı ve internet erişilebilir mi (basit kontrol)."""
        if not self.api_key:
            return False
        try:
            import socket
            socket.setdefaulttimeout(3)
            socket.create_connection(("generativelanguage.googleapis.com", 443))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
#  Ollama (Lokal)
# ---------------------------------------------------------------------------
class OllamaConnector(BaseLLMConnector):
    """Lokal Ollama sunucusu bağlantısı (internet gerektirmez)."""

    def __init__(self, model_name: str = "llama3", host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")

    def generate(self, prompt: str) -> LLMResponse:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests kurulu değil. 'pip install requests'.") from exc

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,            # Tek seferde tam cevap.
            "options": {"temperature": 0.2},
        }
        try:
            # Lokal LLM cevabı uzun sürebilir -> geniş timeout.
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response", "") or "").strip()
            if not text:
                return LLMResponse("", success=False, error="Ollama boş cevap döndürdü.")
            return LLMResponse(text)
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                "",
                success=False,
                error="Ollama sunucusuna bağlanılamadı. "
                      "Lütfen 'ollama serve' komutunun çalıştığından emin olun.",
            )
        except requests.exceptions.HTTPError as exc:
            return LLMResponse(
                "",
                success=False,
                error=f"Ollama HTTP hatası: {exc}. Model '{self.model_name}' indirilmiş mi? "
                      f"('ollama pull {self.model_name}')",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama hatası: %s", exc)
            return LLMResponse("", success=False, error=f"Ollama hatası: {exc}")

    def is_available(self) -> bool:
        """Ollama sunucusu ayakta mı?"""
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def list_models(self) -> List[str]:
        """Ollama'da indirilmiş mevcut modelleri listeler."""
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama model listesi alınamadı: %s", exc)
            return []


# ---------------------------------------------------------------------------
#  Cephe (Facade) - UI bu sınıfla konuşur
# ---------------------------------------------------------------------------
class LLMConnector:
    """
    UI katmanının kullandığı tek giriş noktası.
    Hangi sağlayıcının aktif olduğunu yönetir ve prompt'u oluşturur.
    """

    PROVIDER_GEMINI = "Gemini (Çevrimiçi)"
    PROVIDER_OLLAMA = "Ollama (Çevrimdışı)"

    def __init__(self) -> None:
        self._connector: Optional[BaseLLMConnector] = None
        self._provider_name: str = ""

    # -- Sağlayıcı seçimi -------------------------------------------------- #
    def use_gemini(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
        self._connector = GeminiConnector(api_key=api_key, model_name=model_name)
        self._provider_name = self.PROVIDER_GEMINI

    def use_ollama(self, model_name: str = "llama3", host: str = DEFAULT_OLLAMA_HOST) -> None:
        self._connector = OllamaConnector(model_name=model_name, host=host)
        self._provider_name = self.PROVIDER_OLLAMA

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return bool(self._connector) and self._connector.is_available()

    # -- Statik yardımcılar (UI'ın seçim yapmasına yardımcı) --------------- #
    @staticmethod
    def check_ollama(host: str = DEFAULT_OLLAMA_HOST) -> bool:
        return OllamaConnector(host=host).is_available()

    @staticmethod
    def get_ollama_models(host: str = DEFAULT_OLLAMA_HOST) -> List[str]:
        return OllamaConnector(host=host).list_models()

    # -- Prompt oluşturma -------------------------------------------------- #
    @staticmethod
    def build_prompt(question: str, contexts: List["object"]) -> str:
        """
        Soru ve retrieval'dan gelen bağlam parçalarından tam prompt'u kurar.

        :param contexts: embedding_manager.RetrievedContext nesneleri
                         (text, source, page alanları beklenir).
        """
        if contexts:
            context_blocks = []
            for i, ctx in enumerate(contexts, start=1):
                source = getattr(ctx, "source", "bilinmiyor")
                page = getattr(ctx, "page", 0)
                location = f"{source}" + (f", sayfa {page}" if page else "")
                context_blocks.append(
                    f"[Kaynak {i} - {location}]\n{getattr(ctx, 'text', '')}"
                )
            context_text = "\n\n".join(context_blocks)
        else:
            context_text = "(İlgili doküman bağlamı bulunamadı.)"

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== BAĞLAM (Doküman Alıntıları) ===\n{context_text}\n\n"
            f"=== KULLANICI SORUSU ===\n{question}\n\n"
            f"=== CEVAP ===\n"
        )

    # -- Cevap üretme ------------------------------------------------------ #
    def ask(self, question: str, contexts: List["object"]) -> LLMResponse:
        """Soru + bağlam ile aktif sağlayıcıdan cevap üretir."""
        if self._connector is None:
            return LLMResponse(
                "", success=False,
                error="Önce bir model sağlayıcı seçin (Gemini veya Ollama).",
            )
        prompt = self.build_prompt(question, contexts)
        return self._connector.generate(prompt)
