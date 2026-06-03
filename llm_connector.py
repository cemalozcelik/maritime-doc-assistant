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
    "asistanısın. Sana BAĞLAM (doküman alıntıları) ve bir SORU verilir. Türkçe, "
    "açık ve teknik olarak doğru cevap ver.\n\n"
    "CEVAPLAMA YÖNTEMİ (etiketli hibrit):\n"
    "1. ÖNCE bağlamdaki (dokümanlardaki) bilgiyi kullan. Bu bölümü\n"
    "   '📄 Dokümandan:' başlığı altında ver ve hangi kaynaktan/sayfadan "
    "geldiğini belirt.\n"
    "2. Bağlam soruyu tam karşılamıyorsa, genel gemi mühendisliği bilginle "
    "TAMAMLAYABİLİRSİN; ancak bu bölümü MUTLAKA ayrı olarak\n"
    "   '💡 Genel mühendislik bilgisi (dokümanda doğrulanmadı):' başlığı altında "
    "ver. Kullanıcı neyin kaynaklı, neyin genel bilgi olduğunu net görmeli.\n"
    "3. ÇOK ÖNEMLİ: Spesifik sayı, ölçü, basınç/sıcaklık değeri, tork değeri "
    "veya parça numarasını ASLA uydurma. Bu tür bir değer bağlamda yoksa, "
    "'kesin değer için ilgili manueli kontrol edin' de.\n"
    "4. Ne bağlamda ne de genel bilginde hiçbir şey yoksa, bunu açıkça söyle.\n"
    "5. Mümkün olduğunda maddeler ve gerekiyorsa adım adım açıkla. "
    "Güvenlikle ilgili konularda dikkatli ve net ol.\n"
    "Not: Bağlam ilgili bilgi içeriyorsa '📄 Dokümandan' bölümü esas; "
    "'💡 Genel mühendislik bilgisi' bölümünü yalnızca gerçekten ekleyecek bir "
    "şey varsa kullan."
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

    # Tercih sırası: güncel ve hızlı 'flash' modelleri önce. Belirtilen model
    # API'de yoksa (örn. eski 'gemini-1.5-flash' kullanımdan kalktı), bu listeden
    # ve API'nin döndürdüğü mevcut modellerden otomatik uygun bir model seçilir.
    PREFERRED_MODELS = (
        "gemini-2.5-pro",
        "gemini-pro-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
    )

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def _genai(self):
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
        return genai

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        genai = self._genai()
        self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def _list_supported_models(self) -> List[str]:
        """API'de 'generateContent' destekleyen modelleri listeler (kısa adlarıyla)."""
        genai = self._genai()
        names: List[str] = []
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                # "models/gemini-2.5-flash" -> "gemini-2.5-flash"
                names.append(getattr(m, "name", "").split("/")[-1])
        return [n for n in names if n]

    def _resolve_available_model(self) -> Optional[str]:
        """
        Mevcut modeller arasından uygun bir model seçer:
        önce tercih listesindekiler, sonra herhangi bir 'flash', en sonda
        generateContent destekleyen ilk model.
        """
        try:
            available = set(self._list_supported_models())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini model listesi alınamadı: %s", exc)
            return None
        if not available:
            return None
        for name in self.PREFERRED_MODELS:
            if name in available:
                return name
        for name in sorted(available):
            if "flash" in name:
                return name
        return sorted(available)[0]

    def generate(self, prompt: str) -> LLMResponse:
        try:
            model = self._ensure_model()
            response = model.generate_content(prompt)
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return LLMResponse("", success=False, error="Gemini boş cevap döndürdü.")
            return LLMResponse(text)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # Model adı geçersiz/desteklenmiyorsa: API'den uygun model bul ve bir kez yeniden dene.
            if "404" in msg or "not found" in msg.lower() or "not supported" in msg.lower():
                resolved = self._resolve_available_model()
                if resolved and resolved != self.model_name:
                    logger.info("Gemini modeli '%s' kullanılamadı; '%s' ile yeniden deneniyor.",
                                self.model_name, resolved)
                    self.model_name = resolved
                    self._model = None
                    try:
                        model = self._ensure_model()
                        response = model.generate_content(prompt)
                        text = (getattr(response, "text", "") or "").strip()
                        if text:
                            return LLMResponse(text)
                    except Exception as exc2:  # noqa: BLE001
                        exc = exc2
                        msg = str(exc2)
            logger.error("Gemini hatası: %s", exc)
            return LLMResponse(
                "",
                success=False,
                error=f"Gemini bağlantı hatası (internet/anahtar/model kontrol edin): {exc}",
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

    # Streaming kullandığımız için bu, "token'lar arası izin verilen en uzun
    # boşluk" (read timeout) anlamına gelir; toplam cevap süresi değil. İlk
    # token, büyük bağlamın işlenmesi nedeniyle yavaş donanımda dakikalar
    # sürebileceğinden geniş tutulur. Token akışı başladıktan sonra her token
    # bu süreyi sıfırlar; böylece uzun cevaplar timeout'a takılmaz.
    DEFAULT_TIMEOUT = 600  # saniye (token'lar arası azami bekleme)
    CONNECT_TIMEOUT = 10   # saniye (sunucuya bağlanma)

    def __init__(
        self,
        model_name: str = "llama3",
        host: str = DEFAULT_OLLAMA_HOST,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> LLMResponse:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests kurulu değil. 'pip install requests'.") from exc

        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,             # Akışlı: token geldikçe oku.
            "options": {"temperature": 0.2},
        }
        try:
            import json as _json

            # timeout=(bağlanma, okuma). 'stream=True' ile okuma zaman aşımı her
            # token'da sıfırlanır; tek bir token > self.timeout sürmedikçe takılmaz.
            resp = requests.post(
                url, json=payload, stream=True,
                timeout=(self.CONNECT_TIMEOUT, self.timeout),
            )
            resp.raise_for_status()

            parts: List[str] = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except ValueError:
                    continue  # Bozuk/yarım satırı atla.
                parts.append(chunk.get("response", "") or "")
                if chunk.get("done"):
                    break
            text = "".join(parts).strip()
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
    def use_gemini(self, api_key: str, model_name: str = "gemini-2.5-pro") -> None:
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
