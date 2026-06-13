# -*- coding: utf-8 -*-
"""
llm_connector.py
================
RAG hattının "Generation" katmanı. Kullanıcının seçimine göre:
    * Gemini API (internet varken)  -> google-genai (yeni Google Gen AI SDK)
    * Yerel Model (internet yokken) -> gömülü llama.cpp motoru (llama-cpp-python),
      arayüzden indirilmiş GGUF dosyalarını çalıştırır. Harici program gerekmez.

Sorumluluklar (SRP):
    * Sağlayıcılara bağlanma ve cevap üretme.
    * Soru + bağlam parçalarından gemicilik odaklı bir prompt oluşturma.
    * Yerel motoru (GGUF) yükleme ve cevap üretme.

Tasarım: Strateji deseni (Strategy Pattern). Her sağlayıcı ortak bir arayüzü
(BaseLLMConnector) uygular; LLMConnector cephesi (facade) doğru stratejiyi seçer.
Bu, Açık/Kapalı Prensibi'ne (OCP) uygundur: yeni sağlayıcı eklemek mevcut kodu bozmaz.
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# Gömülü motor (llama-cpp-python) CUDA wheel'i, CUDA runtime DLL'lerine (cudart,
# cublas, cublasLt) ihtiyaç duyar. Bu DLL'ler torch'un CUDA sürümüyle 'torch/lib'
# içinde gelir (önerilen kurulum); torch CPU kullanılıyorsa 'nvidia-*-cu12' pip
# paketlerinden de (nvidia/cublas/bin, nvidia/cuda_runtime/bin) gelebilir. Yardımcı
# her iki yeri de arar. llama_cpp import EDİLMEDEN ÖNCE bu klasörler Windows'un DLL
# arama yoluna eklenir ve DLL'ler açıkça önceden yüklenir; aksi halde 'Could not
# find module llama.dll (or one of its dependencies)' hatası alınır. Hiçbiri yoksa
# (saf CPU llama wheel'i) bu adım zararsızca atlanır.
_LLAMA_DLL_READY = False


def _ensure_llama_loadable() -> None:
    """llama_cpp import edilebilmesi için CUDA DLL arama yolunu hazırlar (Windows)."""
    global _LLAMA_DLL_READY
    if _LLAMA_DLL_READY:
        return
    _LLAMA_DLL_READY = True
    if os.name != "nt":
        return

    candidates: List[str] = []

    # 0) Paketlenmiş (.exe / PyInstaller) ortamda nvidia DLL'leri _MEIPASS altında
    #    'nvidia/<paket>/bin' yapısında bulunur; namespace paket import'u frozen'da
    #    güvenilmez olduğu için yolu doğrudan _MEIPASS'tan kurarız.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        for sub in ("cuda_runtime", "cublas"):
            candidates.append(os.path.join(meipass, "nvidia", sub, "bin"))

    # 1) nvidia-cublas-cu12 / nvidia-cuda-runtime-cu12 pip paketleri (geliştirme ortamı).
    try:
        import nvidia  # type: ignore
        nvidia_root = os.path.dirname(nvidia.__file__)
        for sub in ("cuda_runtime", "cublas"):
            candidates.append(os.path.join(nvidia_root, sub, "bin"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-* DLL paketleri bulunamadı: %s", exc)

    # 2) torch CUDA sürümü kuruluysa 'torch/lib' (yedek).
    try:
        import torch
        candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch/lib DLL yolu eklenemedi: %s", exc)

    dirs = [p for p in candidates if os.path.isdir(p)]
    for path in dirs:
        try:
            os.add_dll_directory(path)
        except OSError as exc:
            logger.debug("DLL yolu eklenemedi (%s): %s", path, exc)

    # ggml-cuda.dll'in muhtaç olduğu CUDA runtime DLL'lerini AÇIKÇA önceden yükle.
    # Yalnızca klasörü arama yoluna eklemek bazı ortamlarda yetmiyor; torch'un
    # CUDA sürümü 'import torch' ile bunları zaten belleğe alıyordu. CPU torch'ta
    # bu adımı kendimiz yaparız. Sıra önemli: cublas, cublasLt'ye bağımlıdır.
    if dirs:
        import ctypes
        for dll in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
            for path in dirs:
                full = os.path.join(path, dll)
                if os.path.isfile(full):
                    try:
                        ctypes.WinDLL(full)
                    except OSError as exc:
                        logger.debug("CUDA DLL önyüklenemedi (%s): %s", dll, exc)
                    break
    else:
        logger.debug("CUDA DLL yolu bulunamadı; CPU llama wheel'i varsayılıyor.")

# Gemicilik bağlamına özel sistem yönergesi (system prompt).
SYSTEM_PROMPT = (
    "Sen bir geminin teknik dokümanlarına hakim, kıdemli bir gemi mühendisi "
    "asistanısın. Sana BAĞLAM (doküman alıntıları) ve bir SORU verilir. Türkçe, "
    "açık ve teknik olarak doğru cevap ver.\n\n"
    "CEVAPLAMA YÖNTEMİ (etiketli hibrit):\n"
    "1. ÖNCE bağlamdaki (dokümanlardaki) bilgiyi kullan. Bu bölümü\n"
    "   'DOKÜMANDAN:' başlığı altında ver ve hangi kaynaktan/sayfadan "
    "geldiğini belirt.\n"
    "2. Bağlam soruyu tam karşılamıyorsa, genel gemi mühendisliği bilginle "
    "TAMAMLAYABİLİRSİN; ancak bu bölümü MUTLAKA ayrı olarak\n"
    "   'GENEL MÜHENDİSLİK BİLGİSİ (dokümanda doğrulanmadı):' başlığı altında "
    "ver. Kullanıcı neyin kaynaklı, neyin genel bilgi olduğunu net görmeli.\n"
    "3. ÇOK ÖNEMLİ: Spesifik sayı, ölçü, basınç/sıcaklık değeri, tork değeri "
    "veya parça numarasını ASLA uydurma. Bu tür bir değer bağlamda yoksa, "
    "'kesin değer için ilgili manueli kontrol edin' de.\n"
    "4. Ne bağlamda ne de genel bilginde hiçbir şey yoksa, bunu açıkça söyle.\n"
    "4b. KONU DOĞRULAMA (ÇOK ÖNEMLİ): Sorunun ANA KONUSU/NESNESİ (örn. RADAR, "
    "TÜRBİN, PERVANE) bağlamdaki dokümanlarda GEÇMİYORSA; bağlamda gevşek ilişkili "
    "başka içerik (örn. BAŞKA bir cihazın kalibrasyonu, başka bir sistemin basıncı) "
    "bulunsa BİLE, önce net olarak 'Bu konu yüklenen dokümanlarda bulunmuyor.' de. "
    "Alakasız bağlamı soruyu yanıtlıyormuş gibi SUNMA, ondan değer/prosedür uydurma. "
    "İstersen ardından 'GENEL MÜHENDİSLİK BİLGİSİ (dokümanda doğrulanmadı):' "
    "başlığıyla genel bilgi verebilirsin; ama bunun dokümana dayanmadığını belirt.\n"
    "5. Mümkün olduğunda maddeler ve gerekiyorsa adım adım açıkla. "
    "Güvenlikle ilgili konularda dikkatli ve net ol.\n"
    "6. BİÇİM: DÜZ METİN yaz. Markdown işaretlerini (**kalın**, #, ##, ###, "
    "* madde işareti) ve LaTeX ($...$, \\rightarrow gibi) KULLANMA; çünkü arayüz "
    "bunları olduğu gibi ham gösterir. Bunun yerine: başlıkları numarayla ve "
    "büyük harfle yaz (örn. '1) NEDENLER'), alt maddeleri '• ' ile başlat, "
    "vurgu için tırnak veya BÜYÜK HARF kullan.\n"
    "Not: Bağlam ilgili bilgi içeriyorsa 'DOKÜMANDAN' bölümü esas; "
    "'GENEL MÜHENDİSLİK BİLGİSİ' bölümünü yalnızca gerçekten ekleyecek bir "
    "şey varsa kullan."
)


# ---------------------------------------------------------------------------
#  Veri Modeli
# ---------------------------------------------------------------------------
class LLMResponse:
    """LLM cevabını ve durum bilgisini taşıyan basit kap."""

    def __init__(
        self,
        text: str,
        success: bool = True,
        error: str = "",
        meta: Optional[dict] = None,
    ) -> None:
        self.text = text
        self.success = success
        self.error = error
        # Ölçüm/benchmark için ek bilgi: token sayıları, süreler vb.
        # (ör. {'input_tokens': 3120, 'output_tokens': 740, 'eval_tps': 41.2})
        self.meta = meta or {}


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
        self._client = None

    def _get_client(self):
        """Yeni Google Gen AI SDK istemcisini (google.genai) hazırlar."""
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai kurulu değil. 'pip install google-genai' çalıştırın. "
                "(Eski 'google-generativeai' paketi kullanımdan kaldırıldı.)"
            ) from exc
        if not self.api_key:
            raise ValueError("Gemini API anahtarı boş olamaz.")
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _list_supported_models(self) -> List[str]:
        """API'de 'generateContent' destekleyen modelleri listeler (kısa adlarıyla)."""
        client = self._get_client()
        names: List[str] = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions:
                # "models/gemini-2.5-pro" -> "gemini-2.5-pro"
                names.append((getattr(m, "name", "") or "").split("/")[-1])
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
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return LLMResponse("", success=False, error="Gemini boş cevap döndürdü.")
            return LLMResponse(text, meta=self._build_meta(response))
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # Model adı geçersiz/desteklenmiyorsa: API'den uygun model bul ve bir kez yeniden dene.
            if "404" in msg or "not found" in msg.lower() or "not supported" in msg.lower():
                resolved = self._resolve_available_model()
                if resolved and resolved != self.model_name:
                    logger.info("Gemini modeli '%s' kullanılamadı; '%s' ile yeniden deneniyor.",
                                self.model_name, resolved)
                    self.model_name = resolved
                    try:
                        client = self._get_client()
                        response = client.models.generate_content(
                            model=self.model_name, contents=prompt
                        )
                        text = (getattr(response, "text", "") or "").strip()
                        if text:
                            return LLMResponse(text, meta=self._build_meta(response))
                    except Exception as exc2:  # noqa: BLE001
                        exc = exc2
                        msg = str(exc2)
            logger.error("Gemini hatası: %s", exc)
            return LLMResponse(
                "",
                success=False,
                error=f"Gemini bağlantı hatası (internet/anahtar/model kontrol edin): {exc}",
            )

    def _build_meta(self, response) -> dict:
        """Gemini yanıtının usage_metadata'sından token sayılarını çıkarır."""
        meta = {"provider": "gemini", "model": self.model_name}
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            in_tok = getattr(usage, "prompt_token_count", None)
            out_tok = getattr(usage, "candidates_token_count", None)
            meta.update({
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": getattr(usage, "total_token_count", None),
                # 'Thinking' modellerinde düşünme token'ları (varsa).
                "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
            })
        return meta

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
#  Yerel Model (gömülü llama.cpp / GGUF)
# ---------------------------------------------------------------------------
class LocalLLMConnector(BaseLLMConnector):
    """
    Gömülü llama.cpp motoruyla (llama-cpp-python) yerel bir GGUF modelini
    çalıştırır. Harici bir program (Ollama vb.) gerektirmez; model dosyası
    bir kez indirildikten sonra tamamen çevrimdışı çalışır.

    Yüklenen model örnekte (RAM/VRAM) tutulur; aynı bağlayıcı yeniden
    kullanıldığında model tekrar yüklenmez.
    """

    # Bağlam penceresi: RAG promptu (sistem + 8 bağlam parçası + soru) sığmalı.
    DEFAULT_N_CTX = 8192
    MAX_OUTPUT_TOKENS = 2048

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = DEFAULT_N_CTX,
    ) -> None:
        """
        :param model_path: GGUF dosyasının tam yolu.
        :param n_gpu_layers: GPU'ya boşaltılacak katman sayısı (-1: hepsi).
            CPU-only kurulumda yok sayılır ve model CPU'da çalışır.
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm = None  # Tembel yüklenen Llama örneği (cache).

    def _get_llm(self):
        """Llama modelini bir kez yükler ve önbelleğe alır."""
        if self._llm is not None:
            return self._llm
        _ensure_llama_loadable()
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python kurulu değil. 'pip install llama-cpp-python' "
                "çalıştırın (GPU için CUDA wheel'i gerekir)."
            ) from exc
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {self.model_path}. "
                "Modeli 'Modeller' sekmesinden indirin."
            )
        logger.info("Yerel model yükleniyor: %s (n_gpu_layers=%s)",
                    self.model_path, self.n_gpu_layers)
        # flash_attn=True: dikkat (attention) çekirdeğini hızlandırır VE KV cache
        # belleğini küçültür -> 8 GB gibi sınırlı VRAM'de daha az taşma, daha hızlı.
        # Eski/uyumsuz wheel'lerde param desteklenmezse parametresiz yüklemeye düşer.
        common = dict(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                      n_ctx=self.n_ctx, verbose=False)
        try:
            self._llm = Llama(flash_attn=True, n_batch=512, **common)
        except Exception as exc:  # noqa: BLE001
            logger.warning("flash_attn ile yükleme başarısız (%s); standart yükleme.", exc)
            self._llm = Llama(**common)
        return self._llm

    def generate(self, prompt: str) -> LLMResponse:
        try:
            llm = self._get_llm()
        except (ImportError, FileNotFoundError) as exc:
            return LLMResponse("", success=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.error("Yerel model yüklenemedi: %s", exc)
            return LLMResponse(
                "", success=False, error=f"Yerel model yüklenemedi: {exc}"
            )

        try:
            t0 = time.perf_counter()
            # GGUF içindeki sohbet şablonu otomatik uygulanır (create_chat_completion).
            stream = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.3,
                top_p=0.9,
                top_k=40,
                # Küçük modellerin düştüğü tekrar (kelime/cümle döngüsü)
                # sorununu engellemek için ceza terimleri.
                repeat_penalty=1.18,
                frequency_penalty=0.4,
                presence_penalty=0.3,
                max_tokens=self.MAX_OUTPUT_TOKENS,
            )
            parts: List[str] = []
            for chunk in stream:
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    parts.append(piece)
            elapsed = time.perf_counter() - t0

            text = "".join(parts).strip()
            if not text:
                return LLMResponse(
                    "", success=False, error="Yerel model boş cevap döndürdü."
                )
            return LLMResponse(text, meta=self._build_meta(llm, prompt, text, elapsed))
        except Exception as exc:  # noqa: BLE001
            logger.error("Yerel model hatası: %s", exc)
            return LLMResponse("", success=False, error=f"Yerel model hatası: {exc}")

    def _build_meta(self, llm, prompt: str, text: str, elapsed: float) -> dict:
        """
        Performans bloğu için ölçüm verisi üretir. Token sayıları, modelin
        tokenizer'ı ile yaklaşık hesaplanır (akışta token sayacı verilmez).
        """
        filename = os.path.basename(self.model_path)
        meta = {"provider": "local", "model": filename}
        meta.update(self._parse_name(filename))

        in_tok = out_tok = None
        try:
            in_tok = len(llm.tokenize(prompt.encode("utf-8")))
            out_tok = len(llm.tokenize(text.encode("utf-8")))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Token sayımı yapılamadı: %s", exc)

        meta.update({
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": (in_tok + out_tok) if (in_tok and out_tok) else None,
            "eval_s": round(elapsed, 2),
            "total_s": round(elapsed, 2),
        })
        if out_tok and elapsed > 0:
            meta["output_tps"] = round(out_tok / elapsed, 2)
        return meta

    @staticmethod
    def _parse_name(filename: str) -> dict:
        """Dosya adından parametre sayısı ve quantization çıkarır (varsa)."""
        out: dict = {}
        params = re.search(r"(\d+(?:\.\d+)?[Bb])(?:[-_.]|$)", filename)
        if params:
            out["parameters"] = params.group(1).upper()
        quant = re.search(r"(Q\d[\w]*|F16|BF16|F32)", filename, re.IGNORECASE)
        if quant:
            out["quantization"] = quant.group(1).upper()
        return out

    def is_available(self) -> bool:
        """llama-cpp-python import edilebiliyor ve model dosyası mevcut mu?"""
        _ensure_llama_loadable()
        try:
            import llama_cpp  # noqa: F401
        except (ImportError, OSError):
            return False
        return bool(self.model_path) and os.path.isfile(self.model_path)


# ---------------------------------------------------------------------------
#  Cephe (Facade) - UI bu sınıfla konuşur
# ---------------------------------------------------------------------------
class LLMConnector:
    """
    UI katmanının kullandığı tek giriş noktası.
    Hangi sağlayıcının aktif olduğunu yönetir ve prompt'u oluşturur.
    """

    PROVIDER_GEMINI = "Gemini (Çevrimiçi)"
    PROVIDER_LOCAL = "Yerel Model (Çevrimdışı)"

    def __init__(self) -> None:
        self._connector: Optional[BaseLLMConnector] = None
        self._provider_name: str = ""
        # Aynı yerel model tekrar seçilirse motoru yeniden yüklememek için cache.
        self._local_path: str = ""

    # -- Sağlayıcı seçimi -------------------------------------------------- #
    def use_gemini(self, api_key: str, model_name: str = "gemini-2.5-pro") -> None:
        self._connector = GeminiConnector(api_key=api_key, model_name=model_name)
        self._provider_name = self.PROVIDER_GEMINI

    def use_local(self, model_path: str, n_gpu_layers: int = -1) -> None:
        """Gömülü llama.cpp motoruyla yerel bir GGUF modelini kullanır."""
        # Aynı model yolu zaten yüklüyse, yüklü motoru (RAM/VRAM) koru.
        if (
            isinstance(self._connector, LocalLLMConnector)
            and self._local_path == model_path
        ):
            self._provider_name = self.PROVIDER_LOCAL
            return
        self._connector = LocalLLMConnector(
            model_path=model_path, n_gpu_layers=n_gpu_layers
        )
        self._local_path = model_path
        self._provider_name = self.PROVIDER_LOCAL

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return bool(self._connector) and self._connector.is_available()

    # -- Statik yardımcılar (UI'ın seçim yapmasına yardımcı) --------------- #
    @staticmethod
    def is_local_engine_available() -> bool:
        """Gömülü motor (llama-cpp-python) kurulu ve yüklenebilir mi?"""
        _ensure_llama_loadable()
        try:
            import llama_cpp  # noqa: F401
            return True
        except (ImportError, OSError):
            return False

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

    def translate_to_english(self, text: str) -> str:
        """
        Türkçe sorguyu, çapraz-dil retrieval için İngilizce'ye çevirir (kısa üretim).
        Korpus ağırlıklı İngilizce olduğundan İngilizce varyant retrieval'i belirgin
        iyileştirir. Başarısızlıkta boş string döner (çağıran orijinali kullanır).
        """
        if self._connector is None or not (text or "").strip():
            return ""
        prompt = (
            "Translate the following Turkish marine-engineering question into English "
            "for a technical document search. Keep technical terms accurate. Output "
            "ONLY the English translation on a single line, with no quotes or "
            "explanation.\n\nTurkish: " + text.strip() + "\nEnglish:"
        )
        try:
            resp = self._connector.generate(prompt)
            if not getattr(resp, "success", False):
                return ""
            lines = [ln.strip() for ln in (resp.text or "").splitlines() if ln.strip()]
            out = lines[0] if lines else ""
            # Olası "English:" tekrarını ve tırnakları temizle.
            out = re.sub(r'^(english|ingilizce)\s*[:\-]\s*', '', out, flags=re.IGNORECASE)
            return out.strip().strip('"').strip()[:300]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Çeviri başarısız: %s", exc)
            return ""
