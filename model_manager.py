# -*- coding: utf-8 -*-
"""
model_manager.py
================
Yerel (gömülü) dil modeli dosyalarının (GGUF) yönetimi: arayüzden indirme,
listeleme ve silme. Harici bir program (Ollama vb.) gerektirmez; modeller
doğrudan Hugging Face'ten indirilir ve uygulamanın yazılabilir veri klasörüne
konur. Bir kez indirildikten sonra tamamen çevrimdışı kullanılır.

Sorumluluklar (SRP):
    * İndirilebilir model kataloğunu (curated) sunmak.
    * İndirilmiş modelleri listelemek ve silmek.
    * Bir GGUF dosyasını ilerleme ve iptal desteğiyle indirmek.

Not: İndirme, Hugging Face'in "resolve" URL'inden requests ile akışlı yapılır;
public repolar için kimlik doğrulama (token) gerekmez. requests zaten projenin
bağımlılıklarında mevcuttur.
"""

from __future__ import annotations

import os
import re
import logging
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# İndirilebilir önerilen modeller (instruct GGUF, Q4_K_M). Küçükten büyüğe sıralı.
# repo_id + filename Hugging Face'teki GERÇEK dosya yollarıdır (API ile doğrulandı).
# approx_mb gerçek dosya boyutu; min_ram önerilen boş RAM (GPU'ya tam sığması için
# yaklaşık aynı VRAM gerekir). Bu liste yalnızca "hızlı seçim"dir; tüm güncel
# modellere 'İndirilenler' sekmesindeki arama/manuel giriş ile erişilebilir.
# NOT: Gömülü motor llama-cpp-python 0.3.4'tür. Qwen3 (yeni mimari) ve phi-4
# (sliding-window attention) bu sürümle YÜKLENMEZ; phi-4 ayrıca süreci abort ile
# çökertir. Bu yüzden curated listeden çıkarıldılar. Motor yükseltilirse geri eklenebilir.
CURATED_MODELS: List[Dict] = [
    {
        "key": "qwen2.5-0.5b",
        "label": "Qwen2.5 0.5B Instruct (en küçük)",
        "repo_id": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "approx_mb": 379, "params": "0.5B", "quant": "Q4_K_M", "min_ram": "2 GB",
        "note": "Çok düşük donanım/deneme için. Kalite sınırlı.",
    },
    {
        "key": "llama3.2-1b",
        "label": "Llama 3.2 1B Instruct",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "approx_mb": 770, "params": "1B", "quant": "Q4_K_M", "min_ram": "2 GB",
        "note": "Çok hızlı, hafif. Basit görevler için.",
    },
    {
        "key": "qwen2.5-1.5b",
        "label": "Qwen2.5 1.5B Instruct (hızlı)",
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "approx_mb": 940, "params": "1.5B", "quant": "Q4_K_M", "min_ram": "3 GB",
        "note": "Düşük CPU için. Hızlı ama teknik akıl yürütmede zayıf.",
    },
    {
        "key": "gemma2-2b",
        "label": "Gemma 2 2B Instruct",
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "approx_mb": 1629, "params": "2B", "quant": "Q4_K_M", "min_ram": "4 GB",
        "note": "Küçük ama Türkçede iyi. CPU'da çalışır.",
    },
    {
        "key": "qwen2.5-3b",
        "label": "Qwen2.5 3B Instruct (dengeli)",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "approx_mb": 1840, "params": "3B", "quant": "Q4_K_M", "min_ram": "4 GB",
        "note": "CPU için iyi denge. Çoğu teknik soru için yeterli.",
    },
    {
        "key": "llama3.2-3b",
        "label": "Llama 3.2 3B Instruct",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "approx_mb": 1926, "params": "3B", "quant": "Q4_K_M", "min_ram": "4 GB",
        "note": "Dengeli, genel amaçlı küçük model.",
    },
    {
        "key": "phi3.5-mini",
        "label": "Phi-3.5 mini Instruct (3.8B)",
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "approx_mb": 2282, "params": "3.8B", "quant": "Q4_K_M", "min_ram": "5 GB",
        "note": "Akıl yürütmede güçlü; Türkçesi orta.",
    },
    {
        "key": "mistral-7b-v03",
        "label": "Mistral 7B Instruct v0.3",
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "approx_mb": 4170, "params": "7B", "quant": "Q4_K_M", "min_ram": "7 GB",
        "note": "Hızlı ve sağlam genel amaçlı model.",
    },
    {
        "key": "qwen2.5-7b",
        "label": "Qwen2.5 7B Instruct (önerilen)",
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "approx_mb": 4466, "params": "7B", "quant": "Q4_K_M", "min_ram": "8 GB",
        "note": "8 GB GPU için en iyi denge. Türkçe/teknikte güçlü.",
    },
    {
        "key": "ministral-8b",
        "label": "Ministral 8B Instruct (2410)",
        "repo_id": "bartowski/Ministral-8B-Instruct-2410-GGUF",
        "filename": "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
        "approx_mb": 4684, "params": "8B", "quant": "Q4_K_M", "min_ram": "8 GB",
        "note": "Mistral'in güncel 8B modeli.",
    },
    {
        "key": "llama3.1-8b",
        "label": "Llama 3.1 8B Instruct",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "approx_mb": 4693, "params": "8B", "quant": "Q4_K_M", "min_ram": "8 GB",
        "note": "Genel amaçlı güçlü model.",
    },
    {
        "key": "gemma2-9b",
        "label": "Gemma 2 9B Instruct",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "approx_mb": 5494, "params": "9B", "quant": "Q4_K_M", "min_ram": "9 GB",
        "note": "Türkçede çok iyi; 8 GB GPU'da kısmen CPU'ya taşar.",
    },
    {
        "key": "qwen2.5-14b",
        "label": "Qwen2.5 14B Instruct",
        "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "approx_mb": 8572, "params": "14B", "quant": "Q4_K_M", "min_ram": "12 GB",
        "note": "Yüksek kalite. 12 GB+ VRAM veya bol RAM gerekir.",
    },
    {
        "key": "mistral-small-22b",
        "label": "Mistral Small Instruct (22B, 2409)",
        "repo_id": "bartowski/Mistral-Small-Instruct-2409-GGUF",
        "filename": "Mistral-Small-Instruct-2409-Q4_K_M.gguf",
        "approx_mb": 12698, "params": "22B", "quant": "Q4_K_M", "min_ram": "16 GB",
        "note": "Kalite/boyut dengesi iyi. 16 GB+ gerekir.",
    },
    {
        "key": "gemma2-27b",
        "label": "Gemma 2 27B Instruct",
        "repo_id": "bartowski/gemma-2-27b-it-GGUF",
        "filename": "gemma-2-27b-it-Q4_K_M.gguf",
        "approx_mb": 15874, "params": "27B", "quant": "Q4_K_M", "min_ram": "20 GB",
        "note": "Çok kaliteli ama ağır. 20 GB+ VRAM/RAM gerekir.",
    },
    {
        "key": "qwen2.5-32b",
        "label": "Qwen2.5 32B Instruct",
        "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        "approx_mb": 18932, "params": "32B", "quant": "Q4_K_M", "min_ram": "24 GB",
        "note": "Yüksek kalite. Güçlü iş istasyonu gerektirir.",
    },
    {
        "key": "mixtral-8x7b",
        "label": "Mixtral 8x7B Instruct v0.1 (MoE)",
        "repo_id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "filename": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "approx_mb": 25190, "params": "47B (13B aktif)", "quant": "Q4_K_M", "min_ram": "30 GB",
        "note": "Uzman karışımı: yüksek kalite, aktif parça az olduğu için makul hız. 30 GB+ gerekir.",
    },
    {
        "key": "llama3.3-70b",
        "label": "Llama 3.3 70B Instruct (en güçlü)",
        "repo_id": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "filename": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        "approx_mb": 40550, "params": "70B", "quant": "Q4_K_M", "min_ram": "48 GB",
        "note": "Üst düzey kalite. 128 GB RAM'de CPU ile çalışır (yavaş). ~48 GB gerekir.",
    },
    {
        "key": "qwen2.5-72b",
        "label": "Qwen2.5 72B Instruct (en güçlü)",
        "repo_id": "bartowski/Qwen2.5-72B-Instruct-GGUF",
        "filename": "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        "approx_mb": 45261, "params": "72B", "quant": "Q4_K_M", "min_ram": "52 GB",
        "note": "En yüksek kalite. 128 GB RAM'de CPU ile çalışır (yavaş). ~52 GB gerekir.",
    },
]

_CHUNK = 1024 * 1024  # 1 MB


class ModelManager:
    """GGUF model dosyalarını indirir, listeler ve siler."""

    def __init__(self, models_dir: str) -> None:
        """
        :param models_dir: GGUF dosyalarının saklanacağı klasör
                           (genelde writable_data_dir()/models_gguf).
        """
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    # -- Katalog ----------------------------------------------------------- #
    @staticmethod
    def curated() -> List[Dict]:
        """İndirilebilir önerilen model listesini döndürür."""
        return list(CURATED_MODELS)

    # -- İndirilmiş modeller ---------------------------------------------- #
    def list_downloaded(self) -> List[str]:
        """models_dir içindeki .gguf dosya adlarını döndürür (sıralı)."""
        try:
            files = [
                n for n in os.listdir(self.models_dir)
                if n.lower().endswith(".gguf")
            ]
        except OSError:
            return []
        return sorted(files)

    def path_of(self, filename: str) -> str:
        """Bir model dosya adının tam yolunu döndürür."""
        return os.path.join(self.models_dir, filename)

    def is_downloaded(self, filename: str) -> bool:
        return os.path.isfile(self.path_of(filename))

    def delete(self, filename: str) -> bool:
        """İndirilmiş bir modeli siler. Başarılıysa True döner."""
        path = self.path_of(filename)
        try:
            if os.path.isfile(path):
                os.remove(path)
                return True
        except OSError as exc:
            logger.warning("Model silinemedi (%s): %s", filename, exc)
        return False

    # -- İndirme ----------------------------------------------------------- #
    def download(
        self,
        repo_id: str,
        filename: str,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Bir GGUF dosyasını Hugging Face'ten indirir.

        :param progress_cb: (indirilen_bayt, toplam_bayt) ile periyodik çağrılır.
                            toplam_bayt bilinmiyorsa 0 verilir.
        :param cancel_event: Set edilirse indirme iptal edilir (yarım dosya silinir).
        :return: İndirilen dosyanın tam yolu.
        :raises: requests/IO hataları, iptalde RuntimeError.
        """
        import requests

        # Yerelde her zaman düz (alt klasörsüz) saklarız; uzak yol alt klasör
        # içerebilir (split GGUF vb.), indirme URL'i tam yolu kullanır.
        dest = self.path_of(os.path.basename(filename))
        if os.path.isfile(dest):
            # Zaten indirilmiş; tam boyutu bildir ve çık.
            size = os.path.getsize(dest)
            if progress_cb:
                progress_cb(size, size)
            return dest

        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        tmp = dest + ".part"
        downloaded = 0
        try:
            with requests.get(url, stream=True, timeout=(10, 60),
                              allow_redirects=True) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                if progress_cb:
                    progress_cb(0, total)
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("İndirme iptal edildi.")
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb:
                            progress_cb(downloaded, total)
            os.replace(tmp, dest)
            return dest
        except BaseException:
            # İptal veya hata: yarım dosyayı temizle.
            try:
                if os.path.isfile(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise

    # -- Hugging Face arama (canlı katalog) ------------------------------- #
    _HF_API = "https://huggingface.co/api"

    def search_hf_models(self, query: str, limit: int = 40) -> List[Dict]:
        """
        Hugging Face'te GGUF içeren modelleri arar (herkese açık HTTP API).
        İndirme/aramada internet gerekir; sonuçlar indirilince çevrimdışı kalır.

        :return: [{repo_id, downloads, likes}] indirme sayısına göre azalan.
        :raises: requests/IO hataları (çağıran arka planda yakalar).
        """
        import requests

        params = {
            "search": query,
            "filter": "gguf",
            "sort": "downloads",
            "direction": "-1",
            "limit": str(limit),
        }
        resp = requests.get(f"{self._HF_API}/models", params=params,
                            timeout=(10, 30))
        resp.raise_for_status()
        out: List[Dict] = []
        for m in resp.json():
            repo_id = m.get("id") or m.get("modelId")
            if not repo_id:
                continue
            out.append({
                "repo_id": repo_id,
                "downloads": int(m.get("downloads") or 0),
                "likes": int(m.get("likes") or 0),
            })
        return out

    def list_repo_gguf_files(self, repo_id: str) -> List[Dict]:
        """
        Bir HF reposundaki .gguf dosyalarını listeler ve her birini sınıflandırır.

        :return: [{filename, size, kind, quant, quality}] — kind: "model" (tek
                 dosya, indirilebilir), "split" (çok parçalı, kullanılamaz),
                 "projector" (görsel eki). Model dosyaları boyuta göre artan,
                 diğerleri sonda. size bayt cinsinden (bilinmiyorsa None).
        :raises: requests/IO hataları (çağıran arka planda yakalar).
        """
        import requests

        url = f"{self._HF_API}/models/{repo_id}/tree/main"
        resp = requests.get(url, params={"recursive": "true"}, timeout=(10, 30))
        resp.raise_for_status()
        out: List[Dict] = []
        for item in resp.json():
            path = item.get("path", "")
            if item.get("type") == "file" and path.lower().endswith(".gguf"):
                entry = {"filename": path, "size": item.get("size")}
                entry.update(self.classify_gguf(path))
                out.append(entry)
        # Modeller önce (boyuta göre artan), sonra parçalı/görsel ekleri.
        out.sort(key=lambda f: (f["kind"] != "model", f.get("size") or 0))
        return out

    # gguf dosya adındaki "-00001-of-00002" gibi parçalı (split) işareti.
    _SPLIT_RE = re.compile(r"-\d{4,5}-of-\d{4,5}", re.IGNORECASE)

    # Sıkıştırma (quantization) -> (etiket, kalite açıklaması). Sıra önemli:
    # daha yüksek bit'ler önce denenir (q4, q8 içinde yanlış eşleşmesin diye).
    _QUANT_TABLE = (
        (r"f32", ("F32", "sıkıştırılmamış, en yüksek kalite, çok büyük")),
        (r"bf16|f16", ("16-bit", "sıkıştırılmamış, en yüksek kalite, çok büyük")),
        (r"q8", ("8-bit", "çok yüksek kalite")),
        (r"q6", ("6-bit", "yüksek kalite")),
        (r"q5", ("5-bit", "yüksek kalite")),
        (r"q4|iq4|mxfp4", ("4-bit", "dengeli — çoğu kullanım için önerilen")),
        (r"q3|iq3", ("3-bit", "küçük, kalite düşer")),
        (r"q2|iq2", ("2-bit", "çok küçük, kalite belirgin düşer")),
        (r"q1|iq1", ("1-bit", "en küçük, kalite çok düşük — önerilmez")),
    )

    @classmethod
    def classify_gguf(cls, filename: str) -> Dict[str, str]:
        """
        Bir .gguf dosya adını sınıflandırır: kullanılabilir model mi, çok parçalı
        mı, yoksa görsel eki (mmproj) mi; ve sıkıştırma seviyesi/kalitesi.
        """
        name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if "mmproj" in name:
            return {"kind": "projector", "quant": "-",
                    "quality": "görsel işleme eki (dil modeli değil)"}
        if cls._SPLIT_RE.search(name):
            return {"kind": "split", "quant": "-",
                    "quality": "çok parçalı dosya (tek başına kullanılamaz)"}
        for pattern, (label, desc) in cls._QUANT_TABLE:
            if re.search(pattern, name):
                return {"kind": "model", "quant": label, "quality": desc}
        return {"kind": "model", "quant": "?", "quality": "bilinmeyen sıkıştırma"}

    # -- Yardımcı ---------------------------------------------------------- #
    @staticmethod
    def parse_meta_from_filename(filename: str) -> Dict[str, str]:
        """
        Dosya adından parametre sayısı ve quantization bilgisini çıkarır
        (ör. 'Qwen2.5-7B-Instruct-Q4_K_M.gguf' -> {'parameters': '7B',
        'quantization': 'Q4_K_M'}). Bulunamazsa ilgili anahtar atlanır.
        """
        import re
        out: Dict[str, str] = {}
        params = re.search(r"(\d+(?:\.\d+)?[Bb])(?:[-_.]|$)", filename)
        if params:
            out["parameters"] = params.group(1).upper()
        quant = re.search(r"(Q\d[\w]*|F16|BF16|F32)", filename, re.IGNORECASE)
        if quant:
            out["quantization"] = quant.group(1).upper()
        return out
