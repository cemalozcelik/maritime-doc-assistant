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
import logging
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# İndirilebilir önerilen modeller (Türkçe-yetkin instruct GGUF'lar, Q4_K_M).
# Küçükten büyüğe sıralı. CPU'da 1.5B / 3B önerilir; GPU varsa 7B-9B akıcıdır.
# repo_id + filename, Hugging Face'teki gerçek dosya yollarıdır.
CURATED_MODELS: List[Dict] = [
    {
        "key": "qwen2.5-1.5b",
        "label": "Qwen2.5 1.5B Instruct (en küçük, hızlı)",
        "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "approx_mb": 1120,
        "params": "1.5B",
        "quant": "Q4_K_M",
        "note": "Düşük RAM/CPU için. En hızlı ama en az yetenekli.",
    },
    {
        "key": "qwen2.5-3b",
        "label": "Qwen2.5 3B Instruct (dengeli)",
        "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "approx_mb": 2020,
        "params": "3B",
        "quant": "Q4_K_M",
        "note": "CPU için iyi denge. Çoğu teknik soru için yeterli.",
    },
    {
        "key": "qwen2.5-7b",
        "label": "Qwen2.5 7B Instruct (kaliteli)",
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "approx_mb": 4680,
        "params": "7B",
        "quant": "Q4_K_M",
        "note": "GPU önerilir. Türkçe ve teknik akıl yürütmede güçlü.",
    },
    {
        "key": "llama3.1-8b",
        "label": "Llama 3.1 8B Instruct",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "approx_mb": 4920,
        "params": "8B",
        "quant": "Q4_K_M",
        "note": "GPU önerilir. Genel amaçlı güçlü model.",
    },
    {
        "key": "gemma2-9b",
        "label": "Gemma 2 9B Instruct",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "approx_mb": 5760,
        "params": "9B",
        "quant": "Q4_K_M",
        "note": "GPU gerekir. En kaliteli ama en ağır seçenek.",
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

        dest = self.path_of(filename)
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
