# -*- coding: utf-8 -*-
"""
perf_monitor.py
===============
Performans/kaynak ölçüm yardımcıları.

Bir sorgu (retrieval + LLM cevap) çalışırken sistem kaynaklarını arka planda
örnekler ve sonucu okunaklı bir metin bloğu olarak biçimlendirir. Hem uygulama
(main.py) hem de benchmark betiği bu modülü kullanır.

İsteğe bağlı kütüphaneler (yoksa ilgili metrik 'yok' olarak geçilir):
    pip install psutil nvidia-ml-py
"""

from __future__ import annotations

import time
import threading
import statistics
from typing import List, Optional

# --- İsteğe bağlı ölçüm kütüphaneleri ---
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001
    pynvml = None
    _GPU_HANDLE = None


def gpu_available() -> bool:
    return _GPU_HANDLE is not None


class ResourceSampler:
    """
    Bir iş çalışırken CPU / RAM / GPU / VRAM kullanımını belirli aralıklarla
    örnekler. 'with' bloğu olarak kullanılır:

        with ResourceSampler() as s:
            ... ölçülecek iş ...
        print(s.summary())
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.cpu: List[float] = []
        self.ram_mb: List[float] = []
        self.gpu: List[float] = []
        self.vram_mb: List[float] = []

    def _run(self) -> None:
        if psutil is not None:
            psutil.cpu_percent(interval=None)  # İlk çağrı referans alır.
        while not self._stop.is_set():
            if psutil is not None:
                self.cpu.append(psutil.cpu_percent(interval=None))
                self.ram_mb.append(psutil.virtual_memory().used / 1e6)
            if _GPU_HANDLE is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                    self.gpu.append(float(util.gpu))
                    self.vram_mb.append(mem.used / 1e6)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(self.interval)

    def __enter__(self) -> "ResourceSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @staticmethod
    def _avg(values: List[float]) -> Optional[float]:
        return round(statistics.mean(values), 1) if values else None

    @staticmethod
    def _max(values: List[float]) -> Optional[float]:
        return round(max(values), 1) if values else None

    def summary(self) -> dict:
        return {
            "cpu_avg": self._avg(self.cpu), "cpu_max": self._max(self.cpu),
            "ram_avg_mb": self._avg(self.ram_mb), "ram_max_mb": self._max(self.ram_mb),
            "gpu_avg": self._avg(self.gpu), "gpu_max": self._max(self.gpu),
            "vram_avg_mb": self._avg(self.vram_mb), "vram_max_mb": self._max(self.vram_mb),
        }


def _v(value, suffix: str = "") -> str:
    """None ise '—', değilse değer + birim."""
    return f"{value}{suffix}" if value is not None else "—"


def format_perf_block(timings: dict, meta: dict, res: dict) -> str:
    """
    Ölçüm sonuçlarını cevabın sonuna eklenecek, hizalı ve okunaklı bir metin
    bloğu olarak biçimlendirir (ham tablo değil; etiketli özet).
    """
    meta = meta or {}
    res = res or {}
    it = meta.get("input_tokens")
    ot = meta.get("output_tokens")
    tt = meta.get("total_tokens")
    ret = timings.get("retrieval_s")
    gen = timings.get("generation_s")
    tot = timings.get("total_s")
    tps = round(ot / gen, 1) if (ot and gen) else meta.get("output_tps")

    lines = [
        "──────────── ⚙️  Performans Ölçümü ────────────",
        f"🔢 Token   : girdi {_v(it)} · çıktı {_v(ot)} · toplam {_v(tt)}",
        f"⏱️ Süre    : getirme {_v(ret, ' s')} · üretim {_v(gen, ' s')} · toplam {_v(tot, ' s')}",
        f"🚀 Hız     : {_v(tps, ' tok/sn')}",
        f"🖥️ CPU     : ort %{_v(res.get('cpu_avg'))} · max %{_v(res.get('cpu_max'))}",
        f"💾 RAM     : ort {_v(res.get('ram_avg_mb'), ' MB')} · max {_v(res.get('ram_max_mb'), ' MB')}",
    ]
    # GPU/VRAM yalnızca ölçülebildiyse göster.
    if res.get("gpu_max") is not None or res.get("vram_max_mb") is not None:
        lines.append(
            f"🎮 GPU     : ort %{_v(res.get('gpu_avg'))} · max %{_v(res.get('gpu_max'))}"
        )
        lines.append(
            f"🟩 VRAM    : ort {_v(res.get('vram_avg_mb'), ' MB')} · max {_v(res.get('vram_max_mb'), ' MB')}"
        )
    lines.append("──────────────────────────────────────────────")
    return "\n".join(lines)
