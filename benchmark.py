# -*- coding: utf-8 -*-
"""
benchmark.py
============
Gemi Teknik Doküman Asistanı - Performans ölçüm (benchmark) aracı.

Tez/rapor için RAG hattının uçtan uca performansını ölçer:
    * Token sayıları (girdi/çıktı/toplam) - modelden gerçek değerler
    * Süreler: retrieval (bağlam getirme), generation (LLM cevap), toplam
    * Üretim hızı (token/saniye)
    * Sistem kaynakları: CPU (ort/max), RAM (ort/max), GPU (ort/max),
      VRAM (ort/max) - sorgu boyunca örneklenir

Sonuçlar hem ekrana (tablo) hem de CSV dosyasına yazılır.

Gereksinim (yalnızca benchmark için):
    pip install psutil nvidia-ml-py

Kullanım örnekleri:
    # Yerel (gömülü llama.cpp) ile, GGUF dosya yolu vererek:
    python benchmark.py --provider local --model data/models_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf

    # Gemini ile:
    python benchmark.py --provider gemini --model gemini-2.5-pro --api-key XYZ

    # Her soruyu 3 kez tekrarla (ortalama için), kendi sorularınla:
    python benchmark.py --provider local --model <gguf-yolu> --repeat 3 --questions sorular.txt
"""

from __future__ import annotations

import os
import csv
import time
import argparse
import threading
import statistics
from datetime import datetime
from typing import List, Optional

# Model lokaldeyse Hugging Face'i çevrimdışına al (uygulamayla aynı davranış).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from embedding_manager import EmbeddingManager
from llm_connector import LLMConnector
# Ölçüm altyapısı tek kaynaktan (uygulama da aynısını kullanır).
import perf_monitor as pm
from perf_monitor import ResourceSampler


EMBEDDING_MODEL = "BAAI/bge-m3"
# Önce projedeki yerel 'models/' klasörü (safetensors, çevrimdışı), yoksa HF adı.
_local_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", EMBEDDING_MODEL.replace("/", "_"))
if os.path.isdir(_local_model):
    EMBEDDING_MODEL = _local_model
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vector_store")
TOP_K = 8

DEFAULT_QUESTIONS = [
    "2 zamanlı gemi ana makinesinin egzoz sıcaklıkları arasında farklılık oluşuyor. "
    "Neden olabilir ve nasıl çözülebilir? Maddeler halinde açıkla.",
    "Yakıt seperatöründe (purifier) verim düşüklüğünün olası nedenleri nelerdir?",
    "Yardımcı kazan (auxiliary boiler) bakımında dikkat edilmesi gereken noktalar nelerdir?",
    "Hava kompresörünün basıncı düşükse hangi kontroller yapılmalıdır?",
]


# ---------------------------------------------------------------------------
#  Benchmark Çekirdeği
# ---------------------------------------------------------------------------
def run_query(embedder: EmbeddingManager, llm: LLMConnector, question: str) -> dict:
    """Tek bir soruyu (retrieval + generation) ölçerek çalıştırır."""
    row: dict = {"question": question[:60] + ("..." if len(question) > 60 else "")}

    with ResourceSampler() as sampler:
        t0 = time.perf_counter()
        contexts = embedder.similarity_search(question, k=TOP_K)
        t1 = time.perf_counter()
        response = llm.ask(question, contexts)
        t2 = time.perf_counter()

    meta = response.meta or {}
    row.update({
        "ok": response.success,
        "n_contexts": len(contexts),
        "parameters": meta.get("parameters"),       # ör. 31.3B
        "quantization": meta.get("quantization"),   # ör. Q4_K_M
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "total_tokens": meta.get("total_tokens"),
        "model_output_tps": meta.get("output_tps"),  # modelin saf üretim hızı (tok/sn)
        "retrieval_s": round(t1 - t0, 3),
        "generation_s": round(t2 - t1, 3),
        "total_s": round(t2 - t0, 3),
    })
    # Uçtan uca üretim hızı (token/sn).
    if row["output_tokens"] and row["generation_s"]:
        row["e2e_tps"] = round(row["output_tokens"] / row["generation_s"], 2)
    else:
        row["e2e_tps"] = None
    row.update(sampler.summary())
    if not response.success:
        row["error"] = response.error
    return row


def print_table(rows: List[dict]) -> None:
    """Sonuçları okunaklı bir tablo olarak yazdırır."""
    cols = [
        ("question", "Soru", 35),
        ("input_tokens", "Girdi tok", 10),
        ("output_tokens", "Çıktı tok", 10),
        ("generation_s", "Üretim s", 9),
        ("e2e_tps", "tok/sn", 7),
        ("cpu_avg", "CPU~", 6), ("cpu_max", "CPUmax", 7),
        ("ram_max_mb", "RAMmax MB", 10),
        ("gpu_avg", "GPU~", 6), ("gpu_max", "GPUmax", 7),
        ("vram_max_mb", "VRAMmax MB", 11),
    ]
    header = " | ".join(name.ljust(w) for _k, name, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        line = " | ".join(str(r.get(k, "") if r.get(k) is not None else "-").ljust(w)
                           for k, _n, w in cols)
        print(line)


def print_aggregate(rows: List[dict]) -> None:
    """Tüm sorulara dair ortalama/zirve özetini yazdırır."""
    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        print("\n[!] Başarılı sorgu yok; özet üretilemedi.")
        return

    def avg(key):
        vals = [r[key] for r in ok_rows if r.get(key) is not None]
        return round(statistics.mean(vals), 2) if vals else None

    def mx(key):
        vals = [r[key] for r in ok_rows if r.get(key) is not None]
        return round(max(vals), 1) if vals else None

    print("\n=== GENEL ÖZET (başarılı %d sorgu) ===" % len(ok_rows))
    print(f"  Ortalama girdi token   : {avg('input_tokens')}")
    print(f"  Ortalama çıktı token   : {avg('output_tokens')}")
    print(f"  Ortalama toplam token  : {avg('total_tokens')}")
    print(f"  Ortalama üretim süresi : {avg('generation_s')} s")
    print(f"  Ortalama toplam süre   : {avg('total_s')} s")
    print(f"  Ortalama hız (tok/sn)  : {avg('e2e_tps')}")
    print(f"  CPU  ortalama / zirve  : {avg('cpu_avg')}% / {mx('cpu_max')}%")
    print(f"  RAM  ortalama / zirve  : {avg('ram_avg_mb')} MB / {mx('ram_max_mb')} MB")
    print(f"  GPU  ortalama / zirve  : {avg('gpu_avg')}% / {mx('gpu_max')}%")
    print(f"  VRAM ortalama / zirve  : {avg('vram_avg_mb')} MB / {mx('vram_max_mb')} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemi Asistanı performans benchmark aracı")
    parser.add_argument("--provider", choices=["local", "gemini"], default="local")
    parser.add_argument("--model", default="",
                        help="Yerel için GGUF dosya yolu; Gemini için model adı")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY", ""),
                        help="Gemini API anahtarı (gemini için)")
    parser.add_argument("--repeat", type=int, default=1, help="Her soruyu kaç kez çalıştır")
    parser.add_argument("--questions", default="", help="Soru dosyası (her satır bir soru)")
    parser.add_argument("--out", default="benchmark_results.csv", help="CSV çıktı dosyası")
    args = parser.parse_args()

    # Soruları hazırla.
    if args.questions and os.path.isfile(args.questions):
        with open(args.questions, encoding="utf-8") as fh:
            questions = [ln.strip() for ln in fh if ln.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    # Ortam bilgisi.
    print("=" * 70)
    print("ORTAM")
    if pm.psutil is not None:
        print(f"  CPU çekirdek (mantıksal): {pm.psutil.cpu_count(logical=True)}")
        print(f"  Toplam RAM             : {round(pm.psutil.virtual_memory().total/1e9,1)} GB")
    if pm.gpu_available():
        name = pm.pynvml.nvmlDeviceGetName(pm._GPU_HANDLE)
        total = pm.pynvml.nvmlDeviceGetMemoryInfo(pm._GPU_HANDLE).total / 1e9
        print(f"  GPU                    : {name} ({round(total,1)} GB VRAM)")
    else:
        print("  GPU                    : tespit edilemedi (pynvml yok / GPU yok)")
    print(f"  Sağlayıcı / model      : {args.provider} / {args.model}")
    print("=" * 70)

    # Modülleri hazırla. Embedding CPU'da: LLM ile aynı GPU'da VRAM çakışmasını
    # önler (uygulamadaki davranışla tutarlı; bkz. main.py).
    embedder = EmbeddingManager(
        model_name_or_path=EMBEDDING_MODEL, persist_directory=DATA_DIR, device="cpu"
    )
    print("Embedding modeli ısındırılıyor...")
    t = time.perf_counter()
    embedder.warm_up()
    print(f"  Embedding hazır ({round(time.perf_counter()-t,1)} s), "
          f"veritabanı parça sayısı: {embedder.get_document_count()}")

    llm = LLMConnector()
    if args.provider == "gemini":
        if not args.api_key:
            parser.error("Gemini için --api-key (veya GEMINI_API_KEY) gerekli.")
        llm.use_gemini(api_key=args.api_key, model_name=args.model or "gemini-2.5-pro")
    else:
        if not args.model or not os.path.isfile(args.model):
            parser.error("Yerel için --model ile geçerli bir GGUF dosya yolu verin.")
        llm.use_local(model_path=args.model)

    # Çalıştır.
    rows: List[dict] = []
    seq = [q for q in questions for _ in range(args.repeat)]
    for i, q in enumerate(seq, start=1):
        print(f"\n[{i}/{len(seq)}] çalışıyor: {q[:70]}...")
        row = run_query(embedder, llm, q)
        rows.append(row)
        if row.get("ok"):
            print(f"    -> {row.get('output_tokens')} çıktı token, "
                  f"{row.get('generation_s')} s, {row.get('e2e_tps')} tok/sn, "
                  f"VRAMmax {row.get('vram_max_mb')} MB")
        else:
            print(f"    -> HATA: {row.get('error')}")

    # Raporla.
    print_table(rows)
    print_aggregate(rows)

    # CSV yaz.
    fieldnames = [
        "question", "ok", "n_contexts", "parameters", "quantization",
        "input_tokens", "output_tokens",
        "total_tokens", "model_output_tps", "e2e_tps", "retrieval_s",
        "generation_s", "total_s", "cpu_avg", "cpu_max", "ram_avg_mb",
        "ram_max_mb", "gpu_avg", "gpu_max", "vram_avg_mb", "vram_max_mb", "error",
    ]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n[+] Sonuçlar CSV'ye yazıldı: {args.out}  ({datetime.now():%Y-%m-%d %H:%M})")

    if pm.pynvml is not None:
        try:
            pm.pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
