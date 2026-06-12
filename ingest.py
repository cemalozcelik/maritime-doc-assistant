# -*- coding: utf-8 -*-
"""
ingest.py
=========
Komut satırından toplu PDF/görsel içe aktarma (ingestion) aracı.

Dokümanları işler (metin katmanı + gerekirse OCR), embed eder ve kalıcı
ChromaDB'ye yazar. Sayfa-bazlı OCR önbelleği sayesinde ikinci çalıştırma çok
hızlıdır. Teknik çizim/şema dosyaları (balanced/fast modda) OCR'sız atlanır.

Kullanım örnekleri:
    # Bir klasörü balanced modda içe aktar (varsayılan):
    python ingest.py Instructions

    # Hızlı mod (DPI 150, çizimler atlanır):
    python ingest.py Instructions --ingest-mode fast

    # Tam mod (DPI 300, çizimler dahil OCR):
    python ingest.py Instructions --ingest-mode full

    # Çizimleri de OCR'la / OCR'ı zorla yenile:
    python ingest.py Instructions --ocr-drawings --force-reocr

    # Tek dosya, özel DPI:
    python ingest.py "Instructions/a.pdf" --ocr-dpi 250
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging

# Embedding modeli lokalse Hugging Face'i çevrimdışına al (uygulamayla aynı).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

from document_processor import (  # noqa: E402
    DocumentProcessor, format_ingestion_report, INGEST_MODES, DEFAULT_INGEST_MODE,
)
from embedding_manager import EmbeddingManager  # noqa: E402

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def _project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _new_agg() -> dict:
    return {
        "source": "TOPLAM", "total_files": 0, "total_pages": 0,
        "text_layer_pages": 0, "ocr_required_pages": 0, "ocr_used_pages": 0,
        "ocr_executed_pages": 0, "ocr_skipped_pages": 0, "drawing_skipped_pages": 0,
        "cache_hit_pages": 0, "fallback_pages": 0, "total_ocr_calls": 0,
        "total_chunks": 0, "total_chars": 0, "text_time": 0.0, "ocr_time": 0.0,
        "embedding_time": 0.0, "chroma_time": 0.0, "failed_pages": [],
        "dominant_rotation_files": 0,
    }


def _merge(agg: dict, st: dict, source: str) -> None:
    for key in ("total_pages", "text_layer_pages", "ocr_required_pages",
                "ocr_used_pages", "ocr_executed_pages", "ocr_skipped_pages",
                "drawing_skipped_pages", "cache_hit_pages", "fallback_pages",
                "total_ocr_calls", "total_chunks", "total_chars", "text_time", "ocr_time"):
        agg[key] += st.get(key, 0) or 0
    for fp in (st.get("failed_pages") or []):
        agg["failed_pages"].append(f"{source}:s{fp}")
    if st.get("dominant_rotation") is not None:
        agg["dominant_rotation_files"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Toplu doküman içe aktarma (ingestion)")
    parser.add_argument("paths", nargs="+", help="Dosya veya klasör yolları")
    parser.add_argument("--ingest-mode", choices=list(INGEST_MODES),
                        default=DEFAULT_INGEST_MODE,
                        help="fast / balanced / full (varsayılan balanced)")
    parser.add_argument("--ocr-dpi", type=int, default=None,
                        help="OCR render DPI (modu override eder; fast 150/balanced 200/full 300)")
    parser.add_argument("--force-reocr", action="store_true",
                        help="OCR önbelleğini yok say (yeniden OCR; önbellek yine güncellenir)")
    drg = parser.add_mutually_exclusive_group()
    drg.add_argument("--ocr-drawings", action="store_true",
                     help="Çizim/şema dosyalarında da OCR yap (atlama)")
    drg.add_argument("--skip-drawings", action="store_true",
                     help="Çizim/şema dosyalarında OCR'ı atla")
    parser.add_argument("--languages", default="tr,en", help="OCR dilleri (örn. tr,en)")
    parser.add_argument("--data-dir", default=os.path.join(_project_dir(), "data"),
                        help="Yazılabilir veri klasörü (vektör DB + OCR cache)")
    parser.add_argument("--no-cache", action="store_true", help="OCR önbelleğini kapat")
    parser.add_argument("--clear", action="store_true",
                        help="İçe aktarmadan önce vektör veritabanını temizle")
    args = parser.parse_args()

    skip_drawings = None
    if args.skip_drawings:
        skip_drawings = True
    elif args.ocr_drawings:
        skip_drawings = False

    data_dir = args.data_dir
    cache_dir = None if args.no_cache else os.path.join(data_dir, "ocr_cache")

    proc = DocumentProcessor(
        ocr_languages=[s.strip() for s in args.languages.split(",") if s.strip()],
        ingest_mode=args.ingest_mode,
        ocr_dpi=args.ocr_dpi,
        skip_drawings=skip_drawings,
        cache_dir=cache_dir,
        force_reocr=args.force_reocr,
        drawings_dir=os.path.join(data_dir, "drawings"),
    )
    embedder = EmbeddingManager(
        model_name_or_path=EMBEDDING_MODEL,
        persist_directory=os.path.join(data_dir, "vector_store"),
        collection_name="gemi_dokumanlari",
    )

    print(f"Mod: {args.ingest_mode} | DPI: {proc.ocr_dpi} | "
          f"cizim atla: {proc.skip_drawings} | cache: {bool(cache_dir)} | "
          f"force-reocr: {args.force_reocr}")

    if args.clear:
        print("Vektör veritabanı temizleniyor...")
        embedder.clear()

    print("Embedding modeli ısındırılıyor...")
    embedder.warm_up()

    # Dosyaları topla.
    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(proc.collect_supported_files(p))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[!] Bulunamadı, atlanıyor: {p}")
    if not files:
        parser.error("İşlenecek desteklenen dosya bulunamadı.")

    print(f"{len(files)} dosya işlenecek.\n")
    agg = _new_agg()
    existing = set(embedder.list_sources())
    t_start = time.perf_counter()

    for i, path in enumerate(files, start=1):
        name = os.path.basename(path)
        if name in existing:
            print(f"[{i}/{len(files)}] ATLANDI (zaten yüklü): {name}")
            continue
        print(f"[{i}/{len(files)}] İşleniyor: {name}")
        res = proc.process_file(path)
        st = res.stats or {}
        agg["total_files"] += 1
        _merge(agg, st, name)
        if res.success and res.chunks:
            embedder.add_chunks(res.chunks)
            agg["embedding_time"] += embedder.last_embedding_time
            agg["chroma_time"] += embedder.last_chroma_write_time
            existing.add(name)
        print("    -> " + (res.message or ""))

    for key in ("text_time", "ocr_time", "embedding_time", "chroma_time"):
        agg[key] = round(agg[key], 2)

    print("\n" + format_ingestion_report(agg))
    print(f"\nToplam süre: {round(time.perf_counter() - t_start, 1)} sn")
    print(f"Veritabanındaki toplam parça: {embedder.get_document_count()}")


if __name__ == "__main__":
    main()
