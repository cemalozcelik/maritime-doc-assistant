# -*- mode: python ; coding: utf-8 -*-
# ============================================================================
#  PyInstaller spec dosyası - Gemi Teknik Doküman Asistanı
# ----------------------------------------------------------------------------
#  Kullanım:
#       pyinstaller gemi_asistani.spec --clean
#
#  Bu spec; CustomTkinter, ChromaDB, sentence-transformers/torch/transformers
#  ve EasyOCR gibi "gizli import" ve "veri dosyası" gerektiren ağır kütüphaneleri
#  otomatik toplar (collect_all). Tek dosya (--onefile) yerine 'onedir' modu
#  kullanılır; çünkü bu büyük modeller --onefile ile çok yavaş açılır ve
#  bazen bozulur. Dağıtım için 'dist/GemiAsistani' klasörünün tamamı kopyalanır.
# ============================================================================

from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = []
binaries = []
hiddenimports = []

# Ağır kütüphaneleri eksiksiz topla (modül + veri + ikili dosyalar).
for pkg in [
    "customtkinter",
    "tkinterdnd2",
    "chromadb",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "torch",
    "easyocr",
    "onnxruntime",
    "google.genai",
    "llama_cpp",        # Gömülü çevrimdışı motor (büyük: ggml-cuda.dll dahil).
    "huggingface_hub",  # GGUF model indirme yardımcıları.
    "requests",         # HF arama + GGUF indirme (HTTPS).
    "certifi",          # HTTPS için CA sertifika paketi (cacert.pem).
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:
        print(f"[spec] '{pkg}' toplanamadı (kurulu olmayabilir): {exc}")

# Ek gizli importlar (dinamik yüklenenler).
hiddenimports += [
    "PIL._tkinter_finder",
    "google.genai",
    "langchain_text_splitters",
    "chromadb.telemetry.product.posthog",
    "chromadb.api.segment",
    "chromadb.segment.impl.vector.local_hnsw",
]

# Uygulama ikonu (çalışan pencerede de kullanılmak üzere pakete eklenir).
import os as _os
if _os.path.isfile("icon.png"):
    datas += [("icon.png", ".")]
if _os.path.isfile("icon.ico"):
    datas += [("icon.ico", ".")]

# ----------------------------------------------------------------------------
#  CUDA runtime DLL'leri (LLM GPU hızlandırması için).
#  torch'un CPU sürümü kullanıldığından bu DLL'ler artık 'nvidia-*-cu12' pip
#  paketlerinden gelir. ggml-cuda.dll bunlara muhtaçtır; onedir modunda aynı
#  klasöre konduklarında Windows otomatik bulur. Paket boyutunu küçük tutmak
#  için (torch CUDA ~4.4 GB yerine ~0.6 GB) bu yöntem tercih edilir.
# ----------------------------------------------------------------------------
import sys as _sys
import glob as _glob
_cuda_dlls = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
_seen = set()
for _p in _sys.path:
    if not _p or not _os.path.isdir(_p):
        continue
    for _name in _cuda_dlls:
        for _hit in _glob.glob(_os.path.join(_p, "nvidia", "*", "bin", _name)):
            if _name not in _seen:
                binaries.append((_hit, "."))
                _seen.add(_name)
                print(f"[spec] CUDA DLL eklendi: {_hit}")
_missing = [d for d in _cuda_dlls if d not in _seen]
if _missing:
    print(f"[spec] UYARI: CUDA DLL bulunamadi: {_missing} -> LLM GPU calismayabilir. "
          f"Kurulum: pip install nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-runtime-cu12==12.1.105")

# ----------------------------------------------------------------------------
#  İSTEĞE BAĞLI: Embedding modelini .exe içine gömmek isterseniz, modeli
#  'models/intfloat_multilingual-e5-base' klasörüne indirip aşağıyı açın.
#  Böylece uygulama internetsiz ilk açılışta bile modeli bulur.
# ----------------------------------------------------------------------------
# datas += [("models", "models")]


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,        # onedir modu
    name="GemiAsistani",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                    # UPX, torch DLL'lerini bozabilir -> kapalı.
    console=False,                # GUI uygulaması -> konsol penceresi yok.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="icon.ico" if _os.path.isfile("icon.ico") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="GemiAsistani",
)
