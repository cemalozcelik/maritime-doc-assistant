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

# Not: LLM GPU hızlandırması için gerekli CUDA runtime DLL'leri (cudart, cublas,
# cublasLt) torch'un CUDA sürümüyle birlikte 'torch/lib' içinde gelir ve
# collect_all("torch") ile pakete eklenir. Uygulama, llama_cpp import edilmeden
# önce bu klasörü DLL yoluna ekleyip DLL'leri önceden yükler (llm_connector).

# ----------------------------------------------------------------------------
#  Boyut optimizasyonu: çalışma zamanında GEREKSİZ dosyaları paketten çıkar.
#  '.lib' (link-time kütüphaneleri, ör. torch/lib/dnnl.lib ~623 MB) ve '.pdb'
#  (hata ayıklama sembolleri) çalışma anında hiç kullanılmaz. Bunları elemek
#  .exe boyutunu ~0.7 GB azaltır. Yalnızca kaynak dosya uzantısına bakılır.
# ----------------------------------------------------------------------------
def _strip_useless(entries):
    kept = []
    dropped = 0
    for item in entries:
        src = item[0]
        if isinstance(src, str) and src.lower().endswith((".lib", ".pdb")):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped

binaries, _d1 = _strip_useless(binaries)
datas, _d2 = _strip_useless(datas)
print(f"[spec] Gereksiz .lib/.pdb dosyalari cikarildi: {_d1 + _d2} adet")

# ----------------------------------------------------------------------------
#  DAĞITIM: yalnızca embedding modeli (bge-m3) pakete gömülür. Bu model torch<2.6
#  ile .bin yükleyemediği (CVE-2025-32434) için safetensors olarak gömülmesi en
#  güvenli yoldur; ayrıca ~2.3 GB'lık indirmeyi de baştan halleder.
#  GGUF (LLM) ve EasyOCR modelleri GÖMÜLMEZ; ilk kullanımda internetten iner
#  (GGUF: İndirilenler sekmesi; EasyOCR: ilk OCR'da otomatik). bkz.
#  local_embedding_model_path (main.py) ve _bundled_easyocr_dir (document_processor.py
#  -> gömülü model yoksa EasyOCR varsayılan indirme davranışına düşer).
# ----------------------------------------------------------------------------
if _os.path.isdir("models"):
    datas += [("models", "models")]          # bge-m3 (safetensors, ~2.3 GB)
else:
    print("[spec] UYARI: 'models/' yok -> embedding modeli pakete gomulmeyecek!")


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
