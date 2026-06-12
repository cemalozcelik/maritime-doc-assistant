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
    # icon="assets/app.ico",      # Kendi ikonunuzu eklemek isterseniz açın.
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
