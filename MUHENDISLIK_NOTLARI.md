# Mühendislik Notları — Gemi Teknik Doküman Asistanı

Bu belge; sistemin mimarisini, alınan teknik kararları (ve nedenlerini), tradeoff'ları,
ölçülen performansı ve dikkat edilmesi gereken tuzakları senior bir mühendisin bakış
açısıyla kaydeder. Amaç: yeni bir geliştiricinin "neden böyle yapılmış?" sorusuna
koda bakmadan cevap bulabilmesi.

Son güncelleme: 2026-06-13

---

## 1. Sistem Özeti

Çevrimdışı (açık deniz) çalışabilen bir RAG (Retrieval-Augmented Generation) doküman
asistanı. Gemiye ait teknik PDF/görselleri (gerektiğinde OCR ile) işler, lokal bir
vektör veritabanına gömer ve bu dokümanlara dayalı soruları Türkçe yanıtlar.

Hat (pipeline):

```
PDF/Görsel ──► DocumentProcessor ──► chunk[]  ──► EmbeddingManager ──► ChromaDB
 (metin/OCR)     (metin + OCR)                      (e5-base vektör)    (kalıcı)
                                                                          │
Soru ──► EmbeddingManager.similarity_search (top-k, eşik) ──► bağlam ─────┘
                                                                  │
                                          LLMConnector (Gemini | yerel llama.cpp) ──► cevap
```

Tüm embedding/OCR/LLM lokalde çalışır; internet yalnızca (a) Gemini sağlayıcısı,
(b) ilk model indirmeleri için gerekir.

---

## 2. Modül Haritası (SRP)

| Modül | Sorumluluk |
|---|---|
| `main.py` | CustomTkinter GUI + koordinatör; ağır işler arka plan thread'inde, UI'a kuyrukla |
| `document_processor.py` | Dosya → metin parçaları; OCR hattı (EasyOCR), rotasyon, cache, modlar, çizim atlama |
| `embedding_manager.py` | sentence-transformers (e5-base) + ChromaDB; yazma ve benzerlik araması |
| `llm_connector.py` | Gemini + gömülü llama.cpp (GGUF); CUDA DLL yükleme; sistem yönergesi |
| `model_manager.py` | GGUF model kataloğu, HF canlı arama/indirme |
| `chat_store.py` | Kalıcı sohbet oturumları (JSON) |
| `ui_components.py` | UI bileşenleri (ChatHistoryRail, ChatArea, ModelsView, DownloadsView, SettingsView) |
| `perf_monitor.py` | CPU/RAM/GPU/VRAM örnekleme (metrik bloğu + benchmark) |
| `ingest.py` | Toplu içe aktarma CLI (modlar, cache, çizim bayrakları) |
| `benchmark.py` | Uçtan uca performans ölçüm CLI |
| `gemi_asistani.spec` / `installer.iss` | PyInstaller paketleme / Inno Setup kurulum |

Tasarım kuralı: `document_processor` ve `ui_components`, LLM/DB/embedding katmanlarını
import ETMEZ. Bağımlılık yönü tek taraflı; `main.py` her şeyi birleştirir (DIP).

---

## 3. Mimari Kararlar (ADR özetleri)

### ADR-1: Çevrimdışı LLM motoru — Ollama yerine gömülü llama.cpp
- **Karar:** Ollama bağımlılığı kaldırıldı; `llama-cpp-python` (GGUF) uygulamaya gömüldü.
- **Neden:** Kullanıcı harici program kurmak/`ollama serve` çalıştırmak istemiyordu.
  Gemideki bilgisayarda "çift tıkla çalışsın" gerekiyordu.
- **Tradeoff:** GPU kurulumu Ollama'dan daha kırılgan (CUDA wheel + DLL bağımlılığı).
  Modeller artık arayüzden indiriliyor (HF GGUF).

### ADR-2: GGUF modelleri arayüzden indir, pakete gömme
- **Karar:** Modeller `.exe`'ye gömülmez; çalışma anında `data/models_gguf/`'a iner.
- **Neden:** Q4_K_M modeller 1–45 GB; pakete gömmek absürt. Katalog HF API ile canlı
  aranabilir (model_manager). Doğrulanmış 24 "hızlı seçim" + arama + manuel giriş.

### ADR-3: Embedding'i biz hesaplayıp ChromaDB'ye vektör veriyoruz
- **Karar:** Koleksiyona "embedding function" verilmez; vektörler `LocalEmbedder` ile
  dışarıda hesaplanıp `collection.add(embeddings=...)` ile verilir.
- **Neden:** ChromaDB 1.5.x özel EF konfigürasyonunu kalıcı saklayıp sonraki açılışta
  tutarlılık doğrular → "embedding function conflict" hataları. Dış hesap = sürümden
  bağımsız, sağlam.
- **Detay:** e5 ailesi prefix'leri zorunlu: doküman `"passage: ..."`, sorgu
  `"query: ..."`; `normalize_embeddings=True` (kosinüs), `batch_size=64`, cihaz auto (cuda).

### ADR-4: torch CUDA sürümü (CPU değil) — GPU OCR için
- **Bağlam:** `.exe` boyutunu düşürmek için torch CPU'ya geçildi (5 GB → 2.4 GB).
  Ama torch SADECE embedding/OCR için kullanılıyor; LLM ayrı motorda (llama.cpp).
- **Tuzak:** torch CPU yapınca EasyOCR GPU'yu kaybetti → taranmış PDF OCR'ı ~5-10x
  yavaşladı (a.pdf 37s → 209s).
- **Karar:** torch CUDA'ya geri dönüldü. OCR ve embedding GPU'da; `.exe` ~5 GB.
- **Çıkarım:** Boyut optimizasyonu, dolaylı bir performans regresyonu yarattı.
  Embedding/OCR ağırlıklı bir üründe torch CUDA'nın bedeli kabul edilebilir.

### ADR-5: CUDA DLL'leri torch/lib'den; .lib dosyaları paketten elenir
- llama.cpp'nin `ggml-cuda.dll`'i `cudart64_12 / cublas64_12 / cublasLt64_12`'ye
  bağlı. Bunlar torch'un cu121 dağıtımıyla `torch/lib` içinde gelir.
- **Kritik:** Bu DLL'ler `import torch` ile önceden yüklenir; uygulama llama_cpp
  import ETMEDEN ÖNCE `torch/lib`'i DLL yoluna ekleyip DLL'leri AÇIKÇA önceden
  yükler (`_ensure_llama_loadable`). Aksi halde "Could not find module llama.dll".
- **Boyut:** spec, çalışma anında kullanılmayan `.lib`/`.pdb` (ör. `dnnl.lib` ~623 MB)
  dosyalarını eler.

### ADR-6: onedir paketleme + Inno Setup installer (onefile DEĞİL)
- **Karar:** PyInstaller `onedir`; tek-dosya dağıtım için Inno Setup `setup.exe`.
- **Neden:** `--onefile` her açılışta ~2.4 GB'ı %TEMP%'e açar → yavaş, kararsız,
  AV/SAC şüphesi artar. Inno Setup tek-dosya kolaylığını bu dezavantajlar olmadan verir.
- **Not:** İmzasız `.exe`/installer Windows SmartScreen / Akıllı Uygulama Denetimi
  (SAC) uyarısı verir. Kalıcı çözüm: kod imzalama sertifikası (EV ideal). Bizim
  değişiklik bu uyarının sebebi değil — her imzasız PyInstaller çıktısında olur.

---

## 4. OCR Hattı (projenin en kritik parçası)

Taranmış teknik dokümanlar (özellikle gemi manuelleri) çoğu zaman metin katmanı
içermez ve sayfalar farklı yönlerde taranmıştır. Naif OCR burada başarısız olur.

### 4.1 Problemin teşhisi (a.pdf vakası)
`a.pdf`: 37 sayfa, metin katmanı SIFIR, sayfalar **270° dönük**. Naif hat:
- Düşük DPI render (2x ≈ 144 DPI)
- Rotasyon denenmez (upright OCR → çöp)
- Agresif "gibberish" filtresi gerçek metni de eliyordu
→ **Sonuç: ChromaDB'ye 0 parça.**

### 4.2 Çözüm bileşenleri
1. **OCR fallback + yüksek DPI:** metinsiz sayfa → PyMuPDF ile render → EasyOCR.
   DPI moda göre (fast 150 / balanced 200 / full 300).
2. **Rotasyon tespiti:** sayfa 0/90/180/270 denenir; en çok ANLAMLI metin veren açı
   seçilir. Skor = `Σ(len(metin) × güven)`.
3. **Belge-geneli dominant rotasyon:** ilk birkaç GÜVENİLİR metinli sayfadan
   (`meaningful_score ≥ 60`) dominant açı belirlenir; kalan sayfalarda ÖNCE o denenir.
   Tüm sayfaları 4x taramaktan ~3x hızlı.
4. **Robust OCR parser** (`extract_text_from_ocr_result`): EasyOCR'ın 3 biçimini
   ((bbox,text,conf) / (bbox,text) / düz string) + PaddleOCR `[bbox,(text,conf)]`
   biçimini güvenle ayrıştırır. Tek indeks varsayımına (`line[1][0]`) güvenmez;
   exception'da çökmez, ham çıktıyı debug'a yazar.
5. **Nazik temizlik** (`clean_ocr_text`): boşluk normalize; Türkçe karakter, teknik
   sembol ve formüller (μ, H₂O) KORUNUR.
6. **Gevşetilmiş chunk filtresi:** eski agresif filtre kaldırıldı; yalnızca açıkça
   çöp (neredeyse harfsiz) parça elenir. `min_chunk_chars=60`. OCR metin üretip 0
   chunk çıkarsa salvage + WARNING.

### 4.3 `meaningful_score` — anahtar sezgi
```python
_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}")
meaningful_score(text) = Σ len(kelime-benzeri token)
```
Gerçek metni OCR gürültüsünden ayırır: gerçek metin yüksek, yanlış rotasyon/saf
gürültü düşük (gerçek ~37 vs gürültü ~4). **Yalnızca rotasyon ve çizim kararında**
kullanılır; chunk içeriğini etkilemez.

### 4.4 Hız optimizasyonları (eşikler)
- Dominant tespiti yalnızca OCR gereken sayfa ≥ 8 ise (`DOMINANT_MIN_PAGES`).
- Dominant açı `meaningful_score ≥ 20` (`ROT_ACCEPT_MIN`) verirse kabul, fallback yok.
- Probe eşiği yüksek (`ROT_PREFERRED_MIN=60`) — güvenilir sinyal.
- **Dominant kurulduğunda fallback kapalı:** çizim sayfaları boşuna 4x taranmaz (tek çağrı).
- **Give-up:** dominant kurulamayan büyük dosyada (≥8 OCR sayfası ama okunur metin yok)
  5 denemeden sonra rotasyon araması bırakılır; kalan sayfalar 0°'de tek çağrı.
- **Gömülü çizim:** OCR sonrası `meaningful_score < 12` ise sayfa "manuel içi çizim"
  sayılır → Chroma'ya EKLENMEZ, `drawings_index.jsonl`'a yazılır.

### 4.5 OCR önbelleği (cache)
Sayfa-bazlı. Anahtar: `file_hash + page + dpi + engine + languages + rotation_policy`.
`file_hash` = boyut + baş/orta/son 64 KB'ın SHA-1'i (büyük PDF'lerde hızlı, pratikte
çakışmaz). Her dosya için tek JSON. Sonuç: 2. çalıştırma anlık (a.pdf 72 sn → 0.2 sn).
Cache-hit sayfalar dominant rotasyon kararına da katkı verir.

### 4.6 İçe aktarım modları
| Mod | DPI | Çizim |
|---|---|---|
| fast | 150 | atla |
| **balanced** (varsayılan) | 200 | atla |
| full | 300 | OCR dahil |

Çizim/şema dosyaları **ad ve klasör** üzerinden tespit (drawing, diagram, piping,
schematic, plan, dwg, wiring...). Balanced/fast'ta OCR atlanır; sayfa görüntüsü
`drawings_dir/<file_hash>/page_NNN.png`, metadata `drawings_index.jsonl`. Aynı
dosyadaki metin katmanlı sayfalar yine eklenir. **Çizim OCR gürültüsü ana retrieval'ı
bozmaz.**

---

## 5. Retrieval / RAG

- Model: `intfloat/multilingual-e5-base` (Türkçe + İngilizce iyi, küçük, hızlı).
- ChromaDB kalıcı, kosinüs (`hnsw:space=cosine`).
- `similarity_search`: `k`dan fazla aday çek → `max_distance` (varsayılan 0.55)
  eşiğinden uzak (alakasız) parçaları ele → en iyi `k`. Eşiği geçen yoksa boş liste
  döner (LLM "dokümanda bulunmuyor" demeli).
- Metadata: `source_file, page, chunk_index, ocr_used, rotation, char_count`.
- **Etiketli hibrit yanıt:** cevap önce dokümana dayanır ("DOKÜMANDAN"); genel
  mühendislik bilgisi ayrı ve açıkça etiketlenir ("GENEL MÜHENDİSLİK BİLGİSİ").
  System prompt düz metin / markdown'sız (kullanıcı tercihi).

---

## 6. Eşzamanlılık / UI

Tkinter thread-safe değildir. Kural: arka plan thread'leri UI'ı DOĞRUDAN güncellemez;
`queue`'ya iş bırakır, ana thread `after` döngüsüyle işler (`_run_in_background` /
`_post_ui` / `_process_ui_queue`). Tek anda tek ağır iş (`_busy`).

---

## 7. Ölçülen Performans (RTX 4060 Laptop, 8 GB VRAM)

### a.pdf (37 sayfa, taranmış, 270° dönük) — iyileştirme yolculuğu
| Aşama | OCR süresi | Chunk |
|---|---|---|
| Rotasyonsuz (naif) | 37 sn | **0** (işe yaramaz) |
| Her sayfa 4x rotasyon (CPU torch) | 209 sn | 36 |
| Her sayfa 4x (GPU torch, DPI 300) | 86 sn | 38 |
| Dominant + anlamlı kabul (DPI 300) | 86 sn | 38 |
| **Balanced (DPI 200) + gömülü-çizim opt.** | **53.5 sn** | **38** |
| 2. çalıştırma (cache) | ~0.2 sn | 38 |

### Tüm "Instructions" klasörü (balanced)
- 44 dosya, 3203 sayfa: 2749 metin katmanlı, 454 OCR gereken.
- 154 sayfa çizim olarak atlandı; ~51 sayfa manuel-içi gömülü çizim.
- Toplam: ~15 dk (OCR ~12 dk baskın), 5647 chunk. Metin çıkarma 4.5 sn (bedava sayılır).
- İkinci çalıştırma cache ile çok daha hızlı.

### LLM (yerel)
- Qwen2.5/Qwen3 7-8B Q4_K_M, RTX 4060'a sığar; ~87 tok/sn.
- 1.5B modeller Türkçe teknik akıl yürütmede yetersiz (döngüye girer) — 7B+ önerilir.

### Paket boyutu
- torch CUDA + llama CUDA ile `.exe` klasörü ~5 GB (ggml-cuda.dll ~700 MB, torch ~4.4 GB).
- `.lib`/`.pdb` eleme ~0.7 GB kazandırır. Inno installer (lzma2/max) ~580 MB tek dosya.

---

## 8. Tuzaklar / Öğrenilenler (gotchas)

- **HF çevrimdışı sırası:** `HF_HUB_OFFLINE`, `huggingface_hub` import EDİLMEDEN ÖNCE
  ayarlanmalı (import anında okunup sabitlenir). `main.py` ağır importlardan önce
  `enable_hf_offline_if_available` çağırır.
- **EasyOCR GPU = torch.cuda.is_available():** torch CPU yapılırsa OCR sessizce CPU'ya
  düşer ve yavaşlar. Boyut/performans bağı buradan geçer (ADR-4).
- **ChromaDB collection adı:** 3–512 karakter, `[a-zA-Z0-9._-]`. "t" gibi kısa adlar hata verir.
- **numpy + EasyOCR:** döndürülen diziyi `np.ascontiguousarray` ile bitişik yapmak gerekir.
- **np.rot90(arr, k):** 0/90/180/270 için kayıpsız döndürme (k=0..3), interpolasyon yok.
- **Windows torch CPU bile ~1.1 GB** (Intel MKL); Linux'taki ~200 MB beklentisi yanıltıcı.
- **`paragraph=True` EasyOCR'da güven skorunu DÜŞÜRÜR** (tuple (bbox,text) olur, conf yok);
  rotasyon skoru için `detail=1, paragraph=False` kullanılır.
- **Çizim tespiti basename değil tam yol:** `Instructions/Diagrams/x.pdf` klasör adından
  yakalanır (`normalized_path`).
- **Smart App Control tek yönlü:** kapatınca Windows sıfırlamadan geri açılmaz.
- **`~orch` gibi `~` önekli klasörler:** pip yarım kalan kaldırma artığı; venv'de çöp,
  silinebilir.

---

## 9. Gelecek İş / Açık Konular

- **Kod imzalama** (EV sertifikası) → SmartScreen/SAC uyarısını kalıcı çözer. En öncelikli
  dağıtım kalemi.
- **Çok-turlu hafıza:** şu an her soru bağımsız RAG; sohbet geçmişini modele geri besleme
  (Gemini/yerel) eklenebilir.
- **Tam-çevrimdışı paket:** embedding modelini `.exe`'ye gömmek (spec'te hazır, kapalı) +
  EasyOCR cache'ini taşımak gerekir (internetsiz hedef için).
- **Çizim sayfaları için ayrı multimodal arama** (görüntü embedding) — şu an sadece
  atlanıp indeksleniyor.
- **OCR motoru soyutlaması:** parser PaddleOCR'ı da destekliyor; istenirse engine seçimi
  konfigüre edilebilir.
- **Daha akıllı çizim tespiti:** sayfa içeriğinden (vektör çizim oranı, görüntü kaplama)
  ada bakmadan sınıflandırma.

---

## 10. CLI Hızlı Referans

```bash
# Toplu içe aktarma (balanced, cache açık)
python ingest.py Instructions

# Hızlı / tam mod
python ingest.py Instructions --ingest-mode fast
python ingest.py Instructions --ingest-mode full

# Çizimleri de OCR'la / OCR'ı zorla yenile / cache kapat
python ingest.py Instructions --ocr-drawings --force-reocr --no-cache

# Performans ölçümü (yerel GGUF / Gemini)
python benchmark.py --provider local --model data/models_gguf/<model>.gguf
python benchmark.py --provider gemini --model gemini-2.5-pro --api-key XYZ

# Paketleme
pyinstaller gemi_asistani.spec --clean --noconfirm
ISCC.exe installer.iss   # -> installer_output/GemiAsistani-Kurulum.exe
```
