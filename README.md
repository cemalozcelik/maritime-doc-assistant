# Maritime Doc Assistant — Gemi Teknik Doküman Asistanı

Açık denizde, internetin olmadığı veya kısıtlı olduğu koşullarda çalışmak üzere
tasarlanmış, tamamen lokal (çevrimdışı) çalışabilen bir yapay zeka doküman asistanı.
Gemiye ait teknik PDF'leri ve görselleri (OCR ile) işler, içeriklerini lokal bir
vektör veritabanına gömer ve bu dokümanlara dayalı soruları Türkçe yanıtlar.

---

## Genel Bakış

Uygulama, Retrieval-Augmented Generation (RAG) mimarisi üzerine kuruludur:

1. Yüklenen dokümanlar metne dönüştürülür (PDF okuma + gerektiğinde OCR).
2. Metin parçaları lokal bir embedding modeli ile vektörleştirilip kalıcı bir
   vektör veritabanına yazılır.
3. Kullanıcının sorusu için en ilgili bağlam parçaları getirilir ve seçilen dil
   modeline (çevrimiçi Gemini veya çevrimdışı gömülü yerel model) gönderilerek
   yanıt üretilir.

Tüm embedding ve OCR işlemleri lokalde çalışır; internet yalnızca Gemini sağlayıcısı
seçildiğinde gereklidir.

---

## Temel Özellikler

- Çift dil modeli sağlayıcısı: internet varken Gemini API (varsayılan
  gemini-2.5-pro), yokken gömülü yerel motor (llama.cpp / GGUF). Arayüzden tek tıkla
  geçiş. Gemini bağlantısı güncel google-genai SDK'sı ile kurulur. Çevrimdışı motor
  için harici bir program (Ollama vb.) GEREKMEZ; tamamen uygulamaya gömülüdür.
- Arayüzden model indirme: "İndirilenler" sekmesinden önerilen GGUF modelleri (Qwen2.5
  1.5B/3B/7B, Llama 3.1 8B, Gemma 2 9B) tek tıkla, ilerleme çubuğuyla indirilir;
  terminal veya harici komut gerekmez. Tüm model kataloğu burada listelenir (başka
  bilgisayarlara kurulumda kolaylık için). İndirilen model "Modeller" sekmesinden
  aktif seçilir; silme de "İndirilenler" sekmesinden yapılır.
- Geçmiş sohbetler (Gemini benzeri): her sohbet diske kaydedilir; uygulama kapanıp
  açıldığında sol panelde listede kalır. Yeni sohbet açma, geçmişe tıklayıp devam
  etme ve silme desteklenir.
- Gemini benzeri arayüz: solda geçmiş sohbet rayı ve Sohbet / Modeller / İndirilenler /
  Ayarlar gezinmesi; ortada sohbet alanı.
- Etiketli hibrit yanıt: cevap önce yüklenen dokümana dayanır ("DOKÜMANDAN"),
  gerektiğinde genel mühendislik bilgisi ayrı ve açıkça etiketlenerek ("GENEL
  MÜHENDİSLİK BİLGİSİ — dokümanda doğrulanmadı") eklenir. Sayı ve parça numaraları
  asla uydurulmaz; her cevapta kaynak ve sayfa bilgisi verilir.
- Uygulama içi performans ölçümü: her cevabın sonuna token sayısı (girdi/çıktı),
  süre (getirme/üretim), hız (tok/sn) ve sorgu boyunca ölçülen CPU/RAM/GPU/VRAM
  (ortalama ve zirve) ile kullanılan model, parametre sayısı ve quantization
  bilgisi düz metin bir blok olarak eklenir.
- Lokal RAG: yanıtlar yalnızca yüklenen dokümanlardaki bağlama dayanır; cevaplarda
  kaynak ve sayfa bilgisi belirtilir.
- Çoklu yükleme yöntemi: tek tek dosya, komple klasör (alt klasörler dahil) veya
  pencereye sürükle-bırak. Aynı dosya ikinci kez yüklenmek istendiğinde tekrar
  işlenmez, atlanır.
- Lokal OCR: görsellerdeki ve taranmış PDF sayfalarındaki metinleri internetsiz okur
  (EasyOCR, Türkçe + İngilizce); GPU varsa otomatik kullanılır.
- Kalıcı vektör veritabanı: ChromaDB ile veriler diske yazılır; uygulama kapatılsa
  bile dokümanlar korunur.
- Türkçe ve İngilizce embedding: intfloat/multilingual-e5-base modeli; GPU varsa
  otomatik olarak CUDA üzerinde çalışır.
- Gerçek çevrimdışı çalışma: model lokalde mevcutsa Hugging Face açılışta otomatik
  çevrimdışı moda alınır, internet yokken ağ çağrısı denenmez.
- Yanıtı tek tıkla panoya kopyalama düğmesi.
- Yanıt veren arayüz: tüm ağır işlemler (OCR, embedding, dil modeli çağrıları) arka
  plan iş parçacıklarında yürütülür.
- Windows için tek klasörlük .exe paketleme desteği (PyInstaller).

---

## Mimari

Proje, SOLID prensiplerine uygun olarak modüllere ayrılmıştır. Her modülün tek bir
sorumluluğu vardır ve diğerlerine sıkı bağlı değildir.

```
maritime-doc-assistant/
├── main.py                 Giriş noktası, ana pencere, modül koordinasyonu, thread yönetimi
├── ui_components.py        Arayüz bileşenleri (sohbet rayı, sohbet alanı, Modeller/İndirilenler/Ayarlar)
├── document_processor.py   PDF okuma, metin parçalama, görsel/taranmış PDF OCR
├── embedding_manager.py    Lokal embedding, ChromaDB, benzerlik araması (RAG retrieval)
├── llm_connector.py        Gemini / gömülü yerel motor bağlantısı ve prompt yönetimi
├── model_manager.py        GGUF model kataloğu, indirme (ilerleme/iptal), silme
├── chat_store.py           Kalıcı sohbet geçmişi (JSON oturumlar)
├── perf_monitor.py         CPU/RAM/GPU/VRAM örnekleme ve performans bloğu biçimlendirme
├── benchmark.py            Uçtan uca performans ölçüm (benchmark) aracı (komut satırı)
├── requirements.txt        Bağımlılıklar
├── gemi_asistani.spec      PyInstaller paketleme yapılandırması
└── build.bat               Tek komutla .exe derleme betiği
```

- `document_processor` dil modeli veya veritabanından habersizdir.
- `ui_components` yalnızca sunum katmanıdır; iş mantığı içermez.
- `llm_connector` sağlayıcıları Strategy deseni ile soyutlar; yeni bir sağlayıcı
  eklemek mevcut kodu değiştirmeyi gerektirmez.

---

## Gereksinimler

- Python 3.11 veya 3.12 (önerilir). PyTorch, EasyOCR ve sentence-transformers için
  Python 3.13/3.14'te kararlı paketler henüz bulunmayabilir.
- Windows 10/11.
- Çevrimdışı dil modeli için harici program GEREKMEZ; motor (llama-cpp-python)
  uygulamaya gömülüdür. GGUF modelleri arayüzden indirilir.

---

## Kurulum

```powershell
git clone https://github.com/cemalozcelik/maritime-doc-assistant.git
cd maritime-doc-assistant

py -3.11 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

GPU (NVIDIA CUDA) ile hızlı çevrimdışı çıkarım isterseniz, gömülü motoru CUDA
destekli wheel ile kurun (torch ile aynı CUDA sürümü; örn. CUDA 12.1):

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

İlk çalıştırmada embedding modeli (yaklaşık 1,1 GB) ve ilk OCR işleminde EasyOCR
modelleri bir kez indirilir. Yerel dil modeli (GGUF) ise "İndirilenler" sekmesinden
bir kez indirilir. Bu indirmelerden sonra uygulama tamamen internetsiz çalışır.

---

## Kullanım

1. Doküman ekleyin: "Ayarlar" sekmesinden "Dosya Yükle" veya "Klasör Yükle" düğmesini
   kullanın, ya da dosya/klasörü doğrudan pencereye sürükleyip bırakın.
2. Dil modelini seçin:
   - Gemini (çevrimiçi): "Modeller" sekmesinde sağlayıcıyı Gemini yapıp API anahtarınızı
     girin.
   - Yerel Model (çevrimdışı): "İndirilenler" sekmesinden bir GGUF model indirin
     (ör. Qwen2.5 7B), ardından "Modeller" sekmesinde "Aktif yerel model" olarak seçin.
3. Soru sorun: "Sohbet" sekmesindeki metin kutusuna yazıp "Gönder" düğmesine basın.
   Yanıtın altında kullanılan kaynaklar ve performans bloğu listelenir.
4. Geçmiş sohbetler: sol panelde listelenir; "+ Yeni Sohbet" ile yeni başlatabilir,
   bir geçmişe tıklayıp devam edebilir veya silebilirsiniz.

---

## Performans Ölçümü

Tez/rapor için RAG hattının performansı iki yoldan ölçülebilir:

- Uygulama içi: her cevabın sonuna otomatik olarak eklenen performans bloğu
  (token, süre, hız ve CPU/RAM/GPU/VRAM ortalama/zirve değerleri ile model adı,
  parametre sayısı ve quantization). Temiz ölçüm için uygulama, model başına
  açılıp kapatılarak kullanılır; ölçüm soru gönderildiği andan cevap gelene kadarki
  kullanımı kapsar.
- Komut satırı: birden fazla soruyu (ve tekrarları) toplu çalıştırıp sonuçları
  tablo ve CSV olarak veren `benchmark.py`:

```powershell
# Yerel (gömülü llama.cpp) - GGUF dosya yolu verilir:
python benchmark.py --provider local --model data/models_gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf --repeat 3

# Gemini:
python benchmark.py --provider gemini --model gemini-2.5-pro --api-key XYZ
```

GPU/VRAM ölçümü için `psutil` ve `nvidia-ml-py` paketleri gerekir (requirements
içinde yer alır). NVIDIA GPU yoksa ilgili metrikler atlanır.

Not: Bir dil modelinin GPU ile hızlı çalışabilmesi için VRAM'e sığması gerekir.
VRAM'i aşan modellerin bir kısmı CPU'ya taşar ve hız belirgin biçimde düşer; bu
durum performans bloğundaki düşük GPU kullanımı ve yüksek RAM değerlerinden
gözlemlenebilir.

---

## Tamamen Çevrimdışı Dağıtım

Hedef bilgisayarda internet hiç bulunmayacaksa modeller önceden indirilip projeyle
birlikte taşınmalıdır.

Embedding modelini projeye dahil etme:

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/multilingual-e5-base").save(
    "models/intfloat_multilingual-e5-base"
)
```

`main.py` açılışta önce `models/<model-adı>` klasörünü kontrol eder; mevcutsa modeli
internetsiz olarak buradan yükler.

EasyOCR modelleri ilk OCR işleminde `C:\Users\<kullanıcı>\.EasyOCR\model` dizinine
iner. Bu dizin hedef bilgisayarda aynı konuma kopyalanmalı veya kod içinde
`model_storage_directory` parametresiyle yönlendirilmelidir.

Yerel dil modeli (GGUF): "İndirilenler" sekmesinden indirilen GGUF dosyaları, uygulamanın
yanındaki `data/models_gguf/` klasörüne yazılır. İnternetin hiç bulunmayacağı bir hedef
bilgisayar için bu klasörü olduğu gibi kopyalamak yeterlidir; model orada bulunursa
indirme gerekmez.

---

## Windows .exe Paketleme

Hazır yapılandırma ile:

```powershell
pyinstaller gemi_asistani.spec --clean
```

Alternatif olarak `build.bat` betiği aynı işlemi yapar. Çıktı
`dist\GemiAsistani\GemiAsistani.exe` konumunda oluşur; dağıtım için klasörün tamamı
kopyalanmalıdır.

Paketleme sırasında dikkat edilmesi gereken noktalar:

- Tek dosya (`--onefile`) yerine klasör (`onedir`) modu kullanılır. torch,
  transformers ve chromadb büyük veri içerdiğinden tek dosya modu açılışı yavaşlatır
  ve kararsızlık yaratabilir.
- `--collect-all` zorunludur: CustomTkinter temaları, ChromaDB dinamik modülleri,
  torch/onnxruntime kütüphaneleri, EasyOCR ve tkinterdnd2 ikili dosyaları otomatik
  algılanmaz; spec dosyası bunları toplar.
- Veritabanı uygulama paketine gömülmez. `main.py`, ChromaDB verilerini çalıştırılabilir
  dosyanın yanındaki yazılabilir `data` klasörüne yazar.
- Embedding ve OCR modelleri ayrıca dahil edilmelidir (bkz. Tamamen Çevrimdışı Dağıtım).
- Gömülü motor (llama-cpp-python) spec'te `collect_all` ile toplanır. CUDA wheel'i
  kullanıldığında `ggml-cuda.dll` çok büyüktür (~700 MB) ve paket boyutunu belirgin
  artırır; sadece CPU dağıtımı için CPU wheel'i tercih edilebilir. CUDA wheel'i çalışma
  anında cudart/cublas DLL'lerine ihtiyaç duyar; bunlar torch ile birlikte gelir ve
  uygulama import öncesi `torch/lib` klasörünü DLL arama yoluna ekler.
- GGUF modelleri pakete gömülmez; çalışma anında `data/models_gguf/`'a indirilir.
- UPX sıkıştırması kapalıdır; torch/onnxruntime kütüphanelerini bozabilir.

---

## Teknoloji Yığını

| Katman            | Teknoloji                                            |
|-------------------|------------------------------------------------------|
| Arayüz            | CustomTkinter, tkinterdnd2                           |
| PDF okuma/render  | PyMuPDF (fitz)                                        |
| OCR               | EasyOCR (Türkçe + İngilizce)                          |
| Metin parçalama   | LangChain RecursiveCharacterTextSplitter             |
| Embedding         | sentence-transformers, intfloat/multilingual-e5-base |
| Vektör veritabanı | ChromaDB (persistent)                                |
| Dil modeli        | Google Gemini API (google-genai), gömülü llama.cpp (llama-cpp-python, GGUF) |
| Model indirme     | huggingface_hub, requests (HF resolve, akışlı)       |
| Sohbet geçmişi    | JSON oturum dosyaları (yerel disk)                   |
| Performans ölçümü | psutil, nvidia-ml-py (pynvml)                        |
| Paketleme         | PyInstaller                                          |

---

## Lisans

Bu depo için uygun bir lisans dosyası eklenebilir (örneğin MIT).

Gemini sağlayıcısı internet bağlantısı ve geçerli bir API anahtarı gerektirir.
Çevrimdışı çalışmanın tamamı gömülü yerel motor (llama.cpp / GGUF), lokal embedding
ve OCR üzerinden sağlanır; harici bir program gerekmez.
