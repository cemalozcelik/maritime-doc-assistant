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
   modeline (çevrimiçi Gemini veya çevrimdışı Ollama) gönderilerek yanıt üretilir.

Tüm embedding ve OCR işlemleri lokalde çalışır; internet yalnızca Gemini sağlayıcısı
seçildiğinde gereklidir.

---

## Temel Özellikler

- Çift dil modeli sağlayıcısı: internet varken Gemini API (varsayılan
  gemini-2.5-pro), yokken Ollama (llama3, mistral, gemma vb.). Arayüzden tek tıkla
  geçiş. Gemini bağlantısı güncel google-genai SDK'sı ile kurulur.
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
├── ui_components.py        CustomTkinter arayüz bileşenleri (sol panel ve sohbet alanı)
├── document_processor.py   PDF okuma, metin parçalama, görsel/taranmış PDF OCR
├── embedding_manager.py    Lokal embedding, ChromaDB, benzerlik araması (RAG retrieval)
├── llm_connector.py        Gemini / Ollama bağlantısı ve prompt yönetimi (RAG generation)
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
- Çevrimdışı dil modeli için Ollama (isteğe bağlı).

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

İlk çalıştırmada embedding modeli (yaklaşık 1,1 GB) ve ilk OCR işleminde EasyOCR
modelleri bir kez indirilir. Sonraki çalıştırmalar internetsiz gerçekleşir.

---

## Kullanım

1. Doküman ekleyin: sol panelden "Dosya Yükle" veya "Klasör Yükle" düğmesini kullanın,
   ya da dosya/klasörü doğrudan pencereye sürükleyip bırakın.
2. Dil modelini seçin:
   - Gemini (çevrimiçi): API anahtarınızı girin.
   - Ollama (çevrimdışı): "ollama serve" çalışırken "Yenile" düğmesine basıp modeli seçin.
3. Soru sorun: alttaki metin kutusuna yazıp "Gönder" düğmesine basın. Yanıtın altında
   kullanılan kaynaklar listelenir.

Çevrimdışı dil modeli için hazırlık (internet varken bir kez):

```powershell
ollama pull llama3      # alternatif: mistral, gemma
```

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
# Ollama (lokal):
python benchmark.py --provider ollama --model llama3.1:8b --repeat 3

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
| Dil modeli        | Google Gemini API (google-genai), Ollama (lokal)     |
| Performans ölçümü | psutil, nvidia-ml-py (pynvml)                        |
| Paketleme         | PyInstaller                                          |

---

## Lisans

Bu depo için uygun bir lisans dosyası eklenebilir (örneğin MIT).

Gemini sağlayıcısı internet bağlantısı ve geçerli bir API anahtarı gerektirir.
Çevrimdışı çalışmanın tamamı Ollama ile lokal embedding ve OCR üzerinden sağlanır.
