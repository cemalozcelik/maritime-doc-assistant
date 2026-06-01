@echo off
REM ===========================================================================
REM  Gemi Teknik Dokuman Asistani - .exe paketleme betigi
REM  Kullanim: Sanal ortam aktifken bu dosyayi cift tiklayin veya calistirin.
REM ===========================================================================
echo [1/3] Eski derlemeler temizleniyor...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [2/3] PyInstaller ile paketleniyor (bu islem birkac dakika surebilir)...
pyinstaller gemi_asistani.spec --clean --noconfirm
if errorlevel 1 (
    echo HATA: Paketleme basarisiz oldu.
    pause
    exit /b 1
)

echo [3/3] Tamamlandi!
echo Calistirilabilir dosya: dist\GemiAsistani\GemiAsistani.exe
echo Dagitim icin dist\GemiAsistani klasorunun TAMAMINI kopyalayin.
pause
