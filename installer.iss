; ============================================================================
;  Inno Setup betiği - Gemi Teknik Doküman Asistanı
; ----------------------------------------------------------------------------
;  Bu betik, PyInstaller'ın ürettiği 'dist\GemiAsistani' klasörünü tek bir
;  kurulum dosyasına ('GemiAsistani-Kurulum.exe') paketler. Hedef bilgisayarda
;  bu tek dosya çalıştırılınca uygulama bir kez kalıcı yere kurulur ve sonra
;  HIZLI açılır (onefile gibi her açılışta temp'e açma yoktur).
;
;  Önkoşullar:
;    1. Inno Setup 6 kurulu olmalı: https://jrsoftware.org/isinfo.php
;    2. Önce uygulamayı derleyin:
;         pyinstaller gemi_asistani.spec --clean --noconfirm
;    3. Sonra bu betiği derleyin:
;         - Inno Setup ile installer.iss'i açıp "Compile" deyin, VEYA
;         - Komut satırı: "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
;    4. Çıktı: installer_output\GemiAsistani-Kurulum.exe
;
;  Not: Kurulum admin GEREKTİRMEZ; uygulama kullanıcının profiline kurulur
;  (Program Files yerine). Böylece uygulama klasörü yazılabilir kalır ve
;  modeller/sohbetler 'data' alt klasörüne sorunsuz iner. Kullanıcı isterse
;  kurulum sırasında "tüm kullanıcılar" (admin) seçeneğine geçebilir.
;
;  ÖNEMLİ: 'dist\GemiAsistani\data' klasörü (test sırasında indirilmiş modeller)
;  pakete DAHİL EDİLMEZ; hedefte modeller arayüzden indirilir.
; ============================================================================

#define MyAppName "Gemi Teknik Doküman Asistanı"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Gemi Asistanı Projesi"
#define MyAppExeName "GemiAsistani.exe"
#define MySourceDir "dist\GemiAsistani"

[Setup]
; AppId benzersiz olmalı; sürümler arası AYNI kalmalı (güncelleme/kaldırma için).
AppId={{8B7A3C21-4E9D-4F6A-9C2B-1D5E7A0F3B62}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\GemiAsistani
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Admin istemeden, kullanıcı profiline kur (uygulama klasörü yazılabilir kalsın).
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=installer_output
OutputBaseFilename=GemiAsistani-Kurulum
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Yalnızca 64-bit Windows (uygulama 64-bit). Inno Setup 6.3+ gerektirir.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Tüm uygulama dosyaları (exe + _internal). 'data' (indirilmiş modeller) HARİÇ.
Source: "{#MySourceDir}\*"; DestDir: "{app}"; \
    Excludes: "data\*,data"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
    Flags: nowait postinstall skipifsilent
