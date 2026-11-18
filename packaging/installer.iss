; StreamClipper installer (Inno Setup 6). Built by scripts\build.ps1 from build\app:
;   ISCC /DAppVersion=x.y.z packaging\installer.iss  ->  dist\StreamClipper-Setup-x.y.z.exe
; Per-user install: no admin prompt, and config.yaml + data\ live in the app folder where the app can write them.
; Shortcuts start the official signed Python runtime (runtime\pythonw.exe), so Smart App Control trusts the app.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6E1F3A52-6C0B-4E0F-9C55-5D1B0C7A2F11}
AppName=StreamClipper
AppVersion={#AppVersion}
AppPublisher=StreamClipper contributors
AppPublisherURL=https://github.com/jebarson-caleb/streamclipper
AppSupportURL=https://github.com/jebarson-caleb/streamclipper/issues
DefaultDirName={localappdata}\Programs\StreamClipper
DefaultGroupName=StreamClipper
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=StreamClipper-Setup-{#AppVersion}
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\icon.ico
UninstallDisplayName=StreamClipper
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\build\app\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\StreamClipper"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m streamclipper desktop"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"; Comment: "Clip your live streams into Shorts"
Name: "{group}\Uninstall StreamClipper"; Filename: "{uninstallexe}"
Name: "{userdesktop}\StreamClipper"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m streamclipper desktop"; WorkingDir: "{app}"; IconFilename: "{app}\icon.ico"; Comment: "Clip your live streams into Shorts"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m streamclipper desktop"; WorkingDir: "{app}"; Description: "Start StreamClipper"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM llama-server.exe /F"; Flags: runhidden; RunOnceId: "StopAI"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"

[Messages]
FinishedLabel=StreamClipper is installed. On first launch it downloads its AI models (about 2.5 GB, once) and shows a short setup guide.
