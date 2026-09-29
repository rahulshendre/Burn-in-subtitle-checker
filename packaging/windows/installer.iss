; Inno Setup script that turns the PyInstaller folder into one
; Subtitle-Checker-Setup.exe. build.ps1 runs it with the paths filled in:
;   iscc /DAppVersion=0.1.0 /DSourceDir=<dist\Subtitle Checker> /DOutputDir=<dist> installer.iss
; Installs for the current user only (no admin prompt), under
; %LOCALAPPDATA%\Programs, with a Start menu entry, an optional desktop
; shortcut and an uninstaller.

[Setup]
AppId={{3570E781-CFD0-474D-A8B9-317FA16E4FE0}
AppName=Subtitle Checker
AppVersion={#AppVersion}
DefaultDirName={autopf}\Subtitle Checker
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=Subtitle-Checker-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\Subtitle Checker.exe

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Subtitle Checker"; Filename: "{app}\Subtitle Checker.exe"
Name: "{autodesktop}\Subtitle Checker"; Filename: "{app}\Subtitle Checker.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Subtitle Checker.exe"; Description: "{cm:LaunchProgram,Subtitle Checker}"; Flags: nowait postinstall skipifsilent
