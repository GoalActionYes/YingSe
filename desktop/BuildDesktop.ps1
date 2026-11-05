$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Add-Type -AssemblyName System.Drawing
$bitmap = New-Object System.Drawing.Bitmap 64,64
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.Clear([System.Drawing.Color]::FromArgb(18,23,21))
$brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(197,232,155))
$graphics.FillEllipse($brush,12,12,40,40)
$inner = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(42,64,38))
$graphics.FillEllipse($inner,26,12,26,40)
$icon = [System.Drawing.Icon]::FromHandle($bitmap.GetHicon())
$stream = [System.IO.File]::Create((Join-Path $PSScriptRoot 'yingse.ico'))
$icon.Save($stream)
$stream.Dispose(); $icon.Dispose(); $graphics.Dispose(); $bitmap.Dispose(); $brush.Dispose(); $inner.Dispose()
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$output = Join-Path $root '映色.exe'
& $compiler /nologo /target:winexe /optimize+ /reference:System.Windows.Forms.dll "/win32icon:$(Join-Path $PSScriptRoot 'yingse.ico')" "/out:$output" (Join-Path $PSScriptRoot 'AppLauncher.cs')
if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed.' }
Write-Output $output
