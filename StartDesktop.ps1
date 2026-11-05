param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$desktopDir = Join-Path $projectDir 'desktop'
$errorFile = Join-Path $desktopDir 'last-start-error.txt'
try {
    $runtime = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $runtime)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw 'Python runtime missing. Read README.md for installation requirements.' }
        $runtime = $pythonCommand.Source
    }
    $edgeCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'),
        (Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\Edge\Application\msedge.exe')
    )
    $edge = $edgeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $edge) { throw 'Microsoft Edge is missing. Use StartDemo.ps1 to open the browser version.' }
    $url = 'http://127.0.0.1:8765'
    $info = $null
    try { $info = Invoke-RestMethod -Uri "$url/api/info" -TimeoutSec 3 } catch {}
    if ($null -ne $info -and $info.palette_version -ne 'cinematic-v2') {
        throw 'Port 8765 is running an incompatible service. Close the older service before launching this version.'
    }
    if ($null -eq $info) {
        $ownedServer = Start-Process -FilePath $runtime -ArgumentList '"prototype/server.py"' -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $projectDir 'prototype/server.log') -RedirectStandardError (Join-Path $projectDir 'prototype/server.err.log')
        for ($attempt = 0; $attempt -lt 40; $attempt++) {
            Start-Sleep -Milliseconds 300
            if ($ownedServer.HasExited) { throw 'Server exited during startup. Read prototype/server.err.log.' }
            try {
                $info = Invoke-RestMethod -Uri "$url/api/info" -TimeoutSec 2
                if ($info.palette_version -eq 'cinematic-v2') { break }
            } catch {}
        }
    }
    if ($null -eq $info -or $info.palette_version -ne 'cinematic-v2') {
        throw 'Server did not become ready. Read prototype/server.err.log.'
    }
    if (-not $info.ffmpeg) { throw 'FFmpeg is unavailable. Read README.md before using video export.' }
    $check = [ordered]@{
        checked_local = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
        palette_version = $info.palette_version
        server_pid = $info.pid
        ffmpeg = $info.ffmpeg
        python = $runtime
        window_host = $edge
        mode = 'local desktop preview; external Python and Edge required'
    }
    $check | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $desktopDir 'startup-check.json') -Encoding UTF8
    if ($CheckOnly) { exit 0 }
    $profile = Join-Path $desktopDir 'profile'
    # A visible application window is the requested deliverable. The Python service remains hidden.
    Start-Process -FilePath $edge -ArgumentList @("--app=$url/?desktop=1", "--user-data-dir=`"$profile`"", '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--window-size=1440,960')
} catch {
    $_.Exception.Message | Set-Content -LiteralPath $errorFile -Encoding UTF8
    exit 1
}
