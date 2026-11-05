$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$runtime = 'C:\Users\31258\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $runtime)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw 'Python runtime missing. See README.md.' }
    $runtime = $pythonCommand.Source
}
$ready = $false
try {
    $info = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/info' -TimeoutSec 2
    $ready = $null -ne $info.labels
} catch {}
if (-not $ready) {
    Start-Process -FilePath $runtime -ArgumentList '"prototype/server.py"' -WorkingDirectory $projectDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectDir 'prototype/server.log') -RedirectStandardError (Join-Path $projectDir 'prototype/server.err.log')
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 300
        try {
            $info = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/info' -TimeoutSec 1
            if ($null -ne $info.labels) { $ready = $true; break }
        } catch {}
    }
}
if (-not $ready) { throw 'Server failed. Inspect prototype/server.err.log.' }
Start-Process 'http://127.0.0.1:8765'
