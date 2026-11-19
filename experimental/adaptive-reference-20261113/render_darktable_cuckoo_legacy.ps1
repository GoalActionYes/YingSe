param(
    [Parameter(Mandatory)][string]$Cli,
    [Parameter(Mandatory)][string]$Raw,
    [Parameter(Mandatory)][string]$SidecarDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateSet('on', 'off', 'on-repeat', 'off-repeat')][string]$State
)

$ErrorActionPreference = 'Stop'
$cliPath = (Resolve-Path -LiteralPath $Cli).Path
$rawPath = (Resolve-Path -LiteralPath $Raw).Path
$sidecars = (Resolve-Path -LiteralPath $SidecarDirectory).Path
$out = (Resolve-Path -LiteralPath $OutputDirectory).Path
$paths = @($cliPath, $rawPath, $sidecars, $out)
if ($paths -match '[^\x00-\x7F]') { throw 'Use the verified ASCII workspace path mapping' }

$arm = ($State -split '-', 2)[0]
$sidecar = Join-Path $sidecars $(if ($arm -eq 'on') { 'author.xmp' } else { 'off.xmp' })
$hashes = @{
    $cliPath = '4db9777f5fee0800d8fbb6f18673635263182e7f2783ed4162a9dfb9ecd73a3a'
    $rawPath = '7fbb1b6b9509e3ebb81aa9e76354c09ff1bf3002f465ef0364fcab6f8595304d'
    (Join-Path $sidecars 'author.xmp') = '578f776ad24571aded0878d286319b85c4f290b398ff18141f277858607332bb'
    (Join-Path $sidecars 'off.xmp') = 'ef95b1097a27ae604c82914a9ae2419fe9321fec837b6fb5138ab95c954686a8'
}
foreach ($file in $hashes.Keys) {
    if ((Get-FileHash -LiteralPath $file).Hash -ne $hashes[$file]) { throw "Unexpected artifact identity: $file" }
}

$png = Join-Path $out "$State.png"
$log = Join-Path $out "$State.log"
$config = Join-Path $out "config-$State"
$cache = Join-Path $out "cache-$State"
if ((Test-Path -LiteralPath $png) -or (Test-Path -LiteralPath $log) -or
    (Test-Path -LiteralPath $config) -or (Test-Path -LiteralPath $cache)) {
    throw "State output already exists: $State"
}
New-Item -ItemType Directory -Path $config,$cache | Out-Null
$renderArguments = @($rawPath, $sidecar, $png) | ForEach-Object { $_.Replace('\', '/') }
& $cliPath @renderArguments --width 1024 --height 1024 --hq true --apply-custom-presets false --icc-type srgb `
    --core --configdir $config.Replace('\', '/') --cachedir $cache.Replace('\', '/') --disable-opencl -t 2 `
    --conf 'write_sidecar_files=never' --conf 'plugins/imageio/format/png/bpp=8' *> $log
$status = $LASTEXITCODE
if ($status -ne 0 -or -not (Test-Path -LiteralPath $png -PathType Leaf)) {
    throw "darktable export failed: state=$State exit=$status. See $log"
}
Get-Item -LiteralPath $png | Select-Object Name,Length
