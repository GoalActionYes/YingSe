param(
    [Parameter(Mandatory)][string]$Cli,
    [Parameter(Mandatory)][string]$InputDirectory,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [ValidateSet('off', 'on')][string]$State = 'off'
)

$ErrorActionPreference = 'Stop'
$Cli = (Resolve-Path -LiteralPath $Cli).Path
$InputDirectory = (Resolve-Path -LiteralPath $InputDirectory).Path
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$config = (Join-Path $OutputDirectory 'config').Replace('\', '/')
$cache = (Join-Path $OutputDirectory 'cache').Replace('\', '/')
New-Item -ItemType Directory -Force -Path $config, $cache | Out-Null

$images = [ordered]@{ A = 'DSCF0419.RAF'; B = 'P1005231.RW2' }
foreach ($group in $images.Keys) {
    $image = (Join-Path $InputDirectory $images[$group]).Replace('\', '/')
    $sidecar = (Join-Path $InputDirectory "$group-$State.xmp").Replace('\', '/')
    $destination = (Join-Path $OutputDirectory "$group-$State.png").Replace('\', '/')
    $log = Join-Path $OutputDirectory "$group-$State.log"
    if (Test-Path -LiteralPath $destination) { throw "Output exists: $destination" }

    & $Cli $image $sidecar $destination --width 1024 --height 1024 `
        --hq true --apply-custom-presets false --icc-type srgb `
        --core --configdir $config --cachedir $cache --disable-opencl -t 2 `
        --conf 'write_sidecar_files=never' --conf 'plugins/imageio/format/png/bpp=8' *> $log
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        throw "darktable export failed for $group-$State. See $log"
    }
    Get-Item -LiteralPath $destination | Select-Object Name, Length
}
