param([Parameter(Mandatory)][string]$Base,[Parameter(Mandatory)][string]$Output)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $Output) {throw 'Output directory exists'}
$previous=Join-Path $Base 'playraw-rendered-input-operator-v1'
$cli=Join-Path $Base 'playraw-mixed-light-skin-v1/runtime/bin/darktable-cli.exe'
$expected='51865b25c91b2b936d249b8d8c71e6de58dcc7cdef7768ab9082781bc401d179'
if ((Get-FileHash -LiteralPath $cli).Hash -ne $expected) {throw 'Unexpected editor'}
Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue
foreach($name in 'mixed','cuckoo-0','cuckoo-1') {
    $inputFolder=Join-Path $previous $name
    $folder=Join-Path $Output $name
    foreach($state in 'off','on') {
        $config=Join-Path $folder "config-$state"
        $cache=Join-Path $folder "cache-$state"
        New-Item -ItemType Directory -Path $config,$cache | Out-Null
        $inputFile=Join-Path $inputFolder 'rendered-input.tif'
        $sidecar=Join-Path $inputFolder "$state.xmp"
        $outputFile=Join-Path $folder "$state.tif"
        & $cli ($inputFile.Replace([char]92,[char]47)) ($sidecar.Replace([char]92,[char]47)) ($outputFile.Replace([char]92,[char]47)) --width 1024 --height 1024 --hq true --apply-custom-presets false --icc-type srgb --export_masks false --core --configdir ($config.Replace([char]92,[char]47)) --cachedir ($cache.Replace([char]92,[char]47)) --disable-opencl -t 2 -d perf --conf write_sidecar_files=never --conf plugins/imageio/format/tiff/bpp=8 --conf plugins/darkroom/sharpen/auto_apply=false *> (Join-Path $folder "$state.log")
        if ($LASTEXITCODE -or !(Test-Path -LiteralPath $outputFile)) {throw "Export failed for $name/$state"}
    }
}
