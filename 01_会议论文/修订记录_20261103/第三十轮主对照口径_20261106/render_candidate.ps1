#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$candidate = Join-Path $PSScriptRoot 'MPSMA-DEIM_conference_EN_revised_第三十轮候选.docx'
$qaDir = Join-Path $PSScriptRoot 'rendered_word'
New-Item -ItemType Directory -Path $qaDir -ErrorAction Stop | Out-Null
$pdfPath = Join-Path $qaDir 'manuscript.pdf'
$wordApp = New-Object -ComObject Word.Application
try {
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $renderDoc = $wordApp.Documents.Open($candidate, $false, $true)
    try {
        $renderDoc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
        $pageCount = $renderDoc.ComputeStatistics(2)
    } finally {
        $renderDoc.Close(0)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($renderDoc) | Out-Null
    }
} finally {
    $wordApp.Quit(0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}
$pdftoppm = 'C:\Users\jiwei\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'
& $pdftoppm -scale-to 1800 -png $pdfPath (Join-Path $qaDir 'page')
if ($LASTEXITCODE -ne 0) { throw 'Poppler page rendering failed' }
$pngCount = @(Get-ChildItem -LiteralPath $qaDir -Filter 'page-*.png' -File).Count
if ($pngCount -ne $pageCount) { throw "Page count mismatch. Word=$pageCount, PNG=$pngCount" }
Write-Output "Word read-only export rendered $pngCount pages"
