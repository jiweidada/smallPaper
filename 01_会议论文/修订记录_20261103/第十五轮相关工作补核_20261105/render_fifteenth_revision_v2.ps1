#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$stage = $PSScriptRoot
$candidate = Join-Path $stage 'MPSMA-DEIM_conference_EN_revised_第十五轮候选_v2.docx'
$qaDir = Join-Path $stage 'rendered_word_v2'
if (Test-Path -LiteralPath $qaDir) { throw 'Render directory already exists' }
New-Item -ItemType Directory -Path $qaDir -ErrorAction Stop | Out-Null
$pdfPath = Join-Path $qaDir 'manuscript.pdf'
Start-Transcript -Path (Join-Path $stage 'render_word_v2.log') | Out-Null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    try {
        $renderDoc = $wordApp.Documents.Open($candidate, $false, $true)
        try {
            $renderDoc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
            $pageCount = $renderDoc.ComputeStatistics(2)
            Write-Output "Word PDF exported read-only; pages=$pageCount"
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
    if ($pngCount -ne $pageCount) { throw "Page count mismatch: Word=$pageCount, PNG=$pngCount" }
    Write-Output "Rendered PNG pages=$pngCount"
} finally {
    Stop-Transcript | Out-Null
}
