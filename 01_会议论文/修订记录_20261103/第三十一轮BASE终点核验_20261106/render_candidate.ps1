#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$candidate = Join-Path $PSScriptRoot 'MPSMA-DEIM_conference_EN_revised_第三十一轮候选.docx'
$qaDir = Join-Path $PSScriptRoot 'rendered_word'
New-Item -ItemType Directory -Path $qaDir -ErrorAction Stop | Out-Null
$pdfPath = Join-Path $qaDir 'manuscript.pdf'
$word = New-Object -ComObject Word.Application
try {
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($candidate, $false, $true)
    try {
        $doc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
        $pages = $doc.ComputeStatistics(2)
    } finally {
        $doc.Close(0)
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
} finally {
    $word.Quit(0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
$pdftoppm = 'C:\Users\jiwei\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'
& $pdftoppm -scale-to 1800 -png $pdfPath (Join-Path $qaDir 'page')
if ($LASTEXITCODE -ne 0 -or @(Get-ChildItem -LiteralPath $qaDir -Filter 'page-*.png' -File).Count -ne $pages) {
    throw 'Word PDF export or page rasterization failed'
}
Write-Output "Word read-only export rendered $pages pages"
