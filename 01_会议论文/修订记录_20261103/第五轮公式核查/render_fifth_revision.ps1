$ErrorActionPreference = 'Stop'
$docPath = Join-Path $PSScriptRoot 'MPSMA-DEIM_conference_EN_revised_第五轮候选.docx'
$qaDir = Join-Path $PSScriptRoot 'rendered_word'
New-Item -ItemType Directory -Path $qaDir -Force | Out-Null
$pdfPath = Join-Path $qaDir 'manuscript.pdf'
$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    $renderDoc = $wordApp.Documents.Open($docPath, $false, $true)
    try {
        $renderDoc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
        $pageCount = $renderDoc.ComputeStatistics(2)
        Write-Output "Word PDF exported; pages=$pageCount"
    } finally {
        $renderDoc.Close(0)
    }
} finally {
    $wordApp.Quit(0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}
& 'C:\Users\jiwei\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -scale-to 1800 -png $pdfPath (Join-Path $qaDir 'page')
if ($LASTEXITCODE -ne 0) { throw 'Poppler render failed' }
