$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$docPath = Join-Path $taskRoot '稿件\MPSMA-DEIM_conference_EN_revised.docx'
$qaDir = Join-Path $PSScriptRoot 'rendered'
New-Item -ItemType Directory -Path $qaDir -Force | Out-Null
$pdfPath = Join-Path $qaDir 'manuscript.pdf'
$wordApp = New-Object -ComObject Word.Application
$wordApp.Visible = $false
$wordApp.DisplayAlerts = 0
try {
    $renderDoc = $wordApp.Documents.Open($docPath, $false, $true)
    $renderDoc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    $pageCount = $renderDoc.ComputeStatistics(2)
    $renderDoc.Close(0)
    Write-Output "Word PDF exported; pages=$pageCount"
} finally {
    $wordApp.Quit(0)
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordApp) | Out-Null
}
& 'C:\Users\jiwei\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -scale-to 1500 -png $pdfPath (Join-Path $qaDir 'page')
if ($LASTEXITCODE -ne 0) { throw 'Poppler render failed' }
