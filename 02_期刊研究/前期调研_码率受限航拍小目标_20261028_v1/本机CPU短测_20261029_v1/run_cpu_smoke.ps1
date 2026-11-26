$ErrorActionPreference = 'Stop'
$runtime = 'C:\Users\jiwei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not (Test-Path -LiteralPath $runtime)) { throw 'The recorded Python runtime was not found.' }
& $runtime -I (Join-Path $PSScriptRoot 'run_bounded.py') --name ('offline_manual_' + $runTag) --timeout 120 -- $runtime -I (Join-Path $PSScriptRoot 'offline_checks.py')
if ($LASTEXITCODE -ne 0) { throw 'Offline checks failed; see the matching log.' }
& $runtime -I (Join-Path $PSScriptRoot 'run_bounded.py') --name ('images_manual_' + $runTag) --timeout 120 -- $runtime -I (Join-Path $PSScriptRoot 'image_checks.py')
if ($LASTEXITCODE -ne 0) { throw 'Image checks failed; see the matching log.' }
& $runtime -I (Join-Path $PSScriptRoot 'run_bounded.py') --name ('model_manual_' + $runTag) --timeout 300 -- $runtime -I (Join-Path $PSScriptRoot 'local_entry.py') --isolated-cpu (Join-Path $PSScriptRoot 'cpu_model_smoke.py')
if ($LASTEXITCODE -ne 0) { throw 'Model checks failed; see the matching log.' }
Write-Output 'PASS: local CPU functional checks; no training or research-effect validation.'
