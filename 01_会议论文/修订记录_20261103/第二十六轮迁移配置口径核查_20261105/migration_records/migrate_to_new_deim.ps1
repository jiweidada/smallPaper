<#
.SYNOPSIS
    把《计算机应用与软件》投稿项目的自定义模块与配置迁移到新版 DEIM 发行包。

.DESCRIPTION
    源 (Src) : C:\Users\jiwei\Desktop\DEIM            —— 你跑论文实验的旧树
    目标(Dst): D:\BaiduNetdiskDownload\DEIM_1572862469 —— 下载的新版发行包

    只搬运“你独有的”代码与配置，绝不覆盖新版包里同名的 engine 文件
    （新版 engine 与你旧树有 233 个同名文件内容不同，直接覆盖会破坏新功能）。
    唯一需要改动的公共文件是 engine\backbone\__init__.py，脚本以“追加缺失
    import”的方式打补丁，幂等，可重复运行。

    默认是演练模式（只打印计划）。确认无误后加 -Apply 真正执行。

.EXAMPLE
    # 1) 先看计划
    .\migrate_to_new_deim.ps1
    # 2) 确认后执行
    .\migrate_to_new_deim.ps1 -Apply
#>
[CmdletBinding()]
param(
    [string]$Src = 'C:\Users\jiwei\Desktop\DEIM',
    [string]$Dst = 'D:\BaiduNetdiskDownload\DEIM_1572862469',
    [switch]$Apply,
    # 不加此开关：把你的 configs\test\visdrone2019.yml 存成
    # visdrone2019_mpsma.yml，并把复制过去的消融配置里的 __include__ 改指向它，
    # 从而不动新版包原有的 visdrone2019.yml（保留其 tiny_metrice 评估口径）。
    [switch]$OverwriteValCfg,
    [string]$BackupSuffix = ('bak-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------- 文件清单
# A. 自定义模块（论文创新点的全部实现）
$Modules = @(
    'engine\extre_module\custom_nn\attention\MSCA.py',          # MSCA / MSCA-Lite / MSCA-Dilated
    'engine\backbone\hgnetv2_MSCA_EnhancedStem.py',             # MPS + 分层 MSCA（论文主模型）
    'engine\backbone\hgnetv2_MSCA.py',                          # 消融：只用 MSCA
    'engine\backbone\hgnetv2_Silu.py',                          # 消融：只换 SiLU
    # 以下两个是可选的历史模块，没有被任何 backbone import，不需要可删掉
    'engine\extre_module\custom_nn\stem\EnhancedStem.py',
    'engine\extre_module\custom_nn\stem\MultiLoG.py'
)

# B. 消融 / 跨数据集实验配置
$ConfigsAblation = @(
    'configs\test\deim_hgnetv2_EnhancedStem_s_visdrone.yml',              # 表1 第8行 = 论文主模型
    'configs\test\deim_hgnetv2_EnhancedStem_n_visdrone.yml',
    'configs\test\deim_hgnetv2_EnhancedStem_m_visdrone.yml',
    'configs\test\deim_hgnetv2_EnhancedStem_x_visdrone.yml',
    'configs\test\deim_hgnetv2_EnhancedStem_NOMSCA_NOSilu_n_visdrone.yml',
    'configs\test\deim_hgnetv2_EnhancedStemNO_MSCA_NOSilu_n_visdrone.yml',
    'configs\test\deim_hgnetv2_EnhancedStemNO_NOMSCA_Silu_n_visdrone.yml',
    'configs\test\deim_hgnetv2_MSCA_n_visdrone.yml',
    'configs\test\deim_hgnetv2_MSCA_Silu_n_visdrone.yml',
    'configs\test\deim_hgnetv2_MSCA_DilatedGCSA_n_visdrone.yml',
    'configs\test\deim_hgnetv2_MSCA_Silu_DilatedGCSA_n_visdrone.yml',
    'configs\test\deim_hgnetv2_Silu_n_visdrone.yml',
    'configs\test\deim_hgnetv2_Simam_n_visdrone.yml',
    'configs\test\deim_hgnetv2_n_MultiLoGStem_Simam_DilatedGCSA_visdrone.yml',
    'configs\test\deim_hgnetv2_n_visdrone_losschange.yml',
    'configs\test\deim_hgnetv2_FSSA_n_custom.yml',
    'configs\test\deim_hgnetv2_EnhancedStem_s_uavdt.yml',
    'configs\test\deim_hgnetv2_s_uavdt.yml',
    'configs\test\deim_hgnetv2_n_uavdt.yml',
    'configs\test\deim_hgnetv2_s_tinyperson.yml',
    'configs\test\deim_hgnetv2_s_tinyperson_MPS.yml',
    'configs\test\tinyperson.yml',
    'configs\test\tinyperson_MPS.yml',
    'configs\test\uavdt.yml'
)

# C. 新版包缺失、必须补的数据集 / 基础配置
$ConfigsSupport = @(
    'configs\dataset\tinyperson_detection.yml',
    'configs\dataset\uavdt_detection.yml',
    'configs\deim\visdrone2019.yml',
    'configs\base\deim_losschange.yml'
)

$ValCfgRel = 'configs\test\visdrone2019.yml'

# ---------------------------------------------------------------- 前置检查
foreach ($p in @($Src, $Dst)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "路径不存在: $p" }
}
if (-not (Test-Path -LiteralPath (Join-Path $Dst 'train.py'))) {
    throw "$Dst 看起来不是 DEIM 根目录（找不到 train.py）"
}
if (-not (Test-Path -LiteralPath (Join-Path $Dst 'engine\backbone\__init__.py'))) {
    throw "$Dst 缺少 engine\backbone\__init__.py"
}

$mode = if ($Apply) { 'APPLY' } else { 'DRY-RUN（不写入，仅演练）' }
Write-Host ""
Write-Host "==== DEIM 自定义模块迁移 ====" -ForegroundColor Cyan
Write-Host "  源    : $Src"
Write-Host "  目标  : $Dst"
Write-Host "  模式  : $mode"
Write-Host "  评估配置: $(if ($OverwriteValCfg) { '覆盖新版包 visdrone2019.yml' } else { '另存为 visdrone2019_mpsma.yml（推荐）' })"
Write-Host ""

$all = @($Modules + $ConfigsAblation + $ConfigsSupport)
$missing = @($all | Where-Object { -not (Test-Path -LiteralPath (Join-Path $Src $_)) })
if ($missing.Count -gt 0) {
    Write-Host "警告：源目录里找不到以下文件，将跳过：" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "   $_" }
    $all = @($all | Where-Object { $_ -notin $missing })
}

# ---------------------------------------------------------------- 复制
$copied = 0; $skipped = 0
foreach ($rel in $all) {
    $s = Join-Path $Src $rel
    $d = Join-Path $Dst $rel
    $dDir = Split-Path -Parent $d
    $exists = Test-Path -LiteralPath $d
    if ($exists) {
        $bak = "$d.$BackupSuffix"
        Write-Host ("[备份+覆盖] {0}" -f $rel) -ForegroundColor Yellow
        if ($Apply) {
            if (-not (Test-Path -LiteralPath $bak)) { Copy-Item -LiteralPath $d -Destination $bak }
            Copy-Item -LiteralPath $s -Destination $d -Force
        }
        $copied++
    } else {
        Write-Host ("[新增]      {0}" -f $rel) -ForegroundColor Green
        if ($Apply) {
            if (-not (Test-Path -LiteralPath $dDir)) { New-Item -ItemType Directory -Path $dDir -Force | Out-Null }
            Copy-Item -LiteralPath $s -Destination $d
        }
        $copied++
    }
}

# ---------------------------------------------------------------- 评估配置
$valSrc = Join-Path $Src $ValCfgRel
if (Test-Path -LiteralPath $valSrc) {
    if ($OverwriteValCfg) {
        $d = Join-Path $Dst $ValCfgRel
        Write-Host "[评估配置]  覆盖 $ValCfgRel（新版包原文件已备份）" -ForegroundColor Yellow
        if ($Apply) {
            $bak = "$d.$BackupSuffix"
            if (-not (Test-Path -LiteralPath $bak)) { Copy-Item -LiteralPath $d -Destination $bak }
            Copy-Item -LiteralPath $valSrc -Destination $d -Force
        }
    } else {
        $newRel = 'configs\test\visdrone2019_mpsma.yml'
        $d = Join-Path $Dst $newRel
        Write-Host "[评估配置]  新增 $newRel（不动新版包原文件）" -ForegroundColor Green
        if ($Apply) {
            Copy-Item -LiteralPath $valSrc -Destination $d -Force
            # 把刚复制过去的消融配置里的 __include__ 改指到新文件名
            $patched = 0
            foreach ($rel in $ConfigsAblation) {
                $f = Join-Path $Dst $rel
                if (-not (Test-Path -LiteralPath $f)) { continue }
                $txt = Get-Content -LiteralPath $f -Raw -Encoding UTF8
                if ($txt -match "'visdrone2019\.yml'") {
                    $new = $txt -replace "'visdrone2019\.yml'", "'visdrone2019_mpsma.yml'"
                    Set-Content -LiteralPath $f -Value $new -Encoding UTF8 -NoNewline
                    $patched++
                }
            }
            Write-Host ("            已改写 {0} 个消融配置的 __include__" -f $patched)
        }
    }
}

# ---------------------------------------------------------------- 注册补丁
$initRel = 'engine\backbone\__init__.py'
$initPath = Join-Path $Dst $initRel
$need = @(
    @{ name = 'HGNetv2_Silu';                line = 'from .hgnetv2_Silu import HGNetv2_Silu' },
    @{ name = 'HGNetv2_MSCA';                line = 'from .hgnetv2_MSCA import HGNetv2_MSCA' },
    @{ name = 'HGNetv2_MSCA_EnhancedStem';   line = 'from .hgnetv2_MSCA_EnhancedStem import HGNetv2_MSCA_EnhancedStem' }
)
$initTxt = Get-Content -LiteralPath $initPath -Raw -Encoding UTF8
$toAdd = @($need | Where-Object { $initTxt -notmatch [regex]::Escape($_.name) })

Write-Host ""
if ($toAdd.Count -eq 0) {
    Write-Host "[注册补丁]  $initRel 已包含全部注册行，无需修改" -ForegroundColor Green
} else {
    Write-Host "[注册补丁]  需向 $initRel 追加 $($toAdd.Count) 行：" -ForegroundColor Yellow
    $toAdd | ForEach-Object { Write-Host "             $($_.line)" }
    if ($Apply) {
        Copy-Item -LiteralPath $initPath -Destination "$initPath.$BackupSuffix" -Force
        $append = "`r`n" + (($toAdd | ForEach-Object { $_.line }) -join "`r`n") + "`r`n"
        Add-Content -LiteralPath $initPath -Value $append -Encoding UTF8
        Write-Host "            已追加（原文件备份为 __init__.py.$BackupSuffix）" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- 汇总
Write-Host ""
Write-Host "==== 汇总 ====" -ForegroundColor Cyan
Write-Host ("  计划/已处理文件 : {0}" -f $copied)
if (-not $Apply) {
    Write-Host ""
    Write-Host "  以上为演练结果。确认后执行：" -ForegroundColor Yellow
    Write-Host "    .\migrate_to_new_deim.ps1 -Apply"
}
Write-Host ""
Write-Host "迁移后仍需手工处理（见 MIGRATION_CHECKLIST.md）：" -ForegroundColor Cyan
Write-Host "  1. 数据路径：新版包配置写的是 /root/dataset/... (Linux)，需改成你的实际路径"
Write-Host "  2. 评估口径：新版包 visdrone2019.yml 带 using_tiny_metrice=True 且 val 指向 test-dev，"
Write-Host "     与论文 AP 口径不一致，必须用迁移过来的 visdrone2019_mpsma.yml"
Write-Host "  3. 环境：新版包要求 pycocotools==2.0.11 / timm==1.0.27 / faster-coco-eval 等"
Write-Host ""
