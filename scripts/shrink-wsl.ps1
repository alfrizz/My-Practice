<#
.SYNOPSIS
  One-click, fully-logged WSL shrink + cleanup.

.DESCRIPTION
  Deletes only:
    • /tmp entries older than 3h  
    • .ipynb_checkpoints dirs under ~/my_practice  
  Then runs fstrim, quiesces WSL, stops LxssManager+VMCompute, compacts ext4.vhdx,
  restarts services, warms up Ubuntu, and logs BEFORE/AFTER sizes + all outputs.
#>

#── 0) REFUSE UNC -----------------------------------------------------------
if ($PSScriptRoot -like '\\wsl.localhost\*') {
  Write-Error 'Run from Windows PowerShell on a local drive (e.g. G:), not \\wsl.localhost.' 
  exit 1
}

#── 1) ELEVATION ------------------------------------------------------------
if (-not ([Security.Principal.WindowsPrincipal] `
      ([Security.Principal.WindowsIdentity]::GetCurrent())
     ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Error 'Please rerun As Administrator.'; exit 1
}
$ErrorActionPreference = 'Stop'

#── 2) LOG SETUP & CWD ------------------------------------------------------
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$logFile = Join-Path $scriptDir 'shrink-wsl.log'
'' | Out-File -FilePath $logFile -Encoding ascii

function Write-Log($m) {
  "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" |
    Out-File -FilePath $logFile -Encoding ascii -Append
}

Write-Log '===== Shrink-WSL run started ====='
Write-Log "Script folder: $scriptDir"

#── 3) PATCH wsl.conf (remove invalid networkDrives) ------------------------
Write-Log 'Patching /etc/wsl.conf to drop invalid networkDrives key…'
wsl -d Ubuntu-22.04 --cd / -u root -- bash -c "sed -i '/^\s*networkDrives/d' /etc/wsl.conf" 2>$null

#── 4) LOCATE VHDX & LOG BEFORE ---------------------------------------------
$vhdx = "$env:LOCALAPPDATA\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx"
Write-Log "VHDX path: $vhdx"
if (-not (Test-Path $vhdx)) {
  Write-Log "ERROR: ext4.vhdx not found"; throw 'Missing VHDX'
}

function Log-Metrics($label) {
  $free = (Get-PSDrive C).Free  /1GB
  $size = (Get-Item $vhdx).Length /1GB
  Write-Log ("{0}: C free={1:N2} GB; ext4.vhdx={2:N2} GB" -f $label, $free, $size)
}
Log-Metrics 'BEFORE'

#── 5) STOP DOCKER DESKTOP (if running) -------------------------------------
Write-Log 'Stopping Docker Desktop…'
Stop-Process -Name 'Docker Desktop','com.docker.backend' `
  -Force -ErrorAction SilentlyContinue
Write-Log 'Docker Desktop stopped (if present)'

#── 6) SHUTDOWN WSL & WAIT vmmem -------------------------------------------
Write-Log 'Shutting down WSL…'
wsl --shutdown 2>$null
for ($i=0; $i -lt 10; $i++) {
  if (-not (Get-Process vmmem -ErrorAction SilentlyContinue)) {
    Write-Log 'vmmem exited'; break
  }
  Start-Sleep -Seconds 1
  Write-Log 'Waiting 1s for vmmem…'
}

#── 7) LIST & DELETE /tmp >3h ----------------------------------------------
Write-Log 'Listing /tmp items >3h…'
$tmpList = wsl -d Ubuntu-22.04 --cd / -u root -- bash -c 'find /tmp -mindepth 1 -mmin +180 -print' 2>$null
foreach ($p in $tmpList) { Write-Log "  [DEL] /tmp$p" }

Write-Log 'Deleting /tmp items…'
wsl -d Ubuntu-22.04 --cd / -u root -- bash -c 'find /tmp -mindepth 1 -mmin +180 -exec rm -rf {} + || true' 2>$null
Write-Log '/tmp cleanup done'

#── 8) LIST & DELETE .ipynb_checkpoints ------------------------------------
Write-Log 'Listing .ipynb_checkpoints dirs…'
$cpList = wsl -d Ubuntu-22.04 --cd / -u root -- bash -c `
  'find /home/alfrizz/my_practice -type d -name ".ipynb_checkpoints" -prune -print' 2>$null
foreach ($p in $cpList) { Write-Log "  [DEL] chkpt$p" }

Write-Log 'Deleting .ipynb_checkpoints…'
wsl -d Ubuntu-22.04 --cd / -u root -- bash -c `
  'find /home/alfrizz/my_practice -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} + || true' 2>$null
Write-Log 'Checkpoints cleanup done'

#── 9) FSTRIM --------------------------------------------------------------
Write-Log 'Running fstrim…'
$fOut = wsl -d Ubuntu-22.04 --cd / -u root -- fstrim -v / 2>$null
foreach ($l in $fOut) { Write-Log "  fstrim: $l" }

#── 10) SHUTDOWN WSL AGAIN -----------------------------------------------
Write-Log 'Shutting down WSL (for compaction)…'
wsl --shutdown 2>$null
Start-Sleep -Seconds 1
Write-Log 'WSL VM stopped'

#── 11) STOP HOST SERVICES -----------------------------------------------
Write-Log 'Stopping LxssManager & VMCompute…'
Stop-Service LxssManager,VMCompute `
  -Force -ErrorAction SilentlyContinue
Write-Log 'Host services stopped'

#── 12) COMPACT ext4.vhdx via DiskPart ------------------------------------
Write-Log "Compacting VHDX: $vhdx"
$tmpDp = [IO.Path]::GetTempFileName()
@"
select vdisk file="$vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@ | Set-Content -LiteralPath $tmpDp -Encoding ASCII

$dpLines = diskpart /s $tmpDp | Select-Object -Unique
foreach ($l in $dpLines) { Write-Log "  DiskPart: $l" }
Remove-Item $tmpDp -Force
Write-Log 'DiskPart compaction done'

#── 13) RESTART HOST SERVICES --------------------------------------------
Write-Log 'Restarting VMCompute & LxssManager…'
Start-Service VMCompute,LxssManager -ErrorAction SilentlyContinue
Write-Log 'Host services restarted'

#── 14) WARM-UP UBUNTU ---------------------------------------------------
Write-Log 'Verifying Ubuntu…'
try {
  $chk = wsl -d Ubuntu-22.04 --cd / -u root -- echo OK 2>$null
  Write-Log "  Ubuntu echo: $chk"
} catch {
  Write-Log "  Ubuntu start error: $_"
}

#── 15) AFTER metrics & COMPLETE -----------------------------------------
Log-Metrics 'AFTER'
Write-Log '===== Shrink-WSL run complete ====='
