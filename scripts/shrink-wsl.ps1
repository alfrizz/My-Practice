<#
.SYNOPSIS
  One-click, fully-logged WSL shrink + cleanup.

.DESCRIPTION
  Deletes only:
    • /tmp entries older than 3h  
    • .ipynb_checkpoints dirs under ~/ 
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

function global:Log-Metrics($label) {
  $free = (Get-PSDrive C).Free / 1GB
  $size = (Get-Item -LiteralPath $vhdx).Length / 1GB
  Write-Log ("{0}: C free={1:N2} GB; ext4.vhdx={2:N2} GB" -f $label, $free, $size)
}

#── 3) PATCH wsl.conf (remove invalid networkDrives) ------------------------
Write-Log 'Patching /etc/wsl.conf to drop invalid networkDrives key…'
wsl -d Ubuntu-22.04 --cd / -u root -- bash -c "sed -i '/^\s*networkDrives/d' /etc/wsl.conf" 2>$null

#── 4) LOCATE VHDX & LOG BEFORE ---------------------------------------------
# Detect distro base path from WSL registry so we work for installs on other drives
$distroName = 'Ubuntu-22.04'
$reg = Get-ItemProperty HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss\* |
       Where-Object { $_.DistributionName -eq $distroName }

if ($null -eq $reg -or -not $reg.BasePath) {
  Write-Log "ERROR: Could not detect $distroName BasePath in registry"
  throw 'Missing VHDX base path'
}

# Trim the \\?\ prefix if present and build ext4.vhdx path
$basePath = $reg.BasePath.TrimStart('\\?\')
$vhdx = Join-Path $basePath 'ext4.vhdx'
Write-Log "VHDX path: $vhdx"

if (-not (Test-Path -LiteralPath $vhdx)) {
  Write-Log "ERROR: ext4.vhdx not found at detected path: $vhdx"
  throw 'Missing VHDX'
}


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
  'find /home/alfrizz -type d -name ".ipynb_checkpoints" -prune -print' 2>$null
foreach ($p in $cpList) { Write-Log "  [DEL] chkpt$p" }

Write-Log 'Deleting .ipynb_checkpoints…'
wsl -d Ubuntu-22.04 --cd / -u root -- bash -c `
  'find /home/alfrizz -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} + || true' 2>$null
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

#── 12) ROBUST: ensure uncompressed/sparse/unencrypted then DiskPart -----------
Write-Log "Preparing VHDX for compaction: $vhdx"

# Minimum free space required on host drive (GB)
$minFreeGB = 20

# Determine host drive letter
$driveRoot = [IO.Path]::GetPathRoot($vhdx)
$driveLetter = if ($driveRoot) { $driveRoot.Substring(0,1) } else { Write-Log 'ERROR: Cannot determine drive root'; throw 'Cannot determine drive root' }

# Quick free-space check
try {
  $vol = Get-Volume -DriveLetter $driveLetter -ErrorAction Stop
  if (($vol.SizeRemaining / 1GB) -lt $minFreeGB) {
    Write-Log ("ERROR: Not enough free space on {0}: need {1} GB" -f $driveLetter, $minFreeGB)
    throw 'Insufficient free space'
  }
  Write-Log ("Host drive {0} free space OK: {1:N2} GB" -f $driveLetter, $vol.SizeRemaining/1GB)
} catch {
  Write-Log "ERROR: Free-space check failed: $_"
  throw 'Free-space check failed'
}

# Read attributes
$itm = Get-Item -LiteralPath $vhdx -ErrorAction Stop
Write-Log ("  Initial Attributes: {0}" -f $itm.Attributes)

# If compressed (possibly inherited from folder), uncompress the folder and file
if ($itm.Attributes -band [IO.FileAttributes]::Compressed) {
  Write-Log 'VHDX is NTFS-compressed — uncompressing file and parent folder (compact /U /S)…'
  try {
    $folder = Split-Path -Parent $vhdx
    & compact /U /S:"$folder" 2>&1 | ForEach-Object { Write-Log "  compact: $_" }
    Start-Sleep -Seconds 2
    $itm = Get-Item -LiteralPath $vhdx -ErrorAction Stop
    if ($itm.Attributes -band [IO.FileAttributes]::Compressed) {
      Write-Log 'ERROR: ext4.vhdx remains compressed after compact /U; aborting.'
      throw 'VHDX still compressed'
    }
    Write-Log 'Uncompress succeeded'
  } catch {
    Write-Log "ERROR: compact/uncompress failed: $_"
    throw 'compact failed'
  }
} else {
  Write-Log 'VHDX not compressed'
}

# Check and clear sparse flag if set
try {
  $sparseOut = fsutil sparse queryflag $vhdx 2>&1
  if ($sparseOut -match 'This file is set as sparse') {
    Write-Log 'VHDX is sparse — clearing sparse flag (fsutil sparse setflag 0)…'
    fsutil sparse setflag $vhdx 0 2>&1 | ForEach-Object { Write-Log "  fsutil: $_" }
    Start-Sleep -Seconds 1
    $sparseOut2 = fsutil sparse queryflag $vhdx 2>&1
    if ($sparseOut2 -match 'This file is set as sparse') {
      Write-Log 'ERROR: Failed to clear sparse flag; aborting compaction.'
      throw 'VHDX sparse flag still set'
    }
    Write-Log 'Sparse flag cleared'
  } else {
    Write-Log 'Sparse flag: not set'
  }
} catch {
  Write-Log "fsutil sparse check/set failed: $_"
  throw 'fsutil error'
}

# If encrypted (EFS), attempt to decrypt (cipher /d)
$itm = Get-Item -LiteralPath $vhdx -ErrorAction Stop
if ($itm.Attributes -band [IO.FileAttributes]::Encrypted) {
  Write-Log 'VHDX is EFS-encrypted — attempting to decrypt (cipher /d)…'
  try {
    & cipher /d $vhdx 2>&1 | ForEach-Object { Write-Log "  cipher: $_" }
    Start-Sleep -Seconds 1
    $itm = Get-Item -LiteralPath $vhdx -ErrorAction Stop
    if ($itm.Attributes -band [IO.FileAttributes]::Encrypted) {
      Write-Log 'ERROR: ext4.vhdx remains encrypted after cipher /d; aborting compaction.'
      throw 'VHDX still encrypted'
    }
    Write-Log 'Decryption succeeded'
  } catch {
    Write-Log "ERROR: cipher decrypt failed: $_"
    throw 'cipher failed'
  }
} else {
  Write-Log 'EFS encryption: not set'
}

# Final verify before DiskPart
$itm = Get-Item -LiteralPath $vhdx -ErrorAction Stop
Write-Log ("Final attributes: {0}" -f $itm.Attributes)
if ($itm.Attributes -band [IO.FileAttributes]::Compressed -or $itm.Attributes -band [IO.FileAttributes]::Encrypted) {
  Write-Log 'ERROR: VHDX still has attributes preventing DiskPart compaction; aborting.'
  throw 'VHDX attributes blocking compaction'
}

# Run DiskPart compaction
Write-Log "Compacting VHDX with DiskPart: $vhdx"
$tmpDp = [IO.Path]::GetTempFileName()
@"
select vdisk file="$vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@ | Set-Content -LiteralPath $tmpDp -Encoding ASCII

try {
  $dpLines = diskpart /s $tmpDp | Select-Object -Unique
  foreach ($l in $dpLines) { Write-Log "  DiskPart: $l" }
} catch {
  Write-Log "ERROR: DiskPart failed: $_"
  Remove-Item $tmpDp -Force -ErrorAction SilentlyContinue
  throw 'DiskPart error'
}
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
