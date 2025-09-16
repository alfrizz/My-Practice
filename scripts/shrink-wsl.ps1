# -------------------------------------------
# shrink-wsl.ps1
# --------------------------------------------

# Remap G: so elevated session can write logs
net use G: \\localhost\g$ /persistent:no 2>$null

# Paths & Distro
$log    = 'G:\My Drive\Ingegneria\Data Science GD\My-Practice\scripts\shrink-wsl.log'
$distro = 'Ubuntu-22.04'
$vdPath = 'C:\WSL\Staging\old-ext4.vhdx'    # ← standalone VHDX

# UTF-8 logging helper
function Log($msg) {
  $entry = "$(Get-Date -Format o)    $msg"
  Write-Host $entry
  $entry | Out-File -FilePath $log -Append -Encoding utf8 -Force
}

# Elevation check
if (-not (
    [Security.Principal.WindowsPrincipal] `
      [Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole(
    [Security.Principal.WindowsBuiltInRole] "Administrator"
  )) {
  Write-Host "ERROR: Please re-run via an elevated PowerShell prompt." -ForegroundColor Red
  exit 1
}

# Start fresh log
Remove-Item $log -ErrorAction SilentlyContinue
Log "===== Shrink + Recovery run started ====="

# 1) Free space before
$before = (Get-PSDrive C).Free / 1GB
Log ("Free space before: {0:N2} GB" -f $before)

# 2) Stop Docker Desktop
Log "Stopping Docker Desktop…"
Stop-Process -Name 'Docker Desktop','com.docker.backend' -Force -ErrorAction SilentlyContinue
Log "Docker Desktop stopped."

# 3) Mount the VHDX in-place (no copy) so we can purge and fstrim
Log "Mounting VHDX via WSL…"
wsl --mount $vdPath --partition 1 --type ext4 2>&1 |
  ForEach-Object { Log "WSL mount: $_" }

# 4) Purge /tmp inside that mount, then fstrim it
Log "Purging /tmp in mounted image and running fstrim…"
wsl -d $distro -u root -- bash -lc @'
set -euo pipefail
echo "Deleting /mnt/wsl/tmp/*…"
rm -rf /mnt/wsl/tmp/*
echo "Running fstrim on /mnt/wsl…"
fstrim -v /mnt/wsl
'@ 2>&1 |
  ForEach-Object { Log "WSL purge+fstrim: $_" }

# 5) Unmount the VHDX so Windows can compact it
Log "Unmounting VHDX…"
wsl --unmount $vdPath 2>&1 |
  ForEach-Object { Log "WSL unmount: $_" }

# 6) Shutdown WSL so DiskPart can access the idle VHDX
Log "Shutting down WSL…"
wsl --shutdown 2>&1 |
  ForEach-Object { Log "WSL: $_" }
for ($i = 0; $i -lt 15; $i++) {
  if (-not (Get-Process vmmem -ErrorAction SilentlyContinue)) {
    Log "vmmem exited."
    break
  }
  Start-Sleep -Seconds 2
  Log "Waiting for vmmem to exit…"
}

# 7) DiskPart compact (select before attach)
$dp = @"
select vdisk file=""$vdPath""
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
Log "Running DiskPart compact on VHDX…"
$dp |
  diskpart 2>&1 |
  Where-Object { $_ -and ($_ -notmatch 'percent completed') } |
  ForEach-Object { Log "DiskPart: $_" }

# 8) Restart WSL service
Log "Restarting LxssManager…"
Stop-Service LxssManager -Force -ErrorAction SilentlyContinue
Start-Service LxssManager -ErrorAction SilentlyContinue
Log "LxssManager restarted."

# 9) Health-check
Log "Health-check: starting WSL $distro…"
try {
  wsl -d $distro -u root -- echo "WSL is healthy" 2>&1 |
    ForEach-Object { Log "WSL: $_" }
  Log "$distro started successfully."
} catch {
  Log "Health-check error: $_"
}

# 10) Free space after
$after = (Get-PSDrive C).Free / 1GB
Log ("Free space after:  {0:N2} GB" -f $after)
Log "===== Shrink + Recovery run ended =====`n"

# Cleanup mapping
net use G: /delete 2>$null
