# --------------------------------------------
# shrink-wsl.ps1
# --------------------------------------------

# Ensure elevation
if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
  Write-Error "Please re-run this script as Administrator."
  exit 1
}

# Paths
$log      = 'G:\My Drive\Ingegneria\Data Science GD\My-Practice\scripts\shrink-wsl.log'
$vdGuid   = '00f6e855-3c9a-48fd-8eaa-f7cdfeb3d4fc'
$vdFolder = "C:\Users\alfri\AppData\Local\wsl\{$vdGuid}"
$vdPath   = "$vdFolder\ext4.vhdx"

# Logging helper:
function Log($msg) { "$(Get-Date -Format o)    $msg" | Out-File -Append $log }

# Start
Log "===== Shrink + Recovery run started ====="

# 1) Free-space before
$before = (Get-PSDrive C).Free / 1GB
Log ("Free space before: {0:N2} GB" -f $before)

# 2) Stop Docker Desktop processes
Log "Stopping Docker Desktop…"
Stop-Process -Name 'Docker Desktop','com.docker.backend' `
  -Force -ErrorAction SilentlyContinue
Log "Docker Desktop processes stopped (if any)."

# 3) Shutdown WSL and wait for vmmem to exit
Log "Shutting down WSL…"
wsl --shutdown 2>&1 | Out-File -Append $log

for ($i=0; $i -lt 15; $i++) {
  if (-not (Get-Process vmmem -ErrorAction SilentlyContinue)) {
    Log "vmmem exited."
    break
  }
  Log "vmmem still running, waiting 2s…"
  Start-Sleep -Seconds 2
}

# 4) Compact the Ubuntu ext4.vhdx via DiskPart
Log "Compacting VHDX: $vdPath"
$dp = @"
select vdisk file="$vdPath"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@
$dp | diskpart 2>&1 | Out-File -Append $log

# 5) Ensure any leftover attachment is removed
Log "Ensuring VHDX is fully detached…"
$dp2 = @"
select vdisk file="$vdPath"
detach vdisk
exit
"@
$dp2 | diskpart 2>&1 | Out-File -Append $log

# 6) Restart the WSL service (LxssManager)
Log "Restarting LxssManager service…"
Stop-Service LxssManager -Force -ErrorAction SilentlyContinue
Start-Service LxssManager -ErrorAction SilentlyContinue
Log "LxssManager restarted."

# 7) Warm up Ubuntu to verify it's back
Log "Starting WSL Ubuntu…"
try {
  wsl -d Ubuntu -u root -- echo "WSL is healthy" 2>&1 |
    ForEach-Object { Log "WSL output: $_" }
  Log "Ubuntu WSL started successfully."
} catch {
  Log "Error starting Ubuntu WSL: $_"
}

# 8) Free-space after
$after = (Get-PSDrive C).Free / 1GB
Log ("Free space after: {0:N2} GB" -f $after)
Log "===== Shrink + Recovery run ended =====`n"
