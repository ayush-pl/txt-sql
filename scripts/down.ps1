# Stop the whole SQLForge stack (containers keep their data volumes; add -Volumes to wipe them).
#   powershell -ExecutionPolicy Bypass -File scripts\down.ps1
param([switch]$Volumes)
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$DockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if (Test-Path $DockerBin) { $env:PATH = "$DockerBin;$env:PATH" }

Write-Host "Stopping containers"
if ($Volumes) { docker compose --env-file observability/.env down -v } else { docker compose --env-file observability/.env down }
Write-Host "Stopping vLLM in WSL2 (frees the GPU)"
wsl -d Ubuntu -- pkill -f "[v]llm serve"
Write-Host "Done."
