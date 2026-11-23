# One command for the whole SQLForge stack on this Windows + WSL2 laptop:
#   powershell -ExecutionPolicy Bypass -File scripts\up.ps1            # FP8 model (default)
#   powershell -ExecutionPolicy Bypass -File scripts\up.ps1 -Model awq  # AWQ 4-bit model
# 1) Docker Desktop  2) secrets (first run)  3) vLLM in WSL2  4) API, UI, Langfuse, Prometheus, Grafana
param([ValidateSet("fp8", "awq")] [string]$Model = "fp8")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$DockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if (Test-Path $DockerBin) { $env:PATH = "$DockerBin;$env:PATH" }

function Wait-Http($Url, $Name, $Seconds) {
    for ($i = 0; $i -lt $Seconds; $i += 3) {
        try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 $Url | Out-Null; Write-Host "  $Name is up"; return } catch { Start-Sleep 3 }
    }
    throw "$Name did not come up at $Url within $Seconds s"
}

Write-Host "[1/4] Docker"
# probe through cmd: in Windows PowerShell 5.1 with ErrorActionPreference=Stop, a native command's
# stderr ("daemon not running") becomes a terminating error before $LASTEXITCODE can be checked
function Test-Docker { cmd /c "docker version --format {{.Server.Version}} >nul 2>&1"; return ($LASTEXITCODE -eq 0) }
function Move-StaleDockerSockets {
    # Docker Desktop deletes and recreates its AF_UNIX socket files on start. Since Windows updates
    # KB5121794/KB5127694 (Nov 2026) deleting or renaming those files fails ("The file cannot be
    # accessed by the system") and Docker quits with an error dialog. Renaming the folder that holds
    # them still works, so move such folders aside while Docker is stopped; Docker recreates them.
    if (Get-Process "com.docker.backend" -ErrorAction SilentlyContinue) { return }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    foreach ($dir in @("$env:LOCALAPPDATA\Docker\run", "$env:LOCALAPPDATA\docker-secrets-engine")) {
        if (-not (Test-Path -LiteralPath $dir)) { continue }
        $sockets = Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }
        if ($sockets) {
            Rename-Item -LiteralPath $dir -NewName ("{0}.stale-{1}" -f (Split-Path -Leaf $dir), $stamp)
            Write-Host "  moved stale sockets aside: $dir"
        }
    }
}
if (-not (Test-Docker)) {
    Move-StaleDockerSockets
    Write-Host "  starting Docker Desktop..."
    Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    $ok = $false
    for ($i = 0; $i -lt 60; $i++) { Start-Sleep 5; if (Test-Docker) { $ok = $true; break } }
    if (-not $ok) { throw "Docker engine did not start" }
}
Write-Host "  Docker is up"

Write-Host "[2/4] Secrets"
if (-not (Test-Path "observability\.env")) { & .venv\Scripts\python.exe scripts\init_observability.py }

Write-Host "[3/4] vLLM ($Model) in WSL2"
$up = $false
try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:8000/v1/models" | Out-Null; $up = $true } catch {}
if ($up) { Write-Host "  already running" } else {
    $wslRepo = (wsl -d Ubuntu -- wslpath -a ($Repo -replace '\\', '/')).Trim()
    if ($Model -eq "awq") { $serveArgs = @("models/sqlforge-3b-awq", "sqlforge") } else { $serveArgs = @("models/sqlforge-3b-v2-step1014", "sqlforge", "--quantization", "fp8") }
    $wslArgs = @("-d", "Ubuntu", "--cd", "`"$wslRepo`"", "--", "bash", "scripts/run_logged.sh", "outputs/vllm_up.log", "scripts/serve_vllm.sh") + $serveArgs
    Start-Process wsl -WindowStyle Hidden -ArgumentList $wslArgs
    Wait-Http "http://127.0.0.1:8000/v1/models" "vLLM" 300
}

Write-Host "[4/4] API, UI, Langfuse, Prometheus, Grafana"
$ErrorActionPreference = "Continue"  # compose prints progress on stderr
docker compose --env-file observability/.env up -d --build 2>&1 | ForEach-Object { "$_" }
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($code -ne 0) { throw "docker compose up failed" }
Wait-Http "http://127.0.0.1:8080/health" "API" 180
Wait-Http "http://127.0.0.1:8501/_stcore/health" "UI" 120
Wait-Http "http://127.0.0.1:3000/api/public/health" "Langfuse" 240

Write-Host ""
Write-Host "SQLForge is up:"
Write-Host "  UI          http://localhost:8501"
Write-Host "  API docs    http://localhost:8080/docs"
Write-Host "  Langfuse    http://localhost:3000   (login: observability\.env LANGFUSE_ADMIN_*)"
Write-Host "  Grafana     http://localhost:3001   (login: admin / observability\.env GRAFANA_ADMIN_PASSWORD)"
Write-Host "  Prometheus  http://localhost:9090"
Write-Host "Stop with: powershell -ExecutionPolicy Bypass -File scripts\down.ps1"
