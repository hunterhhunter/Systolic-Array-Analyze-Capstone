param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Cmd
)

if ($Cmd.Count -eq 0) {
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 make env"
    Write-Host "  .\run.ps1 make test"
    Write-Host "  .\run.ps1 make boundary-sweep"
    Write-Host "  .\run.ps1 python -m tools.boundary_sweep --dry-run --mnk 32x32x64 --array 8x8"
    exit 1
}

# The docker-compose dev service stores its virtualenv in the named /opt/venv volume.
# Give a clear first-run message instead of failing later with a missing python path.
if ($Cmd.Count -ge 2 -and $Cmd[0] -eq "make" -and $Cmd[1] -ne "env") {
    docker compose run --rm dev bash -lc "test -x /opt/venv/bin/python || { echo 'Python environment is missing. Run: .\run.ps1 make env'; exit 2; }"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

docker compose run --rm dev @Cmd
