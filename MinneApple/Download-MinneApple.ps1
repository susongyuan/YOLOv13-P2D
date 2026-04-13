# MinneApple 官方文件直链（DSpace REST）。支持断点续传（curl -C -）。
# 无高校/作者提供的国内镜像；若本机已开 Clash/V2 等，可先设置代理再运行，例如：
#   $env:HTTPS_PROXY = "http://127.0.0.1:7890"

$ErrorActionPreference = "Stop"
$OutDir = $PSScriptRoot
$Base = "https://conservancy.umn.edu/server/api/core/bitstreams"

$files = @(
    @{ Name = "MinneApple_Data_README.txt"; Uuid = "33213e11-0a73-47f3-b357-f6905fa33993" },
    @{ Name = "detection.tar.gz";           Uuid = "3ef26f04-6467-469b-9857-f443ffa1bb61" },
    @{ Name = "counting.tar.gz";            Uuid = "9a8d9477-4549-4896-974b-e824c5ab2d19" },
    @{ Name = "test_data.zip";              Uuid = "7856047c-1f23-43ef-a50d-9327ec2ae0ff" }
)

foreach ($f in $files) {
    $url = "$Base/$($f.Uuid)/content"
    $dest = Join-Path $OutDir $f.Name
    Write-Host "Downloading $($f.Name) ..."
    & curl.exe -fL --retry 10 --retry-delay 10 --connect-timeout 60 --max-time 0 -C - -o $dest $url
    if ($LASTEXITCODE -ne 0) { throw "curl failed for $($f.Name) exit=$LASTEXITCODE" }
}

Write-Host "Done. Extract with: tar -xzf detection.tar.gz && tar -xzf counting.tar.gz (or use 7-Zip for zip)."
