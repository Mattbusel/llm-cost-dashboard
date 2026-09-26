# Install llm-dash (llm-cost-dashboard) on Windows from the latest GitHub release.
#
#   irm https://raw.githubusercontent.com/Mattbusel/llm-cost-dashboard/master/install.ps1 | iex
#
# Options (environment variables):
#   LLM_DASH_VERSION       release tag to install, e.g. v1.2.1 (default: latest)
#   LLM_DASH_INSTALL_DIR   install folder (default: %LOCALAPPDATA%\Programs\llm-dash)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = 'Mattbusel/llm-cost-dashboard'
$bin = 'llm-dash.exe'
$target = 'x86_64-pc-windows-msvc'
$dir = if ($env:LLM_DASH_INSTALL_DIR) { $env:LLM_DASH_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\llm-dash' }

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'llm-dash ships a 64-bit Windows binary only. On 32-bit Windows use: cargo install llm-cost-dashboard'
}

$tag = $env:LLM_DASH_VERSION
if (-not $tag) {
    $tag = (Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers @{ 'User-Agent' = 'llm-dash-installer' }).tag_name
}
if (-not $tag) { throw 'Could not work out the latest release tag.' }

$name = "llm-cost-dashboard-$tag-$target"
$base = "https://github.com/$repo/releases/download/$tag"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("llm-dash-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "Downloading $name.zip"
    Invoke-WebRequest "$base/$name.zip" -OutFile "$tmp\$name.zip" -UseBasicParsing
    Invoke-WebRequest "$base/SHA256SUMS.txt" -OutFile "$tmp\SHA256SUMS.txt" -UseBasicParsing

    $line = Get-Content "$tmp\SHA256SUMS.txt" | Where-Object { $_ -match "\s$([regex]::Escape("$name.zip"))$" } | Select-Object -First 1
    if (-not $line) { throw "$name.zip is not listed in SHA256SUMS.txt" }
    $want = ($line -split '\s+')[0].ToLower()
    $got = (Get-FileHash "$tmp\$name.zip" -Algorithm SHA256).Hash.ToLower()
    if ($want -ne $got) { throw "Checksum mismatch for $name.zip (expected $want, got $got)" }
    Write-Host 'Checksum OK'

    Expand-Archive "$tmp\$name.zip" -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Copy-Item "$tmp\$name\$bin" (Join-Path $dir $bin) -Force
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (($userPath -split ';') -contains $dir)) {
    $newPath = if ($userPath) { "$userPath;$dir" } else { $dir }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Host "Added $dir to your user PATH (open a new terminal to pick it up)."
}
$env:Path = "$env:Path;$dir"

$version = & (Join-Path $dir $bin) --version
Write-Host "Installed $version to $(Join-Path $dir $bin)"
Write-Host ''
Write-Host 'Try it:  llm-dash --demo'
