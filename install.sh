#!/bin/sh
# Install llm-dash (llm-cost-dashboard) from the latest GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/Mattbusel/llm-cost-dashboard/master/install.sh | sh
#
# Options (environment variables):
#   LLM_DASH_VERSION       release tag to install, e.g. v1.2.1 (default: latest)
#   LLM_DASH_INSTALL_DIR   where to put the binary (default: ~/.local/bin)
set -eu

REPO="Mattbusel/llm-cost-dashboard"
BIN="llm-dash"
DIR="${LLM_DASH_INSTALL_DIR:-$HOME/.local/bin}"

say() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "this installer needs '$1'. Install it, or use: cargo install llm-cost-dashboard"; }

need curl
need tar

os="$(uname -s)"
arch="$(uname -m)"
ext="tar.gz"
exe=""
case "$os" in
  Linux) os_t="unknown-linux-gnu" ;;
  Darwin) os_t="apple-darwin" ;;
  MINGW* | MSYS* | CYGWIN*) os_t="pc-windows-msvc"; ext="zip"; exe=".exe" ;;
  *) die "no prebuilt binary for $os. Use: cargo install llm-cost-dashboard" ;;
esac
case "$arch" in
  x86_64 | amd64) arch_t="x86_64" ;;
  arm64 | aarch64)
    [ "$os" = "Darwin" ] || die "no prebuilt binary for $os $arch yet. Use: cargo install llm-cost-dashboard"
    arch_t="aarch64" ;;
  *) die "no prebuilt binary for $arch. Use: cargo install llm-cost-dashboard" ;;
esac
target="$arch_t-$os_t"

tag="${LLM_DASH_VERSION:-}"
if [ -z "$tag" ]; then
  # github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>
  tag="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest")"
  tag="${tag##*/}"
fi
[ -n "$tag" ] || die "could not work out the latest release tag"

name="llm-cost-dashboard-$tag-$target"
base="https://github.com/$REPO/releases/download/$tag"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

say "Downloading $name.$ext"
curl -fsSL "$base/$name.$ext" -o "$tmp/$name.$ext" || die "download failed: $base/$name.$ext"
curl -fsSL "$base/SHA256SUMS.txt" -o "$tmp/SHA256SUMS.txt" || die "could not download SHA256SUMS.txt"

want="$(grep " $name.$ext\$" "$tmp/SHA256SUMS.txt" | cut -d' ' -f1)"
[ -n "$want" ] || die "$name.$ext is not listed in SHA256SUMS.txt"
if command -v sha256sum >/dev/null 2>&1; then
  got="$(sha256sum "$tmp/$name.$ext" | cut -d' ' -f1)"
else
  got="$(shasum -a 256 "$tmp/$name.$ext" | cut -d' ' -f1)"
fi
[ "$want" = "$got" ] || die "checksum mismatch for $name.$ext (expected $want, got $got)"
say "Checksum OK"

if [ "$ext" = "zip" ]; then
  need unzip
  unzip -q "$tmp/$name.$ext" -d "$tmp"
else
  tar xzf "$tmp/$name.$ext" -C "$tmp"
fi

mkdir -p "$DIR"
cp "$tmp/$name/$BIN$exe" "$DIR/$BIN$exe"
chmod +x "$DIR/$BIN$exe"
say "Installed $("$DIR/$BIN$exe" --version) to $DIR/$BIN$exe"

case ":$PATH:" in
  *":$DIR:"*) ;;
  *) say ""
     say "$DIR is not on your PATH. Add this line to your shell profile:"
     say "  export PATH=\"$DIR:\$PATH\"" ;;
esac
say ""
say "Try it:  $BIN --demo"
