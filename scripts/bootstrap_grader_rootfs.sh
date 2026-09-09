#!/usr/bin/env bash
set -Eeuo pipefail

# RunPod's container forbids the mount operations debootstrap needs. Assemble a
# mount-free, minimal chroot from the image's trusted Python runtime instead.
# Keep it on the ephemeral container disk and store only data in /workspace.
ROOTFS="${1:-/opt/aisi-grader-rootfs-minimal}"
BIN_DIR="${2:-/usr/local/bin}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MARKER="${ROOTFS}/.aisi-minimal-python-rootfs"

if [[ -e "${ROOTFS}" && ! -f "${MARKER}" ]]; then
    printf 'Refusing to reuse an unrecognised rootfs: %s\n' "${ROOTFS}" >&2
    exit 1
fi

apt-get update
apt-get install -y --no-install-recommends \
    gcc \
    libseccomp-dev \
    python3 \
    python3-pytest

if [[ ! -f "${MARKER}" ]]; then
    install -d -m 0755 \
        "${ROOTFS}/usr/bin" \
        "${ROOTFS}/usr/lib" \
        "${ROOTFS}/lib64" \
        "${ROOTFS}/tmp" \
        "${ROOTFS}/work"

    install -m 0755 /usr/bin/python3.12 "${ROOTFS}/usr/bin/python3.12"
    ln -s python3.12 "${ROOTFS}/usr/bin/python3"
    cp -a /usr/lib/python3.12 "${ROOTFS}/usr/lib/"
    cp -a /usr/lib/python3 "${ROOTFS}/usr/lib/"

    runtime_objects=(/usr/bin/python3.12)
    while IFS= read -r -d '' object; do
        runtime_objects+=("${object}")
    done < <(
        find /usr/lib/python3.12 /usr/lib/python3/dist-packages \
            -type f -name '*.so' -print0
    )

    for object in "${runtime_objects[@]}"; do
        while IFS= read -r dependency; do
            [[ -n "${dependency}" ]] || continue
            install -d -m 0755 "${ROOTFS}$(dirname -- "${dependency}")"
            cp -L --preserve=mode,timestamps \
                "${dependency}" "${ROOTFS}${dependency}"
        done < <(
            ldd "${object}" 2>/dev/null | awk \
                '/=> \// {print $3} /^[[:space:]]*\/.*\(/ {print $1}'
        )
    done

    touch "${MARKER}"
fi

install -d -m 0755 "${ROOTFS}/work"
chmod 0555 "${ROOTFS}/tmp"
install -d -m 0755 "${BIN_DIR}"
gcc -O2 -Wall -Wextra -Werror \
    "${PROJECT_DIR}/sandbox/grader_exec.c" \
    -lseccomp \
    -o "${BIN_DIR}/aisi-grader-exec"
chmod 0755 "${BIN_DIR}/aisi-grader-exec"

chroot "${ROOTFS}" /usr/bin/python3 -I -B --version
chroot "${ROOTFS}" /usr/bin/python3 -I -B -m pytest --capture=sys --version
printf 'Restricted grader ready: %s\n' "${BIN_DIR}/aisi-grader-exec"
