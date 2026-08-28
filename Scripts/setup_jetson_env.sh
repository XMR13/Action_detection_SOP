#!/usr/bin/env bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
env_dir=${1:-"$project_dir/.venv"}
python_bin=${PYTHON_BIN:-python3}

if [ -e "$env_dir" ]; then
    echo "Refusing to replace existing environment: $env_dir" >&2
    echo "Pass a new path or move the existing environment first." >&2
    exit 2
fi

python_version=$(
    "$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)
if [ "$python_version" != "3.10" ]; then
    echo "Python 3.10 is required; $python_bin reports $python_version." >&2
    exit 2
fi

echo "Creating Jetson environment with access to JetPack system packages:"
echo "  $env_dir"
"$python_bin" -m venv --system-site-packages "$env_dir"

# JetPack commonly installs trtexec outside PATH. Expose it only inside this
# virtual environment so Scripts/build_trt_engine.py can invoke it by name.
if ! PATH="$env_dir/bin:$PATH" command -v trtexec >/dev/null 2>&1; then
    for candidate in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
        if [ -x "$candidate" ]; then
            ln -s "$candidate" "$env_dir/bin/trtexec"
            break
        fi
    done
fi

echo "No pip packages were installed; JetPack remains the owner of GPU/CV packages."
echo "Running the repository's Jetson runtime checks..."
cd "$project_dir"
if ! PATH="$env_dir/bin:$PATH" "$env_dir/bin/python" -m Scripts.check_jetson_runtime; then
    echo >&2
    echo "The environment was kept at: $env_dir" >&2
    echo "After fixing the failed prerequisite, rerun the check with:" >&2
    echo "  source $env_dir/bin/activate" >&2
    echo "  python -m Scripts.check_jetson_runtime" >&2
    exit 1
fi

echo
echo "Environment ready. Activate it with:"
echo "  source $env_dir/bin/activate"
