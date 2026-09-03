#!/usr/bin/env bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
default_project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

component="all"
project_dir="$default_project_dir"
python_bin=""
data_dir=""
service_user="${SUDO_USER:-$(id -un)}"
enable_services=false
start_services=false

usage() {
    cat <<'EOF'
Install the Action Detection SOP systemd service units.

Usage:
  sudo bash Scripts/install_systemd_services.sh [options]

Options:
  --component all|rtsp|web  Install both units or only one (default: all).
  --project-dir PATH        Absolute repository path (default: detected).
  --python-bin PATH         Python inside the prepared environment
                            (default: <project-dir>/.venv/bin/python).
  --data-dir PATH           Shared runner/web data root
                            (default: <project-dir>/data/roll_sop_v1).
  --service-user USER       Linux account that owns the processes
                            (default: SUDO_USER/current user).
  --enable                  Enable installed units for boot after validation.
  --start                   Enable and start/restart units after validation.
  -h, --help                Show this help.

The installer preserves existing /etc/action-sop/*.env files. It never puts
RTSP or website passwords directly in a unit file.
EOF
}

need_value() {
    if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --component)
            need_value "$@"
            component=$2
            shift 2
            ;;
        --project-dir)
            need_value "$@"
            project_dir=$2
            shift 2
            ;;
        --python-bin)
            need_value "$@"
            python_bin=$2
            shift 2
            ;;
        --data-dir)
            need_value "$@"
            data_dir=$2
            shift 2
            ;;
        --service-user)
            need_value "$@"
            service_user=$2
            shift 2
            ;;
        --start)
            enable_services=true
            start_services=true
            shift
            ;;
        --enable)
            enable_services=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$component" in
    all|rtsp|web) ;;
    *)
        echo "--component must be all, rtsp, or web" >&2
        exit 2
        ;;
esac

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this installer with sudo; systemd units are written under /etc." >&2
    exit 2
fi

if ! id "$service_user" >/dev/null 2>&1; then
    echo "Service user does not exist: $service_user" >&2
    exit 2
fi

project_dir=$(readlink -f "$project_dir")
if [ ! -d "$project_dir" ]; then
    echo "Project directory does not exist: $project_dir" >&2
    exit 2
fi

if [ -z "$python_bin" ]; then
    python_bin="$project_dir/.venv/bin/python"
fi
# Preserve the virtual-environment executable path. Dereferencing the final
# `python` symlink would turn `.venv/bin/python` into `/usr/bin/python3.10` and
# make systemd bypass packages installed in the virtual environment.
python_bin=$(realpath -s "$python_bin")
if [ ! -x "$python_bin" ]; then
    echo "Python executable is missing or not executable: $python_bin" >&2
    exit 2
fi

if [ -z "$data_dir" ]; then
    data_dir="$project_dir/data/roll_sop_v1"
fi
mkdir -p "$data_dir"
data_dir=$(readlink -f "$data_dir")

service_group=$(id -gn "$service_user")
template_dir="$project_dir/deploy/systemd"
env_dir="/etc/action-sop"
unit_dir="/etc/systemd/system"

if [ ! -d "$template_dir" ]; then
    echo "Systemd template directory is missing: $template_dir" >&2
    exit 2
fi

install -d -o root -g "$service_group" -m 0750 "$env_dir"
chown "$service_user:$service_group" "$data_dir"
chmod 0750 "$data_dir"

escape_sed_replacement() {
    printf '%s' "$1" | sed 's/[\\&|]/\\&/g'
}

render_unit() {
    template=$1
    destination=$2
    escaped_project=$(escape_sed_replacement "$project_dir")
    escaped_python=$(escape_sed_replacement "$python_bin")
    escaped_user=$(escape_sed_replacement "$service_user")
    escaped_group=$(escape_sed_replacement "$service_group")
    tmp_file=$(mktemp)
    sed \
        -e "s|@PROJECT_DIR@|$escaped_project|g" \
        -e "s|@PYTHON_BIN@|$escaped_python|g" \
        -e "s|@SERVICE_USER@|$escaped_user|g" \
        -e "s|@SERVICE_GROUP@|$escaped_group|g" \
        "$template" > "$tmp_file"
    install -o root -g root -m 0644 "$tmp_file" "$destination"
    rm -f "$tmp_file"
}

install_env_if_missing() {
    template=$1
    destination=$2
    if [ -e "$destination" ]; then
        echo "Preserved existing environment file: $destination"
        return
    fi
    escaped_data=$(escape_sed_replacement "$data_dir")
    tmp_file=$(mktemp)
    sed -e "s|@DATA_DIR@|$escaped_data|g" "$template" > "$tmp_file"
    install -o root -g "$service_group" -m 0640 "$tmp_file" "$destination"
    rm -f "$tmp_file"
    echo "Created environment file: $destination"
}

install_env_if_missing "$template_dir/common.env.example" "$env_dir/common.env"

units=""
if [ "$component" = "all" ] || [ "$component" = "rtsp" ]; then
    render_unit \
        "$template_dir/action-sop-rtsp.service.in" \
        "$unit_dir/action-sop-rtsp.service"
    install_env_if_missing "$template_dir/rtsp.env.example" "$env_dir/rtsp.env"
    units="$units action-sop-rtsp.service"
fi

if [ "$component" = "all" ] || [ "$component" = "web" ]; then
    render_unit \
        "$template_dir/action-sop-web.service.in" \
        "$unit_dir/action-sop-web.service"
    install_env_if_missing "$template_dir/web.env.example" "$env_dir/web.env"
    units="$units action-sop-web.service"
fi

systemctl daemon-reload

if [ "$enable_services" = true ]; then
    if [ "$component" = "all" ] || [ "$component" = "rtsp" ]; then
        if grep -q 'CAMERA_IP\|STREAM_PATH' "$env_dir/rtsp.env"; then
            echo "Edit $env_dir/rtsp.env before using --enable or --start." >&2
            exit 2
        fi
    fi
    if [ "$component" = "all" ] || [ "$component" = "web" ]; then
        if grep -q 'SOP_ADMIN_PASSWORD="CHANGE_ME"' "$env_dir/web.env"; then
            echo "Edit $env_dir/web.env before using --enable or --start." >&2
            exit 2
        fi
    fi
    for unit in $units; do
        systemctl enable "$unit"
    done
fi

if [ "$start_services" = true ]; then
    if [ "$component" = "all" ] || [ "$component" = "web" ]; then
        systemctl restart action-sop-web.service
    fi
    if [ "$component" = "all" ] || [ "$component" = "rtsp" ]; then
        systemctl restart action-sop-rtsp.service
    fi
fi

echo
echo "Installed:$units"
echo "Configuration directory: $env_dir"
echo "Shared data directory: $data_dir"
if [ "$enable_services" = false ]; then
    echo "Edit the environment files, then enable and start with:"
    echo "  sudo bash Scripts/install_systemd_services.sh --component $component --service-user $service_user --start"
elif [ "$start_services" = false ]; then
    echo "Enabled for boot but not started. Start with:"
    echo "  sudo systemctl start$units"
fi
echo "Follow logs with:"
echo "  sudo journalctl -u action-sop-rtsp.service -u action-sop-web.service -f"
