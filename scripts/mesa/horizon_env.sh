#!/bin/bash

ACTION_HORIZON="${ACTION_HORIZON:-1}"

if ! [[ "$ACTION_HORIZON" =~ ^[0-9]+$ ]] || [ "$ACTION_HORIZON" -lt 1 ]; then
    echo "ACTION_HORIZON must be a positive integer; got '$ACTION_HORIZON'." >&2
    return 2 2>/dev/null || exit 2
fi

mesa_data_path_for_horizon() {
    local base_path="$1"
    if [ "$ACTION_HORIZON" -eq 1 ]; then
        printf '%s' "$base_path"
    else
        printf '%s/h%s' "$base_path" "$ACTION_HORIZON"
    fi
}

mesa_run_suffix_for_horizon() {
    if [ "$ACTION_HORIZON" -eq 1 ]; then
        printf ''
    else
        printf -- '-T%s' "$ACTION_HORIZON"
    fi
}
