#!/usr/bin/env bash
# Discover one logical CPU ID per preferred P-core for RAPID --core-list / CORES.
# Prefer P-cores (core_type=0 or highest MAXMHZ); emit comma-separated logical IDs.
set -euo pipefail

declare -A CORE_TO_CPU=()
declare -A CORE_MHZ=()
declare -A CORE_TYPE=()

if [[ -r /proc/cpuinfo ]]; then
  cpu="" core="" mhz=""
  while IFS= read -r line; do
    case "$line" in
      processor*) cpu="${line##*: }" ;;
      "core id"*) core="${line##*: }" ;;
      "cpu MHz"*) mhz="${line##*: }" ;;
      "")
        if [[ -n "${cpu:-}" && -n "${core:-}" ]]; then
          # Prefer the first (usually sibling 0) logical CPU for each physical core.
          if [[ -z "${CORE_TO_CPU[$core]+x}" ]]; then
            CORE_TO_CPU[$core]="$cpu"
            CORE_MHZ[$core]="${mhz:-0}"
          fi
        fi
        cpu=""; core=""; mhz=""
        ;;
    esac
  done < /proc/cpuinfo
fi

for f in /sys/devices/system/cpu/cpu[0-9]*/topology/core_type; do
  [[ -e "$f" ]] || continue
  cpu_id="${f#/sys/devices/system/cpu/cpu}"
  cpu_id="${cpu_id%%/*}"
  ctype="$(cat "$f" 2>/dev/null || echo missing)"
  # Map back via core id for this logical CPU.
  core_id="$(cat "/sys/devices/system/cpu/cpu${cpu_id}/topology/core_id" 2>/dev/null || true)"
  [[ -n "$core_id" ]] || continue
  CORE_TYPE[$core_id]="$ctype"
done

# Prefer cores with core_type=0 (Intel P-core) when available.
p_cores=()
e_cores=()
unknown=()
for core in "${!CORE_TO_CPU[@]}"; do
  ctype="${CORE_TYPE[$core]:-missing}"
  if [[ "$ctype" == "0" || "$ctype" == "performance" ]]; then
    p_cores+=("$core")
  elif [[ "$ctype" == "1" || "$ctype" == "efficient" ]]; then
    e_cores+=("$core")
  else
    unknown+=("$core")
  fi
done

selected=()
if ((${#p_cores[@]} > 0)); then
  # Sort P-cores numerically and take their representative logical CPUs.
  mapfile -t p_sorted < <(printf '%s\n' "${p_cores[@]}" | sort -n)
  for core in "${p_sorted[@]}"; do
    selected+=("${CORE_TO_CPU[$core]}")
  done
else
  # Fallback: unique physical cores ordered by descending MAXMHZ / cpu MHz.
  mapfile -t by_mhz < <(
    for core in "${!CORE_TO_CPU[@]}"; do
      mhz_file="/sys/devices/system/cpu/cpu${CORE_TO_CPU[$core]}/cpufreq/cpuinfo_max_freq"
      if [[ -r "$mhz_file" ]]; then
        mhz="$(cat "$mhz_file")"
      else
        mhz="$(printf '%.0f' "${CORE_MHZ[$core]:-0}")"
      fi
      printf '%s %s\n' "$mhz" "$core"
    done | sort -nr -k1,1
  )
  declare -A seen_core=()
  for row in "${by_mhz[@]}"; do
    core="${row##* }"
    [[ -z "${seen_core[$core]+x}" ]] || continue
    seen_core[$core]=1
    selected+=("${CORE_TO_CPU[$core]}")
  done
fi

if ((${#selected[@]} < 5)); then
  echo "ERROR: need >=5 isolated logical CPUs; found ${#selected[@]}: ${selected[*]:-none}" >&2
  exit 1
fi

# Emit ascending logical IDs so core_slice(n) always takes the same first-n set.
mapfile -t selected < <(printf '%s\n' "${selected[@]}" | sort -n)
(IFS=,; echo "${selected[*]}")
