#!/usr/bin/env bash
set -euo pipefail
{
  echo "==== uname -a ===="
  uname -a || true
  echo
  echo "==== lscpu ===="
  lscpu || true
  echo
  echo "==== free -h ===="
  free -h || true
  echo
  echo "==== lsblk ===="
  lsblk || true
}