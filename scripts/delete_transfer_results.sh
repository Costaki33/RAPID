#!/usr/bin/env bash
set -euo pipefail
cd /home/cgs2528/RAPID
rm -rf results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19
rm -rf results/locked_recipe_transfer/BEGE-TEXA75535L_smoke
rm -rf /mnt/c/Users/cgs2528/Projects/RAPID/results/locked_recipe_transfer 2>/dev/null || true
echo "remaining:"
ls -la results/locked_recipe_transfer/ 2>/dev/null || echo "dir empty or gone"
echo DONE
