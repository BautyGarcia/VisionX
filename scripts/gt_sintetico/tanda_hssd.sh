#!/usr/bin/env bash
# HSSD (20/11, ~4:45): cámara sorteada (16:9) + ojo derecho ±2 mm, 20 vistas por escena, en las 125
# escenas train y 30 val de HSSD (las 13 de test no se renderizan). Una sola variante: cargar una
# escena de HSSD (navmesh + cientos de objetos) cuesta ~5 min, así que cada escena se carga una
# vez. Exposición automática en las vistas oscuras (misma curva en los dos ojos; vista.json).
# 6 flujos; copia al Drive cada 30 min y al final.
set -u
cd /d/VisionX
LOG=outputs/gtsynth/hssd.log
DATA=/mnt/d/VisionX/data/gt-sintetico-elp
VAR=hssd-camara-sorteada
PY=/d/VisionX/.venv/Scripts/python.exe
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
copiar() {
  MSYS_NO_PATHCONV=1 robocopy "D:\\VisionX\\data\\gt-sintetico-elp\\$VAR" \
    "H:\\Mi unidad\\VisionX-Personal\\gt-sintetico-elp\\$VAR" /E /MT:8 /NFL /NDL /NJH /NP | tail -3 >> "$LOG"
}

"$PY" - <<'PYEOF'
import json
d = json.load(open("D:/VisionX-wt/stand/src/visionx/gtsynth/hssd_particion.json"))
open("D:/VisionX/outputs/gtsynth/hssd-escenas.txt", "w").write("\n".join(d["train"] + d["val"]))
PYEOF
mapfile -t ESC < outputs/gtsynth/hssd-escenas.txt
log "##### HSSD: ${#ESC[@]} escenas, 6 flujos"
pids=()
for k in 0 1 2 3 4 5; do
  parte=$(printf "%s\n" "${ESC[@]}" | awk -v k=$k 'NR % 6 == k' | tr -d '\r' | tr '\n' ' ')
  (
    MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc "export MAMBA_ROOT_PREFIX=~/mamba; cd /tmp; \
      ~/bin/micromamba run -n habitat python /mnt/d/VisionX/scripts/gt_sintetico/render_tanda.py \
      --hssd /mnt/d/datasets/hssd-hab --escenas $parte --salida $DATA --variante $VAR \
      --perfil sorteada --proporciones 16:9 --vistas 20 --semilla $((30 + k)) --ojo-sorteado 0.002" \
      2>&1 | tr -d '\0' | grep --line-buffered -E "vistas$|Traceback|ESP_CHECK|Error:" \
      | grep --line-buffered -v -E "Metadata|Reloading" >> "$LOG"
  ) & pids+=($!)
done
( while true; do sleep 1800; log "== copia parcial"; copiar; done ) & copia=$!
wait "${pids[@]}"
kill $copia 2>/dev/null
log "== $VAR: verificar"
PYTHONIOENCODING=utf-8 PYTHONPATH=/d/VisionX-wt/stand/src "$PY" -m visionx.gtsynth verificar data/gt-sintetico-elp \
  --variante $VAR --max 40 2>&1 | tail -1 | tee -a "$LOG"
log "== copia final"; copiar
log "##### HSSD terminada: $(ls -d data/gt-sintetico-elp/$VAR/*/* 2>/dev/null | wc -l) vistas"
