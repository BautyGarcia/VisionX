#!/usr/bin/env bash
# Recorridos con cámara sorteada para el plug and play (PLUG-AND-PLAY.md, sección 7.2).
# Un recorrido de 30 cuadros por escena de HSSD (125 train + 30 val; test no), cada uno con su
# propia cámara (campo 55–130°, separación 3–20 cm, 16:9 o 4:3), a 960 px de ancho (la red de
# entrada trabaja a 640). Lienzo con margen 1,25 para aplicar lentes al entrenar.
# 6 flujos; al final copia la variante al Drive (VisionX-Personal/gt-sintetico-elp).
set -u
cd /d/VisionX
LOG=outputs/gtsynth/recorridos.log
DATA=/mnt/d/VisionX/data/gt-sintetico-elp
VAR=hssd-recorridos-sorteada
PY=/d/VisionX/.venv/Scripts/python.exe
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

"$PY" - <<'PYEOF'
import json
d = json.load(open("D:/VisionX-wt/stand/src/visionx/gtsynth/hssd_particion.json"))
open("D:/VisionX/outputs/gtsynth/hssd-escenas.txt", "w").write("\n".join(d["train"] + d["val"]))
PYEOF
mapfile -t ESC < outputs/gtsynth/hssd-escenas.txt
log "##### recorridos sorteados: ${#ESC[@]} escenas, 6 flujos"
pids=()
for k in 0 1 2 3 4 5; do
  parte=$(printf "%s\n" "${ESC[@]}" | awk -v k=$k 'NR % 6 == k' | tr -d '\r' | tr '\n' ' ')
  (
    MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc "export MAMBA_ROOT_PREFIX=~/mamba; cd /tmp; \
      ~/bin/micromamba run -n habitat python /mnt/d/VisionX-wt/stand/scripts/gt_sintetico/render_tanda.py \
      --hssd /mnt/d/datasets/hssd-hab --escenas $parte --salida $DATA --variante $VAR \
      --perfil sorteada --ancho-sorteada 960 --proporciones 16:9 4:3 --tipo-pose recorrido \
      --vistas 30 --semilla $((40 + k))" \
      2>&1 | tr -d '\0' | grep --line-buffered -E "vistas$|Traceback|ESP_CHECK|Error:" \
      | grep --line-buffered -v -E "Metadata|Reloading" >> "$LOG"
  ) & pids+=($!)
done
wait "${pids[@]}"
log "render terminado: $(find /d/VisionX/data/gt-sintetico-elp/$VAR -name vista.json | wc -l) vistas, $(du -sh /d/VisionX/data/gt-sintetico-elp/$VAR | cut -f1)"
MSYS_NO_PATHCONV=1 robocopy "D:\VisionX\data\gt-sintetico-elp\$VAR" \
  "H:\Mi unidad\VisionX-Personal\gt-sintetico-elp\$VAR" /E /MT:8 /NFL /NDL /NJH /NP | tail -4 >> "$LOG"
log "copia al Drive terminada"
