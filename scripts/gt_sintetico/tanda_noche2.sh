#!/usr/bin/env bash
# Cola nocturna del 20/11 (pedido de Tomás: muchos datos para reentrenar la red de alineación v3,
# con variantes de cámara, lentes y ojo derecho corrido; de Replica y de HSSD).
#
#   A. Replica (arranca ya):
#      secuencias (rehechas: ya no se meten en las paredes), camara-sorteada en val,
#      camara-sorteada-ojo y elp-margen-ojo en train (ojo derecho corrido ±2 mm por vista, con
#      paralaje real), camara-sorteada-ojo en val.
#   B. HSSD (cuando termina la descarga y una escena de prueba renderiza bien):
#      hssd-elp-margen (ELP a media resolución, 12 por escena) y hssd-camara-sorteada (cámara
#      sorteada + ojo ±2 mm, 16 por escena) en las 125 escenas train y 30 de val. Las otras 13 de
#      val de HSSD quedan como test (no se renderizan).
# Cada variante se verifica (20 vistas, `gtsynth verificar`) y se copia al Drive al terminar.
set -u
cd /d/VisionX
LOG=outputs/gtsynth/noche2.log
DATA=/mnt/d/VisionX/data/gt-sintetico-elp
HSSD=/mnt/d/datasets/hssd-hab
VAL="frl_apartment_3 office_2 room_1 hotel_0"
TRAIN_A="apartment_0 apartment_1 frl_apartment_0 frl_apartment_1"
TRAIN_B="frl_apartment_2 office_0 office_1 room_0"
PY=/d/VisionX/.venv/Scripts/python.exe

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

render() {  # render <variante> <fuente: replica|hssd> <escenas> <args...>
  local var=$1 fuente=$2 esc=$3; shift 3
  local src="--replica /mnt/d/datasets/replica"
  [ "$fuente" = hssd ] && src="--hssd $HSSD"
  log "== $var ($fuente): render ($(echo $esc | wc -w) escenas) $*"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc "export MAMBA_ROOT_PREFIX=~/mamba; cd /tmp; \
    ~/bin/micromamba run -n habitat python /mnt/d/VisionX/scripts/gt_sintetico/render_tanda.py \
    $src --escenas $esc --salida $DATA --variante $var $*" 2>&1 \
    | tr -d '\0' | grep -E "vistas$|Traceback|Error:|ESP_CHECK" | grep -v Reloading | tee -a "$LOG"
}

verificar() {
  log "== $1: verificar"
  PYTHONIOENCODING=utf-8 "$PY" -m visionx.gtsynth verificar data/gt-sintetico-elp --variante "$1" --max 20 \
    2>&1 | tail -1 | tee -a "$LOG"
}

copiar() {
  local dst; dst=$(head -1 outputs/gtsynth/destino_drive.txt | tr -d '\r')
  log "== $1: copia a $dst"
  MSYS_NO_PATHCONV=1 robocopy "D:\\VisionX\\data\\gt-sintetico-elp\\$1" "$dst\\$1" /E /MT:8 /NFL /NDL /NJH /NP \
    | tail -3 | tee -a "$LOG"
}

listo() { verificar "$1"; copiar "$1"; }

# ---------------------------------------------------------------------------------- A. Replica
log "##### noche 2: empieza (A: Replica)"
rm -rf data/gt-sintetico-elp/secuencias "H:/Mi unidad/VisionX-Personal/gt-sintetico-elp/secuencias"
(
  render secuencias replica "$VAL" --perfil elp --tipo-pose secuencia --vistas 80 --semilla 4
  listo secuencias
  render camara-sorteada replica "$VAL" --perfil sorteada --vistas 60 --semilla 5
  listo camara-sorteada
  render camara-sorteada-ojo replica "$VAL" --perfil sorteada --vistas 40 --semilla 6 --ojo-sorteado 0.002
) & a1=$!
render camara-sorteada-ojo replica "$TRAIN_A" --perfil sorteada --vistas 300 --semilla 7 --ojo-sorteado 0.002 & a2=$!
render camara-sorteada-ojo replica "$TRAIN_B" --perfil sorteada --vistas 300 --semilla 17 --ojo-sorteado 0.002 & a3=$!
render elp-margen-ojo replica "$TRAIN_A $TRAIN_B" --perfil elp-media --vistas 150 --semilla 8 --ojo-sorteado 0.002 & a4=$!

# ------------------------------------------------------------------------------------- B. HSSD
(
  log "== HSSD: esperando que termine la descarga"
  while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*snapshot_download*hssd*' }) { exit 0 } else { exit 1 }"; do
    sleep 60
  done
  log "== HSSD: descarga terminada ($(find /d/datasets/hssd-hab/objects -type f | wc -l) archivos de objetos)"
  "$PY" - <<'PYEOF'
import yaml
s = yaml.safe_load(open("D:/datasets/hssd-hab/scene_splits.yaml"))
val = sorted(s["val"])
open("D:/VisionX/outputs/gtsynth/hssd-train.txt", "w").write(" ".join(sorted(s["train"])))
open("D:/VisionX/outputs/gtsynth/hssd-val.txt", "w").write(" ".join(val[:30]))
open("D:/VisionX/outputs/gtsynth/hssd-test.txt", "w").write(" ".join(val[30:]))
PYEOF
  rm -rf data/gt-prueba-hssd
  log "== HSSD: escena de prueba"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc "export MAMBA_ROOT_PREFIX=~/mamba; cd /tmp; \
    ~/bin/micromamba run -n habitat python /mnt/d/VisionX/scripts/gt_sintetico/render_tanda.py \
    --hssd $HSSD --escenas 103997478_171030525 --salida /mnt/d/VisionX/data/gt-prueba-hssd \
    --variante prueba --perfil elp-media --vistas 4 --semilla 1" 2>&1 | tr -d '\0' \
    | grep -E "vistas$|Traceback|ESP_CHECK" | tee -a "$LOG"
  n=$(ls -d data/gt-prueba-hssd/prueba/*/* 2>/dev/null | wc -l)
  if [ "$n" -lt 2 ]; then log "== HSSD: la escena de prueba no renderizó ($n vistas): HSSD no se corre"; exit 0; fi
  log "== HSSD: prueba ok ($n vistas)"
) & hb=$!

wait $a2 $a3 $a4
listo camara-sorteada-ojo
listo elp-margen-ojo
wait $a1
listo camara-sorteada-ojo
wait $hb
if [ -f outputs/gtsynth/hssd-train.txt ] && [ -d data/gt-prueba-hssd/prueba ]; then
  log "##### B: HSSD"
  read -r -a TR < outputs/gtsynth/hssd-train.txt
  read -r -a VA < outputs/gtsynth/hssd-val.txt
  # 5 flujos: train en 4 partes (cada una hace las dos variantes) + val.
  pids=()
  for k in 0 1 2 3; do
    parte=$(printf "%s\n" "${TR[@]}" | awk -v k=$k 'NR % 4 == k' | tr '\n' ' ')
    (
      render hssd-elp-margen hssd "$parte" --perfil elp-media --vistas 12 --semilla $((20 + k))
      render hssd-camara-sorteada hssd "$parte" --perfil sorteada --vistas 16 --semilla $((30 + k)) --ojo-sorteado 0.002
    ) & pids+=($!)
  done
  (
    render hssd-elp-margen hssd "${VA[*]}" --perfil elp-media --vistas 10 --semilla 40
    render hssd-camara-sorteada hssd "${VA[*]}" --perfil sorteada --vistas 10 --semilla 41 --ojo-sorteado 0.002
  ) & pids+=($!)
  wait "${pids[@]}"
  listo hssd-elp-margen
  listo hssd-camara-sorteada
fi
log "##### noche 2: terminada"
