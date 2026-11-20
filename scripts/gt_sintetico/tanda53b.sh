#!/usr/bin/env bash
# Segunda parte de la tanda #53 (relanzada 20/11 ~0:40), en paralelo: sortear poses nuevas cuesta
# ~8 s por vista y un render usa menos de 2 núcleos de 8. Ya hechos: elp-margen val y los 6 perfiles.
# Cinco flujos a la vez:
#   1-2  elp-margen train (media resolución), 4 escenas cada uno
#   3-4  camara-sorteada train, 4 escenas cada uno
#   5    val en serie: ojo derecho corrido, poses difíciles, secuencias, ojo de pez
# Cada variante se verifica (20 vistas) y se copia al Drive cuando terminan todos sus flujos.
set -u
cd /d/VisionX
LOG=outputs/gtsynth/tanda53.log
DATA=/mnt/d/VisionX/data/gt-sintetico-elp
ELP=$DATA/elp
VAL="frl_apartment_3 office_2 room_1 hotel_0"
TRAIN_A="apartment_0 apartment_1 frl_apartment_0 frl_apartment_1"
TRAIN_B="frl_apartment_2 office_0 office_1 room_0"
PY=/d/VisionX/.venv/Scripts/python.exe

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

render() {  # render <variante> <escenas> <args...>
  local var=$1 esc=$2; shift 2
  log "== $var: render ($esc) $*"
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -lc "export MAMBA_ROOT_PREFIX=~/mamba; cd /tmp; \
    ~/bin/micromamba run -n habitat python /mnt/d/VisionX/scripts/gt_sintetico/render_tanda.py \
    --replica /mnt/d/datasets/replica --escenas $esc --salida $DATA --variante $var $*" 2>&1 \
    | tr -d '\0' | grep -E "vistas|Traceback|Error:|rror:" | grep -v Reloading | tee -a "$LOG"
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
    | tail -4 | tee -a "$LOG"
}

val_en_serie() {
  for spec in "elp-arriba-1|0 0.001 0" "elp-arriba-2|0 0.002 0" "elp-adelante-2|0 0 -0.002" \
              "elp-arriba-1-adelante-2|0 0.001 -0.002"; do
    var=${spec%%|*}; ojo=${spec#*|}
    render "$var" "$VAL" --perfil elp --poses-desde $ELP --ojo-derecho $ojo
    verificar "$var"; copiar "$var"
  done
  for t in altura poca-textura cerca; do
    render pose-$t "$VAL" --perfil elp --tipo-pose $t --vistas 20 --semilla 3
    verificar pose-$t; copiar pose-$t
  done
  render secuencias "$VAL" --perfil elp --tipo-pose secuencia --vistas 80 --semilla 4
  verificar secuencias; copiar secuencias
  render ojo-de-pez "$VAL" --perfil ojo-de-pez --ojo-de-pez --margen 1.0 --poses-desde $ELP
  copiar ojo-de-pez
}

log "##### tanda #53, segunda parte (5 flujos en paralelo)"
render elp-margen "$TRAIN_A" --perfil elp-media --vistas 300 --semilla 1 & p1=$!
render elp-margen "$TRAIN_B" --perfil elp-media --vistas 300 --semilla 11 & p2=$!
render camara-sorteada "$TRAIN_A" --perfil sorteada --vistas 300 --semilla 2 & p3=$!
render camara-sorteada "$TRAIN_B" --perfil sorteada --vistas 300 --semilla 12 & p4=$!
val_en_serie & p5=$!
wait $p1 $p2; verificar elp-margen; copiar elp-margen
wait $p3 $p4; verificar camara-sorteada; copiar camara-sorteada
wait $p5
log "##### tanda #53 terminada"
