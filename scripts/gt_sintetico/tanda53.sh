#!/usr/bin/env bash
# Tanda nocturna del issue #53 (decidida por Tomás el 19/11): E1–E6 y ojo de pez al final.
# Correr desde Git Bash en la PC de escritorio: bash scripts/gt_sintetico/tanda53.sh
# Por tanda: render (WSL + Habitat) -> `gtsynth verificar` (20 vistas) -> copia a Drive si
# existe outputs/gtsynth/destino_drive.txt (una línea con la carpeta de destino, en Windows).
# Log: outputs/gtsynth/tanda53.log. Si una tanda falla, se anota y sigue la siguiente.
set -u
cd /d/VisionX
LOG=outputs/gtsynth/tanda53.log
mkdir -p outputs/gtsynth
DATA=/mnt/d/VisionX/data/gt-sintetico-elp
ELP=$DATA/elp
VAL="frl_apartment_3 office_2 room_1 hotel_0"
TRAIN="apartment_0 apartment_1 frl_apartment_0 frl_apartment_1 frl_apartment_2 office_0 office_1 room_0"
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

verificar() {  # verificar <variante> (pinhole)
  log "== $1: verificar"
  PYTHONIOENCODING=utf-8 "$PY" -m visionx.gtsynth verificar data/gt-sintetico-elp --variante "$1" --max 20 \
    2>&1 | tail -1 | tee -a "$LOG"
}

copiar() {  # copiar <variante>: a Drive personal, si ya se definió el destino
  if [ -f outputs/gtsynth/destino_drive.txt ]; then
    local dst; dst=$(head -1 outputs/gtsynth/destino_drive.txt | tr -d '\r')
    log "== $1: copia a $dst"
    MSYS_NO_PATHCONV=1 robocopy "D:\\VisionX\\data\\gt-sintetico-elp\\$1" "$dst\\$1" /E /MT:8 /NFL /NDL /NJH /NP \
      | tail -4 | tee -a "$LOG"
  else
    log "== $1: sin destino de Drive todavía (se copia después)"
  fi
}

paso() {  # paso <variante> <escenas> <verificar sí/no> <args...>
  local var=$1 esc=$2 ver=$3; shift 3
  render "$var" "$esc" "$@"
  [ "$ver" = si ] && verificar "$var"
  copiar "$var"
}

log "##### tanda #53 empieza"
# E1 val: la ELP con margen, mismas poses que elp.
paso elp-margen "$VAL" si --perfil elp --poses-desde $ELP
# E2 val: seis perfiles de otras cámaras, mismas poses.
for p in angosta-60 base-corta-3 media-90 cuatro-tercios-90 base-larga-20 gran-angular-130; do
  paso $p "$VAL" si --perfil $p --poses-desde $ELP
done
# E1 train: la ELP con margen a media resolución, 300 vistas por escena.
paso elp-margen "$TRAIN" si --perfil elp-media --vistas 300 --semilla 1
# E3 train: una cámara sorteada por vista, 300 por escena.
paso camara-sorteada "$TRAIN" si --perfil sorteada --vistas 300 --semilla 2
# E4 val: ojo derecho desplazado (+y arriba, −z adelante).
paso elp-arriba-1 "$VAL" si --perfil elp --poses-desde $ELP --ojo-derecho 0 0.001 0
paso elp-arriba-2 "$VAL" si --perfil elp --poses-desde $ELP --ojo-derecho 0 0.002 0
paso elp-adelante-2 "$VAL" si --perfil elp --poses-desde $ELP --ojo-derecho 0 0 -0.002
paso elp-arriba-1-adelante-2 "$VAL" si --perfil elp --poses-desde $ELP --ojo-derecho 0 0.001 -0.002
# E5 val: poses difíciles, 20 por escena de cada tipo.
for t in altura poca-textura cerca; do
  paso pose-$t "$VAL" si --perfil elp --tipo-pose $t --vistas 20 --semilla 3
done
# E6 val: 10 recorridos de 8 cuadros por escena.
paso secuencias "$VAL" si --perfil elp --tipo-pose secuencia --vistas 80 --semilla 4
# E7 val (solo para evaluar): ojo de pez, mismas poses. Sin `verificar` (no es pinhole).
paso ojo-de-pez "$VAL" no --perfil ojo-de-pez --ojo-de-pez --margen 1.0 --poses-desde $ELP
log "##### tanda #53 terminada"
