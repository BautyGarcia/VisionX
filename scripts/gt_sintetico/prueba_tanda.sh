#!/usr/bin/env bash
# Prueba corta de cada modo de render_tanda.py (3 vistas, una escena de val) antes de la noche.
# Correr dentro de WSL: bash /mnt/d/VisionX/scripts/gt_sintetico/prueba_tanda.sh
export MAMBA_ROOT_PREFIX=~/mamba
R=/mnt/d/VisionX/scripts/gt_sintetico/render_tanda.py
OUT=/mnt/d/VisionX/data/gt-prueba53
ELP=/mnt/d/VisionX/data/gt-sintetico-elp/elp
run() {
  local inicio=$(date +%s)
  ~/bin/micromamba run -n habitat python "$R" --replica /mnt/d/datasets/replica --escenas room_1 \
    --salida "$OUT" "$@" 2>&1 | grep -E "vistas|Error|Traceback|rror:" | grep -v Reloading
  echo "   ($(( $(date +%s) - inicio )) s) $*"
}
rm -rf "$OUT"
run --perfil elp --variante elp-margen --poses-desde "$ELP" --max-vistas 3
run --perfil angosta-60 --variante angosta-60 --poses-desde "$ELP" --max-vistas 3
run --perfil gran-angular-130 --variante gran-angular-130 --poses-desde "$ELP" --max-vistas 3
run --perfil cuatro-tercios-90 --variante cuatro-tercios-90 --poses-desde "$ELP" --max-vistas 3
run --perfil elp-media --variante elp-media --vistas 3
run --perfil sorteada --variante camara-sorteada --vistas 4
run --perfil elp --variante elp-arriba-1-adelante-2 --ojo-derecho 0 0.001 -0.002 --poses-desde "$ELP" --max-vistas 3
run --perfil elp --variante pose-poca-textura --tipo-pose poca-textura --vistas 3
run --perfil elp --variante pose-cerca --tipo-pose cerca --vistas 3
run --perfil elp --variante pose-altura --tipo-pose altura --vistas 3
run --perfil elp --variante secuencias --tipo-pose secuencia --vistas 16
run --perfil ojo-de-pez --variante ojo-de-pez --ojo-de-pez --margen 1.0 --poses-desde "$ELP" --max-vistas 3
