"""GT-sintético-ELP: vistas de Replica con la geometría de la ELP (docs/GT-SINTETICO.md).

El render corre aparte (scripts/gt_sintetico/render_habitat.py, entorno de Habitat-Sim). Este
paquete lee las vistas, arma la tabla de objetos con su distancia de referencia y las variantes
de cámara, y evalúa contra ellas.
"""

from visionx.gtsynth.dataset import (
    PARTICION,
    REPLICA_A_COCO,
    ObjetoGT,
    Vista,
    objetos_de,
    vistas,
)

__all__ = ["PARTICION", "REPLICA_A_COCO", "ObjetoGT", "Vista", "objetos_de", "vistas"]
