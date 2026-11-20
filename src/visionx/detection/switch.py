"""Dos detectores, uno por cuadro: COCO para todo y YOLOE cuando se busca algo que COCO no ve.

Medido el 20/11 en Replica (`gtsynth deteccion`, 240 vistas): YOLOE con el vocabulario de
VisionX (215 clases) encuentra objetos que el YOLO COCO no conoce (mesa de luz 96 %, alfombra
73 %, almohada 79 %, lámpara 26 %), pero en las clases COCO acierta menos que yolo26s-seg con el
mismo recall (precisión 0,57 contra 0,70). Reemplazarlo empeoraría lo de siempre. Por eso:

- sin búsqueda, o buscando algo COCO (silla, taza, persona…): el detector COCO, como siempre;
- buscando o midiendo algo que solo conoce YOLOE (pizarrón, puerta, llaves…): YOLOE.

Corre **un** modelo por cuadro: la latencia no cambia. El controlador avisa qué se busca con el
mismo mecanismo que usan los avisos (`active_target` / `active_mode` antes de cada `observe`).
"""

from __future__ import annotations

from typing import Any

from visionx.shared.contracts import Detection, StereoFrame


class VocabularySwitch:
    """Detector COCO + detector de vocabulario abierto; elige uno por cuadro."""

    def __init__(self, base: Any, extra: Any) -> None:
        self.base = base
        self.extra = extra
        self._base_classes = frozenset(base.class_names)
        nuevas = tuple(c for c in extra.class_names if c not in self._base_classes)
        self.class_names: tuple[str, ...] = tuple(base.class_names) + nuevas
        self.model_id = f"{base.model_id}+{extra.model_id}"
        # Los escribe el controlador antes de cada observe() (ver Controller.tick).
        self.active_target: str | None = None
        self.active_mode: str | None = None
        self.last_model: str = base.model_id

    def uses_extra(self) -> bool:
        target = self.active_target
        return bool(target) and target not in self._base_classes and target in self.class_names

    def predict(self, frame: StereoFrame) -> Detection:
        detector = self.extra if self.uses_extra() else self.base
        self.last_model = detector.model_id
        return detector.predict(frame)

    @property
    def last_speed_ms(self) -> dict[str, float]:
        detector = self.extra if self.last_model == self.extra.model_id else self.base
        return getattr(detector, "last_speed_ms", {})

    def observe(self, view: Any, now_ns: int) -> list:
        """Para registrarse como observador del controlador: no dice nada."""
        return []
