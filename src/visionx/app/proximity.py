"""Alerta de cercanía (T20): avisar de un obstáculo muy cerca adelante, sin orden de por medio.

Observador del controlador, como `camera_status`. Política (TECNICO «Alerta — T20»):

- **Zona central:** columnas `zone_x` y filas `zone_y` (fracciones de la imagen de profundidad).
- **Piso excluido** (método declarado): con la cámara a `camera_height_m` del piso y nivelada,
  un píxel a profundidad Z y fila v está a altura y = (v − cy)·Z/fy por debajo del eje óptico;
  si y ≥ camera_height_m − floor_margin_m es piso y no cuenta. Cámara inclinada hacia abajo =
  más piso detectado como obstáculo: por eso además se cortan las filas de abajo (`zone_y`).
- **Persistencia:** la condición (fracción `min_fraction` de la zona válida a menos de `near_m`,
  y al menos `min_px` píxeles) tiene que sostenerse `hold_s` en mediciones distintas.
- **Profundidad fresca:** mediciones más viejas que `max_age_s` suspenden la alerta.
- **Prioridad ALERT** (corta descripciones y lecturas) y pausa `cooldown_s` entre avisos; se
  vuelve a armar recién cuando la zona se despeja.
- Callada durante los modos de `silent_modes` (leer: el objeto se acerca a propósito;
  distancia: acercarse al objeto es lo pedido).
- **Mientras guía** (`guide_modes`, buscar): el objetivo está adelante y se acerca a propósito,
  así que sus píxeles (su máscara de YOLO) no cuentan; avisa solo de OTRA cosa en el camino.
- `only_modes` (por defecto, buscar): fuera de esos modos no avisa (en el stand, nadie camina).
  Vacío = siempre. Sin bip (Tomás, 19/11): «Cuidado: algo adelante, a 0,6 metros.».

No es navegación validada: es un aviso más, con su tasa de falsas alarmas medida en clips.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionx.app.controller import Say
from visionx.shared.contracts import DepthStatus, Priority


@dataclass(frozen=True, slots=True)
class ProximitySettings:
    enabled: bool = True
    near_m: float = 0.8
    zone_x: tuple[float, float] = (0.3, 0.7)
    zone_y: tuple[float, float] = (0.15, 0.75)
    camera_height_m: float = 1.3
    floor_margin_m: float = 0.3
    min_fraction: float = 0.25
    min_px: int = 200
    # Zona casi sin píxeles válidos (pared lisa, contraluz): SGBM inventa disparidad grande.
    min_coverage: float = 0.3
    ceiling_m: float = 0.5  # más de esto por encima del eje óptico es techo (cámara mirando arriba)
    quiet_after_s: float = 5.0  # sin avisos este tiempo después de que termina una orden
    hold_s: float = 0.4
    max_age_s: float = 0.8
    cooldown_s: float = 6.0
    silent_modes: tuple[str, ...] = ("read", "distance")
    guide_modes: tuple[str, ...] = ("search",)
    only_modes: tuple[str, ...] = ("search",)
    phrase: str = "Cuidado: algo adelante, a {metros}."

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ProximitySettings:
        c = dict(cfg.get("proximity") or {})
        for key in ("zone_x", "zone_y", "silent_modes", "guide_modes", "only_modes"):
            if key in c:
                c[key] = tuple(c[key])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in c.items() if k in known})


@dataclass
class ProximityAlert:
    """`observe(view, now_ns) -> list[Say]`; `active_mode` lo pone el controlador."""

    s: ProximitySettings = field(default_factory=ProximitySettings)
    geometry: Any = None  # StereoGeometry de los cuadros rectificados (para fy y cy)
    active_mode: str | None = None
    active_target: str | None = None  # clase del objetivo de buscar (lo pone el controlador)
    last_fraction: float = 0.0
    last_z_m: float | None = None
    last_coverage: float = 0.0
    _since_ns: int | None = None
    _last_key: Any = None
    _armed: bool = True
    _said_ns: int | None = None
    _hits: int = 0
    _mode_ended_ns: int | None = None
    _prev_mode: str | None = None

    def observe(self, view: Any, now_ns: int) -> list[Say]:
        if not self.s.enabled:
            return []
        if self._prev_mode is not None and self.active_mode is None:
            self._mode_ended_ns = now_ns  # recién terminó una orden («Llegaste», una lectura)
        self._prev_mode = self.active_mode
        if self.s.only_modes and self.active_mode not in self.s.only_modes:
            self._since_ns, self._hits = None, 0
            return []
        joint = getattr(view, "measured", None)
        depth = getattr(joint, "depth", None)
        if depth is None or depth.depth_m is None or depth.status is DepthStatus.UNMEASURABLE:
            self._since_ns = None
            return []
        if (now_ns - joint.frame.received_ns) / 1e9 > self.s.max_age_s:
            self._since_ns, self._hits = None, 0  # datos viejos: suspendida
            return []
        key = joint.frame.key
        if key == self._last_key:
            return []  # misma medición: no cuenta dos veces
        self._last_key = key
        near = self.check(depth.depth_m, joint.frame, self._target_mask(joint, depth.depth_m.shape))
        if not near:
            self._since_ns, self._hits, self._armed = None, 0, True
            return []
        self._hits += 1
        if self._since_ns is None:
            self._since_ns = joint.frame.received_ns
        held = (joint.frame.received_ns - self._since_ns) / 1e9 >= self.s.hold_s
        if not held or self._hits < 2 or not self._armed:
            return []
        if self.active_mode in self.s.silent_modes:
            return []
        if (
            self._mode_ended_ns is not None
            and (now_ns - self._mode_ended_ns) / 1e9 < self.s.quiet_after_s
        ):
            return []
        if self._said_ns is not None and (now_ns - self._said_ns) / 1e9 < self.s.cooldown_s:
            return []
        self._armed, self._said_ns = False, now_ns
        z = self.last_z_m if self.last_z_m is not None else self.s.near_m
        metros = (
            f"{z:.1f} metros".replace(".", ",")
            if z >= 1
            else f"{round(z * 100, -1):.0f} centímetros"
        )
        return [Say(self.s.phrase.format(metros=metros), Priority.ALERT)]

    def _target_mask(self, joint: Any, shape: tuple[int, int]) -> np.ndarray | None:
        """Mientras guía: los píxeles del objetivo (no son un obstáculo), en la grilla de Z."""
        if self.active_mode not in self.s.guide_modes or not self.active_target:
            return None
        det = getattr(joint, "detection", None)
        if det is None or det.masks is None:
            return None
        idx = [i for i in range(len(det)) if det.classes[i] == self.active_target]
        if not idx:
            return None
        import cv2

        union = np.any(det.masks[idx], axis=0).astype(np.uint8)
        small = cv2.resize(union, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.dilate(small, np.ones((5, 5), np.uint8)).astype(bool)

    def check(self, depth_m: np.ndarray, frame: Any, exclude: np.ndarray | None = None) -> bool:
        """¿Hay un obstáculo cerca en la zona central, sin contar el piso?"""
        h, w = depth_m.shape
        x0, x1 = (int(f * w) for f in self.s.zone_x)
        y0, y1 = (int(f * h) for f in self.s.zone_y)
        zone = depth_m[y0:y1, x0:x1]
        valid = np.isfinite(zone) & (zone > 0)
        if exclude is not None:
            valid &= ~exclude[y0:y1, x0:x1]
        self.last_coverage = float(valid.mean()) if valid.size else 0.0
        if valid.sum() < self.s.min_px:
            self.last_fraction, self.last_z_m = 0.0, None
            return False
        # Altura bajo el eje óptico con la geometría escalada a la grilla de profundidad.
        fy, cy = self._intrinsics(h)
        rows = np.arange(y0, y1, dtype=np.float32)[:, None]
        below = (rows - cy) * np.where(valid, zone, 0) / fy
        floor = below >= self._camera_height() - self.s.floor_margin_m
        floor |= below <= -self.s.ceiling_m  # techo o algo muy arriba: no es un obstáculo de paso
        close = valid & ~floor & (zone < self.s.near_m)
        fraction = float(close.sum()) / float(valid.sum())
        self.last_fraction = fraction
        self.last_z_m = float(np.median(zone[close])) if close.any() else None
        return (
            close.sum() >= self.s.min_px
            and fraction >= self.s.min_fraction
            and self.last_coverage >= self.s.min_coverage
        )

    def _camera_height(self) -> float:
        """La de la persona (estatura y piso, modes.body) si se conoce; si no, la de la config."""
        from visionx.modes import body

        b = body.current()
        return b.camera_m if (b.height_known or b.floor_seen) else self.s.camera_height_m

    def _intrinsics(self, h: int) -> tuple[float, float]:
        """fy, cy de la grilla de profundidad. Sin geometría conocida: campo vertical ~60°."""
        if self.geometry is not None:
            scale = h / self.geometry.height
            return self.geometry.fy * scale, self.geometry.cy * scale
        return h / (2 * np.tan(np.radians(30))), h / 2
