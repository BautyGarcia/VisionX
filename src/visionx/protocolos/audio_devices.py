"""Nombres PortAudio legibles y selección sin acentos, sin abrir dispositivos."""

from __future__ import annotations

import re
import unicodedata


def readable_name(name: str) -> str:
    # Reparar UTF-8 leído como ANSI cuando todavía se conservan los bytes originales.
    for encoding in ("cp1252", "latin1"):
        try:
            fixed = name.encode(encoding).decode("utf-8")
        except (UnicodeError, LookupError):
            continue
        if fixed != name:
            return fixed
    return name


def normalized(name: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", name.casefold()) if not unicodedata.combining(c)
    )


def input_devices(sd=None) -> list[dict]:
    if sd is None:
        import sounddevice as sd

    apis = sd.query_hostapis()
    devices = [
        dict(index=i, name=readable_name(d["name"]), api=apis[d["hostapi"]]["name"])
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]
    for device in devices:
        # U+FFFD perdió el byte: recuperar el nombre de la misma entrada en otra API.
        # MME puede truncar a 31 caracteres; comparar como prefijo, sin inventar letras.
        if "�" in device["name"]:
            pattern = re.escape(normalized(device["name"])).replace("�", ".")
            siblings = [
                d
                for d in devices
                if "�" not in d["name"] and re.match(pattern, normalized(d["name"]))
            ]
            if siblings:
                device["name"] = min(siblings, key=_preference)["name"]
    return devices


def _preference(device):
    api = device["api"].casefold()
    return (
        "�" in device["name"],
        0 if "wasapi" in api else 1 if "directsound" in api else 2,
        device["index"],
    )


def resolve_device(device, sd=None):
    if device is None or isinstance(device, int):
        return device
    if device.isdigit():
        return int(device)
    needle = normalized(readable_name(device)).strip()
    if not needle:
        raise ValueError("nombre de dispositivo vacío")
    matches = [d for d in input_devices(sd) if needle in normalized(d["name"])]
    if not matches:
        raise ValueError(f"no hay entrada de audio que coincida con {device!r}")
    # Duplicados de una entrada entre APIs se resuelven por preferencia; entradas
    # distintas requieren una subcadena más específica o su índice.
    names = {normalized(d["name"]) for d in matches}
    if len(names) > 1:
        options = ", ".join(f"{d['index']}: {d['name']}" for d in matches)
        raise ValueError(f"dispositivo ambiguo {device!r}: {options}; usar índice o más texto")
    return min(matches, key=_preference)["index"]
