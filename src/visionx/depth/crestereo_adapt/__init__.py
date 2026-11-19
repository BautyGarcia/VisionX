"""CREStereo adaptable (T12): CREStereo congelado, envuelto para usarlo desde VisionX.

CREStereo (Li et al., CVPR 2022) es una red de estéreo que ya entrenaron otros. Acá no se le
cambia ningún peso: se la carga, se le da un par rectificado y devuelve disparidad. El resto
del paquete son las piezas propias que la rodean (TECNICO, «CREStereo adaptable»).

    port.py        ubica y carga el código y los pesos descargados (fuera de Git)
    matcher.py     prepara el par, corre la red y devuelve la disparidad
    estimator.py   lo expone con el contrato DepthEstimator que ya usa la app
    correction.py  la corrección de un ojo respecto del otro como cuenta geométrica
    classic.py     estimador clásico de esa corrección con puntos emparejados
    perturb.py     generador de perturbaciones (etapa 1): desalinea un par de forma conocida

Código y pesos de CREStereo se bajan con `scripts/profundidad/descargar_crestereo.py`.
"""
