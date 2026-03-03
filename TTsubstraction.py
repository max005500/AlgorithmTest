import numpy as np

def tip_tilt_sub(cov: np.ndarray, pupil_mask: np.ndarray):
    """
        slodar_refFuncs2D():
      - proyect cov to sub-apertures pairs 
      - tip/tilt subtraction 
      - Re-bined 
    """
    nsubx = pupil_mask.shape[0]
    nn = 2 * nsubx - 1

    # índices de sub-aperturas activas
    active = [(i, j) for j in range(nsubx) for i in range(nsubx) if pupil_mask[j, i] > 0]
    nsubtot = len(active)

    psfs = np.zeros((2, nn, nn), dtype=np.float64)

    # pcov: matriz completa (nsubtot x nsubtot) para x e y intercalados
    pcov = np.zeros((nsubtot, nsubtot, 2), dtype=np.float64)

    # 1) llenar pcov tomando valores desde cov[2*delta] (x) y cov[2*delta+1] (y)
    for a, (i1, j1) in enumerate(active):
        for b, (i2, j2) in enumerate(active):
            di = i2 - i1 + (nsubx - 1)
            dj = j2 - j1 + (nsubx - 1)
            pcov[a, b, 0] = cov[1, di, dj]
            pcov[a, b, 1] = cov[0, di, dj]

    # 2) tip/tilt subtraction: C' = C - rowMean - colMean + globalMean
    row_mean = pcov.mean(axis=1, keepdims=True)   # (nsubtot,1,2)
    col_mean = pcov.mean(axis=0, keepdims=True)   # (1,nsubtot,2)
    glob_mean = pcov.mean(axis=(0, 1), keepdims=True)  # (1,1,2)
    pcov2 = pcov - row_mean - col_mean + glob_mean

    # 3) rebin a separaciones (nn x nn) 
    acc = np.zeros((nn, nn, 2), dtype=np.float64)
    cnt = np.zeros((nn, nn), dtype=np.int64)

    for a, (i1, j1) in enumerate(active):
        for b, (i2, j2) in enumerate(active):
            di = i2 - i1 + (nsubx - 1)
            dj = j2 - j1 + (nsubx - 1)

            acc[di, dj, :] += pcov2[a, b, :]
            cnt[di, dj] += 1

    # cnt>0
    m = cnt > 0
    psfs[0, m] = (acc[m, 0] / cnt[m])
    psfs[1, m] = (acc[m, 1] / cnt[m])

    return psfs