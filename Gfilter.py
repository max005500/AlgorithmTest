import numpy as np
import KolmogorovPSD as cm

def slope_weighting_function(d: float,
                             dx: float,
                             nsubx: int,
                             samp: int,
                             cn2r0: float = 1.,
                             wavelength: float =500e-9,
                             h:float =0.,
                             scalingFactor: float = 1.,
                             glob: bool = False):

    """
    calculate weighting function (cov) for SHIMM.
    
    Args:

        dx(float)           : steps in phychic aperture [m]
        h (float)           : layer altitude    [m].
        d (float)           : sub-aperture size [m].
        samp(int)           : frequency oversampling
        glob (Bool)         : if you want to use a Phi_K with a global r0 or not
        nsubx (int)         : grid size.
        cn2r0(float)        : value of r0 or cn2 in K_PSD depends on glob
        sclaingFactor(float): if you want to get the result in rad2 or arcsec2

    Returns:
    return  W_z (ndarray): [2,N,N]. where N = 2*nsubx-1 

    cov matrix for x-slope and y-slope .

    cov[0, di, dj] -> cov_xx  
    cov[1, di, dj] -> cov_yy 

    """
    # C: matriz espacial (p.ej. covarianza) tamaño N0xN0
    Nfft = max(256, samp) 

    # 1. spatio-frequency domain (fx, fy)
    freq = np.fft.fftfreq(Nfft, d=dx)   #<-- spatio-frequency vector based on sub-aperture grid size [cycles/m]
    fx, fy = np.meshgrid(freq, freq) 

    # spatio-frequency magnitude 
    f = np.hypot(fx,fy)   # f = ||f||
    f2 = f*f

    # ---------------------------------------------------------
    
    # 3. Shack-Hartmann Aperture filter (Ec. 2.44)
    A_f =  (np.sinc(d * fx)**2) * (np.sinc(d * fy)**2)
    

    # 4. propagation ( sinusoidal term in Ec. 2.42/2.45) fresnel phase param
    Fresnel_term = np.cos(np.pi * wavelength * h * f2)**2

    #5. Kolmogorov PSD 
    Phi_K = cm.Kolmogorov_PSD(f,wavelength,glob,cn2r0)
    scaling = wavelength**2

    spectral_densityX = fx * fx * Phi_K * Fresnel_term * A_f * scaling 
    spectral_densityY = fy * fy * Phi_K * Fresnel_term * A_f * scaling

    df = freq[1] - freq[0]
    fft_integral_scale = (Nfft**2) * (df**2)

    # 6. Numerical solution using IFFT 
    covariance_mapX = np.fft.ifft2(spectral_densityX)  * fft_integral_scale
    covariance_mapY = np.fft.ifft2(spectral_densityY)  * fft_integral_scale
    
    # fftshift centra la frecuencia cero en el medio de la imagen para visualización
    covariance_mapX = np.fft.fftshift(covariance_mapX) 
    covariance_mapY = np.fft.fftshift(covariance_mapY) 

    mid = int(Nfft/2)
    stride = int(round(d / dx))   # if dx=d/2 -> stride=2
    k = np.arange(-(nsubx-1), nsubx)  # [-5..+5] for nsubx=6
    idx = mid + stride * k

    W_zX = np.real(covariance_mapX[np.ix_(idx,idx)]) * scalingFactor  # X slope on [Rad^2] -> [arcsec^2]
    W_zY = np.real(covariance_mapY[np.ix_(idx,idx)]) * scalingFactor  # Y slope on [Rad^2] -> [arcsec^2]

    return np.array([W_zX,W_zY])



def scintillation_weighting_function(d: float,
                             dx: float,
                             nsubx: int,
                             samp: int,
                             cn2r0: float = 1.,
                             wavelength: float =500e-9,
                             h:float =0.,
                             scalingFactor: float = 1.,
                             glob: bool = False):

    """
    calculate weighting function (cov) for SHIMM.
    
    Args:

        dx(float)           : steps in phychic aperture [m]
        h (float)           : layer altitude    [m].
        d (float)           : sub-aperture size [m].
        samp(int)           : frequency oversampling
        glob (Bool)         : if you want to use a Phi_K with a global r0 or not
        nsubx (int)         : grid size.
        cn2r0(float)        : value of r0 or cn2 in K_PSD depends on glob
        sclaingFactor(float): if you want to get the result in rad2 or arcsec2

    Returns:
    return  W_z (ndarray): [N,N]. where N = 2*nsubx-1 

    cov matrix for Scintillation index .

    cov[di, dj] -> cov_I 

    """
    
    # 1. spatio-frequency domain (fx, fy)
    Nfft = max(256, samp) 

    # 1. spatio-frequency domain (fx, fy)
    freq = np.fft.fftfreq(Nfft, d=dx)   #<-- spatio-frequency vector based on sub-aperture grid size [cycles/m]
    fx, fy = np.meshgrid(freq, freq) 

    # spatio-frequency magnitude 
    f = np.hypot(fx,fy)   # f = ||f||
    f2 = f*f
    
    # ---------------------------------------------------------
    
    # 3. aperture filter Shack-Hartmann (Ec. 2.44)
    A_f =  (np.sinc(d * fx)**2) * (np.sinc(d * fy)**2)
    
    # 4. propagation ( sinusoidal term in Ec. 2.42/2.45) fresnel phase param
    Fresnel_term = np.sin(np.pi * wavelength * h * f2)**2

    #5. Kolmogorov PSD obtain
    Phi_K = cm.Kolmogorov_PSD(f,wavelength,glob,cn2r0)

    spectral_density = Phi_K * Fresnel_term * A_f * 4

    # 6. Numerical solution using IFFT 
    df = freq[1] - freq[0]
    fft_integral_scale = (Nfft**2) * (df**2)

    covariance_map = np.fft.ifft2(spectral_density) * fft_integral_scale
    covariance_map = np.fft.fftshift(covariance_map) 
    
    # fftshift centra la frecuencia cero en el medio de la imagen para visualización

    mid = int(Nfft/2) 
    stride = int(round(d / dx))   # si dx=d/2 -> stride=2
    k = np.arange(-(nsubx-1), nsubx)  # [-5..+5] para nsubx=6
    idx = mid + stride * k

    # W_z = np.real(covariance_map[mid:mid+N0,mid:mid+N0]) * scalingFactor  # Y slope on [Rad^2] -> [arcsec^2]
    W_z = np.real(covariance_map[np.ix_(idx,idx)]) * scalingFactor  

    return W_z



