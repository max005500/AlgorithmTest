import numpy as np

def Kolmogorov_PSD(f: np.ndarray, wavelength: float, glob: bool, r0Cn2=1.0):
    """
        spectral filter Z_x(f) (Ec. 3.8).
    
        Args:
            f (ndarray)       : spatial frecuency in X.
            r0 float          : strength of the turbulent in the layer  
            glob (Bool)       : if you want to use a Phi_K with a global r0 or not
            
        Returns:

            Phi_K(F) (ndarray): kolmogorov_PSD.
    """

    k = 2 * np.pi / wavelength
    k2 = k*k

    Phi_K = np.zeros_like(f)

    # x/0 prevention
    mask_f = f > 0 

    if glob:
        # kolmogorov PSD Integrated r0 
        # r0 = cn2_to_r0(r0,lamda=wavelength)
        Phi_K = 0.023 *(r0Cn2 ** (-5/3)) * (f[mask_f] ** (-11/3)) 
       
    else:
        Cn2_dh = r0Cn2           # asumes a layer with cn2 = 1

        #Kolmogorov PSD eq (2.11): layer aproach
        Phi_K[mask_f] = 9.7e-3 * k2 * (f[mask_f] ** (-11/3)) * Cn2_dh

    # piston remove
    Phi_K[0,0] = 0

    return Phi_K

