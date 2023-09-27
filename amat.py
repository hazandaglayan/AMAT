import numpy as np
from joblib import Parallel, delayed
import vip_hci as vip
from vip_hci.var import get_annulus_segments
from vip_hci.var import  frame_center
from hciplot import plot_frames
from l1lra import L1LRAcd
from utils import trajectory_mask, psf_cube, pixels_in_annulus

def amat_all(cube, angle_list, psfn, pixels=None, prad=1, fwhm=4, ncomp=1, 
             n_jobs=1, asize=4, MAX_ITER=100, norm=1, full_output=False, 
             maxiter_inner=1, alg='ann', eps=0.001):
    """
    Computes flux map and models the PSF with AMAT algorithm.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        ADI cube
    angle_list : numpy ndarray, 1d
        Parallactic angles
    psfn : numpy ndarray, 2d or 3d
        normalized PSF
    pixels : tuple, optional
        starting pixels for trajectories. The default is None.
    prad : int or float, optional
        coefficient for the radius of apendeture (prad*fwhm). 
        The default is 1.
    fwhm : float, optional
        the size of FWHM. The default is 4.
    ncomp : int, optional
        number of components (rank). The default is 1.
    n_jobs : int, optional
        number of jobs for parallel computing. The default is 1.
    asize : float, optional
        annular size. The default is 4.
    MAX_ITER : int, optional
        max iteration number for AMAT. The default is 100.
    norm : int, optional
        norm, 1 or 2. The default is 1.
    full_output : boolean, optional
        Return flux map or cube-fluxmap-processing frame-iteration map. 
        The default is False.
    maxiter_inner : int, optional
        max iteartion number for L1-LRA. The default is 1.
    alg : string, optional
        full or annular version. The default is 'ann'.
    eps : float, optional
        max relative error for a_g. The default is 0.001.

    Returns
    -------
    Return processing frame, rotated model PSF (cube), flux map, and 
    iteration numbers map.

    """
    nfr, m, n =  cube.shape
    cy, cx = frame_center(cube)
    
    if not isinstance(pixels, tuple):
        pixels = pixels_in_annulus(cube[0].shape, (cy,cx), 6, n/2-2)
    

    if n_jobs == 1:  # Not parallel
        frame_value_list = []
        planet_pos_list = []
        res_array_list = []
        flux_list = []
        iter_list = []
        count = 0
        

        for planet_position in zip(*pixels):
            values = amat_for_a_trajectory(cube, angle_list, psfn, planet_position, 
                                           prad, ncomp, fwhm, MAX_ITER, asize, 
                                           norm, maxiter_inner, alg, eps, False)
            frame_value, res_array, planet_pos, flux, i = values
            frame_value_list.append(frame_value)
            res_array_list.append(res_array)
            flux_list.append(flux)
            iter_list.append(i)
            planet_pos_list.append(planet_pos)
            count = count+1    
    else:        
        with Parallel(n_jobs=n_jobs, verbose=True) as parallel:
            results = zip(*parallel(delayed(amat_for_a_trajectory)(cube, angle_list, psfn,
                                                                   planet_position,prad, 
                                                                   ncomp, fwhm, MAX_ITER, 
                                                                   asize, norm, 
                                                                   maxiter_inner, alg, 
                                                                   eps, False)
                               for planet_position in zip(*pixels)))
        frame_value_list, res_array_list, planet_pos_list, flux_list, iter_list =results
    
    frame = np.zeros_like(cube[0,:,:])
    residual_cube_ = np.zeros_like(cube)
    fluxmap = np.zeros_like(cube[0,:,:])
    iter_map = np.zeros_like(cube[0,:,:])


    for frame_value, res_array, planet_pos, a, i in zip(frame_value_list, 
                                                        res_array_list, 
                                                        planet_pos_list, 
                                                        flux_list,
                                                        iter_list):
        frame[planet_pos[0], planet_pos[1]] = frame_value
        residual_cube_[:, planet_pos[0], planet_pos[1]] = res_array
        fluxmap[planet_pos[0], planet_pos[1]] = a 
        iter_map[planet_pos[0], planet_pos[1]] = i


    if full_output:
        return frame, residual_cube_, fluxmap, iter_map
    else:    
        return frame


def amat_for_a_trajectory(cube, angle_list, psfn, planet_position, prad=1, 
                          ncomp=1, fwhm=4, MAX_ITER=100, asize=4, norm=1, 
                          maxiter_inner=5, alg='ann', eps=0.001, map_output=False, plot=False):
    """
    Computes flux and model PSF for given trajectory.

    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        ADI cube
    angle_list : numpy ndarray, 1d
        Parallactic angles
    psfn : numpy ndarray, 2d or 3d
        normalized PSF
    planet_position : tuple, optional
        starting pixels for trajectories. The default is None.
    prad : int or float, optional
        coefficient for the radius of apendeture (prad*fwhm). 
        The default is 1.
    fwhm : float, optional
        the size of FWHM. The default is 4.
    ncomp : int, optional
        number of components (rank). The default is 1.
    n_jobs : int, optional
        number of jobs for parallel computing. The default is 1.
    asize : float, optional
        annular size. The default is 4.
    MAX_ITER : int, optional
        max iteration number for AMAT. The default is 100.
    norm : int, optional
        norm, 1 or 2. The default is 1.
    maxiter_inner : int, optional
        max iteartion number for L1-LRA. The default is 1.
    alg : string, optional
        full or annular version. The default is 'ann'.
    eps : float, optional
        max relative error for a_g. The default is 0.001.

    Returns
    -------
    Return the value in processing frame, the vector in rotated model PSF 
    (cube), flux, and iteration number.

    """
    
    cy, cx = vip.var.frame_center(cube)
    y, x = planet_position
    y_cy, x_cx = y - cy, x - cx
    rad, theta = vip.var.cart_to_pol(x_cx, y_cy)
    mask = trajectory_mask(cube.shape, (cy, cx), angle_list, rad, theta, fwhm, prad)
    P = psf_cube(cube, psfn, angle_list, rad, theta)
    if alg == 'ann':
        res = AMAT_ann(cube, P, mask, angle_list, norm, ncomp, MAX_ITER, rad, 
                       asize, fwhm, maxiter_inner, eps)
    elif alg == 'full':
        res = AMAT_full(cube, P, mask, angle_list, norm, ncomp, MAX_ITER, 
                        maxiter_inner, eps)
        
    
    frame, L, res_, a, i = res
    
    from vip_hci.metrics import snrmap
    if plot:
        snr = snrmap(frame, fwhm, verbose=False)
        if x == 42:
            plot_frames(snr, circle=(x,y), vmax=18, vmin=-4.30, axis=False)
            plot_frames(snr, circle=(x,y), save='l{}_iter{}.pdf'.format(norm,MAX_ITER), axis=False, vmax=18, vmin=-4.30)
        else:
            plot_frames(snr, circle=(x,y), vmax=10, vmin=-4.30, axis=False)
            plot_frames(snr, circle=(x,y), save='l{}_iter{}_noplanet.pdf'.format(norm,MAX_ITER), axis=False, vmax=10, vmin=-4.30)
            
    
        snr_value = vip.metrics.snr(frame, (x,y), fwhm)
        print("Iter:", MAX_ITER)
        print("SNR:", snr_value)
        print("a:", a)
    
    if map_output:
        return frame, res_, planet_position, a, i
    else:
        return frame[planet_position[0],planet_position[1]], res_[:,planet_position[0],planet_position[1]], planet_position, a, i 

# Annular version of AMAT
def AMAT_ann(cube, P, mask, angles, norm, rank, MAX_ITER, rad=4, asize=4, fwhm=4, 
             maxiter_l1=1, eps=0.001):
    """
    Annular version of AMAT.

    """
    
    nfr, m, n = cube.shape
    yy, xx = get_annulus_segments(cube[0], rad-asize, asize*2)[0]

    ann_cube = cube[:, yy, xx]    
    ann_P_cube = P[:, yy, xx]
    mask_  = mask[:, yy, xx]
    


    U = None
    V = None
    rel_error= []
    a_amat = np.zeros(MAX_ITER+1)
    for i in range(MAX_ITER):
        res = ann_cube-a_amat[i]*ann_P_cube

        # Find L(k+1)
        if norm == 2:
            u,s,v = np.linalg.svd(res, full_matrices=False)
            L = np.dot(u[:,:rank]*s[:rank],v[:rank])

        else:
            U, V, rel_error = L1LRAcd(res, r=rank, maxiter=maxiter_l1, 
                                      U0=U, V0=V, rel_error=rel_error)
            L = np.dot(U,V)
            


        # Find a(k+1)
        residuals = ann_cube-L
        a_amat[i+1] = calculate_flux(residuals, ann_P_cube, mask_, norm)
        if np.abs(a_amat[i+1]-a_amat[i])/np.abs(a_amat[i+1])<eps:
            break



    # Then obtain residual cube, derotate the cube, and obtain median frame
    res = ann_cube-L
    res_fin = np.zeros_like(cube)
    res_fin[:, yy, xx] = res
    residuals_cube_der = vip.preproc.cube_derotate(res_fin, angles, imlib='opencv')
    frame_amat = np.nanmedian(residuals_cube_der, axis=0)
    return frame_amat, L, residuals_cube_der, a_amat[i+1], i+1

# Annular version of AMAT
def AMAT_full(cube, P, mask, angles, norm=1, rank=1, MAX_ITER=100, maxiter_l1=1, 
              eps=0.001):
    """
    Full version of AMAT.

    """
    
    nfr, m, n = cube.shape

    cube_2D = cube.reshape(nfr,m*n)
    P_2D = P.reshape(nfr,m*n)
    mask_2D = mask.reshape(nfr,m*n)
    
    U = None
    V = None
    rel_error= []
    a_amat = np.zeros(MAX_ITER+1)
    for i in range(MAX_ITER):
        res = cube_2D-a_amat[i]*P_2D

        # Find L(k+1)
        if norm == 2:
            u,s,v = np.linalg.svd(res, full_matrices=False)
            L = np.dot(u[:,:rank]*s[:rank],v[:rank])

        else:
            U, V, rel_error = L1LRAcd(res, r=rank, maxiter=maxiter_l1, 
                                      U0=U, V0=V, rel_error=rel_error)
            L = np.dot(U,V)
            


        # Find a(k+1)
        residuals = cube_2D-L
        a_amat[i+1] = calculate_flux(residuals, P_2D, mask_2D, norm)
        if np.abs(a_amat[i+1]-a_amat[i])/np.abs(a_amat[i+1])<eps:
            break



    # Then obtain residual cube, derotate the cube, and obtain median frame
    res_fin = residuals.reshape(nfr, m, n)
    residuals_cube_der = vip.preproc.cube_derotate(res_fin, angles, imlib='opencv')
    frame_amat = np.nanmedian(residuals_cube_der, axis=0)
    
    return frame_amat, L, residuals_cube_der, a_amat[i+1], i+1


def calculate_flux(res, P, mask, norm):
    """"
    Compute flux (intensity) --> Coefficient a_g of P_g
    """
    if norm == 2:
        res_ = res[mask]
        P_ = P[mask]
        a = np.sum(res_ * P_) / np.sum(P_ ** 2)
    else:
        ais = res[mask] / P[mask]
        ind_ais = np.argsort(ais)
        ais_sorted = ais[ind_ais]
        P_sorted = P[mask][ind_ais]
        slope0 = np.sum(-P_sorted)
        cum_sum_P = np.cumsum(P_sorted)
        slopes = 2*cum_sum_P+slope0
        i_min = np.argwhere(slopes>0)[0]
        a = ais_sorted[i_min]

    return a

