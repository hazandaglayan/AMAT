import numpy as np
import vip_hci as vip

def trajectory_pixels(center, angle_list, rad, theta):
    """Compute the pixel locations in a frame of a trajectory.
    
    A trajectory is defined as the path followed by a fixed point in the
    ADI cube. 
    """


    cy = center[0]
    cx = center[1]

    ys = cy + np.sin(np.deg2rad(-angle_list + theta)) * rad
    xs = cx + np.cos(np.deg2rad(-angle_list + theta)) * rad

    return ys, xs

def trajectory_mask(cube_shape, center, angle_list, rad, theta, fwhm=4, prad=1.):
    """Create a mask along a trajectory.

    Given a trajectory starting point, creates a boolean 3D array equal to True
    for a disk of radius `prad` * FWHM/2 repeated along that path.

    """
    mask = np.zeros(cube_shape, bool)

    pixels = trajectory_pixels(center, angle_list, rad, theta)
    radius = prad * fwhm/2
    r2 = radius*radius

    yy, xx = np.ogrid[:cube_shape[1], :cube_shape[2]]
    for i, (y, x) in enumerate(zip(*pixels)):
        mask[i] = ( (xx - x)**2 + (yy - y)**2 <= r2 )
    return mask

def psf_cube(cube, psfn, angles, rad, theta):
    """Creates a cube filled with zero and a copy of the reference PSF along
    a trajectory."""
    return vip.fm.cube_inject_companions(
        np.zeros_like(cube), psfn, angles,
        1.,  rad, theta=theta, verbose=False
    )

def mask_annulus(shape, center, inner_radius, outer_radius):
    """
    Create a mask of an annulus.

    """
    cy, cx = center

    ys, xs = np.indices(shape)
    return ((ys - cy )**2 + (xs - cx )**2 <= (outer_radius)**2) &\
           ((ys - cy )**2 + (xs - cx )**2 >= inner_radius**2)

def pixels_in_annulus(shape, center, inner_radius, outer_radius):
    
    ys, xs = np.indices(shape)
    mask = mask_annulus(shape, center, inner_radius, outer_radius)
    return ys[mask], xs[mask]
