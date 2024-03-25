import numpy as np
import astropy.io.fits as fits
import scipy.spatial.distance as dist
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from tqdm import tqdm

def save_vorbin(bin_num_new, ubins, x, y, xbin, ybin, signal, sn, npix, pixelsize, obj, target_sn):
    """
    Save all relevant information about the Voronoi binning to disk. In
    particular, this allows to later match spaxels and their corresponding bins. 
    """
    outfits = f"output/{obj}/vorbin_out.fits"

    # Expand data to spaxel level
    xbin_new = np.zeros(len(x))
    ybin_new = np.zeros(len(x))
    sn_new = np.zeros(len(x))
    npix_new = np.zeros(len(x))
    
    for i in range(len(ubins)):
        idx = np.where(ubins[i] == np.abs(bin_num_new))[0]
        xbin_new[idx] = xbin[i]
        ybin_new[idx] = ybin[i]
        sn_new[idx] = sn[i]
        npix_new[idx] = npix[i]

    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with output data
    cols = []
    cols.append(fits.Column(name='ID', format='J', array=np.arange(len(x))))
    cols.append(fits.Column(name='BIN_ID', format='J', array=bin_num_new))
    cols.append(fits.Column(name='X', format='D', array=x))
    cols.append(fits.Column(name='Y', format='D', array=y))
    cols.append(fits.Column(name='XBIN', format='D', array=xbin_new))
    cols.append(fits.Column(name='YBIN', format='D', array=ybin_new))
    cols.append(fits.Column(name='SIGNAL', format='D', array=signal))
    cols.append(fits.Column(name='SNBIN', format='D', array=sn_new))
    cols.append(fits.Column(name='NSPAX', format='J', array=npix_new))

    vorbinHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    vorbinHDU.name = "VORBIN"

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, vorbinHDU])
    HDUList.writeto(outfits, overwrite=True)
    fits.setval(outfits, "OBJECT", value=obj)
    fits.setval(outfits, "PIXSIZE", value=pixelsize)
    fits.setval(outfits, "TARGETSN", value=target_sn)
    fits.setval(outfits, "NBINS", value=len(ubins))



def calculate_vorbin(cube, target_sn):
    """
    Calculate voronoi binning pattern according to target_sn.
    """
    print('\n----- Spatial binning -----')
    pbar = tqdm(desc=f"Running vorbin with target_sn={target_sn}", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")

    # Read maskfile
    maskfile = f"output/{cube['object']}/mask.fits"
    mask = fits.open(maskfile)[1].data.MASK
    idx_good = np.where(mask == 0)[0]
    idx_bad = np.where(mask == 1)[0]

    # Calculate voronoi bins
    voronoi_output = voronoi_2d_binning(cube['x'][idx_good], cube['y'][idx_good], cube['signal'][idx_good], cube['noise'][idx_good],\
                                        target_sn, pixelsize=cube['pixelsize'], plot=False, quiet=True)
    bin_num, _, _, xbin, ybin, sn, npix, _ = voronoi_output


    # Find the nearest Voronoi bin for the pixels outside the Voronoi region
    bin_num_outside = find_nearest_voronoibin(cube['x'], cube['y'], idx_bad, xbin, ybin)

    # Generate extended bin_num-list: 
    #   Positive bin_num (including zero) indicate the Voronoi bin of the spaxel (for unmasked spaxels)
    #   Negative bin_num indicate the nearest Voronoi bin of the spaxel (for masked spaxels)
    ubins = np.unique(bin_num)
    nbins = len(ubins)
    bin_num_long = np.zeros( len(cube['x']) )
    bin_num_long[:] = np.nan
    bin_num_long[idx_good] = bin_num
    bin_num_long[idx_bad] = -1 * bin_num_outside

    pbar.update(100)
    pbar.close()

    pbar = tqdm(desc=f"Writing: ../output/{cube['object']}/vorbin_out.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_vorbin(bin_num_long, ubins, cube['x'], cube['y'], xbin, ybin, cube['signal'], sn, npix, cube['pixelsize'], cube['object'], target_sn)
    pbar.update(100)
    pbar.close()

def find_nearest_voronoibin(x, y, idx_outside, xbin, ybin):
    """
    This function determines the nearest Voronoi-bin for all spaxels which do
    not satisfy the minimum SNR threshold. 
    """
    x = x[idx_outside]
    y = y[idx_outside]
    pix_coords = np.concatenate((x.reshape((len(x),1)), y.reshape((len(y),1))), axis=1 )
    bin_coords = np.concatenate((xbin.reshape((len(xbin),1)), ybin.reshape((len(ybin),1))), axis=1)

    dists = dist.cdist(pix_coords, bin_coords, 'euclidean') 
    closest = np.argmin(dists, axis=1)

    return(closest)

