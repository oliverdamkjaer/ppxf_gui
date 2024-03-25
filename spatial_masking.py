import os
import numpy as np
from astropy.io import fits

from tqdm import tqdm

def generate_spatial_mask(cube, min_snr, crop):
    """ 
    Default implementation of the spatialMasking module. 

    This function masks defunct spaxels, rejects spaxels with a signal-to-noise ration below a given threshold, and
    masks spaxels according to a provided mask file. Finally, all masks are combined and saved. 
    """

    print('----- Spatial masking -----')

    # Mask defunct spaxels
    pbar = tqdm(desc=f"Masking defunct spaxels", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    maskedDefunct = mask_defunct(cube)
    pbar.update(100)
    pbar.close()

    # Mask spaxels with SNR below threshold
    pbar = tqdm(desc=f"Masking spaxels below SNR={min_snr}", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    maskedSNR = mask_snr(cube['snr'], cube['signal'], min_snr)
    pbar.update(100)
    pbar.close()

    # Mask spaxels according to spatial mask file
    maskedMask = mask_file(cube, crop)

    # Create combined mask
    pbar = tqdm(desc=f"Combining all masks", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    combinedMaskIdx = np.where( np.logical_or.reduce((maskedDefunct == True, maskedSNR == True, maskedMask == True)) )[0] #maskedDefunct == True, 
    combinedMask = np.zeros(len(cube['snr']), dtype=bool)
    combinedMask[combinedMaskIdx] = True
    pbar.update(100)
    pbar.close()
   
    # Save mask to file
    pbar = tqdm(desc=f"Writing: ../output/{cube['object']}/mask.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_spatial_mask(cube['object'], combinedMask, maskedDefunct, maskedSNR, maskedMask)
    pbar.update(100)
    pbar.close()


def mask_defunct(cube):
    """
    Mask defunct spaxels, in particular those containing np.nan's or have a 
    negative median. 
    """
    idx_good = np.where(np.logical_and(np.all(np.isnan(cube['spec']) == False, axis=0), np.nanmedian(cube['spec'], axis=0) > 0.0 ))[0]
    idx_bad = np.where(np.logical_or(np.any(np.isnan(cube['spec']) == True, axis=0), np.nanmedian(cube['spec'], axis=0) <= 0.0 ))[0]

    masked = np.zeros(len(cube['snr']), dtype=bool)
    masked[idx_good] = False
    masked[idx_bad] = True 

    return masked


def mask_snr(snr, signal, min_snr):
    """ 
    Mask those spaxels that are above the isophote level with a mean 
    signal-to-noise ratio of MIN_SNR. 
    """
    idx_snr = np.where( np.abs(snr - min_snr) < 2. )[0]
    meanmin_signal = np.mean( signal[idx_snr] )
    idx_inside  = np.where( signal >= meanmin_signal )[0]
    idx_outside = np.where( signal < meanmin_signal )[0]

    if len(idx_inside) == 0 and len(idx_outside) == 0:
        idx_inside = np.arange( len(snr) )
        idx_outside = np.array([], dtype=np.int64)

    masked = np.zeros(len(snr), dtype=bool)
    masked[idx_inside]  = False
    masked[idx_outside] = True

    return(masked)


def mask_file(cube, crop):
    """
    Select those spaxels that are unmasked in the input masking file.
    """
    short_name = cube['object'].split("_")
    short_name = f"{short_name[0]}_{short_name[1]}"
    print(short_name)
    maskfile = f"input/{short_name}_premask.fits"
    print(maskfile)
    
    if os.path.isfile(maskfile) == True:
        mask = fits.open(maskfile)[0].data[crop[0]-crop[2]:crop[0]+crop[2]+1, crop[1]-crop[2]:crop[1]+crop[2]+1]
        s = np.shape(mask)
        mask = np.reshape(mask, s[0]*s[1])
        
        idxGood = np.where( mask == 0 )[0]
        idxBad  = np.where( mask == 1 )[0]
    
    elif os.path.isfile(maskfile) == False: 
        idxGood = np.arange( len(cube['snr']) )
        idxBad  = np.array([], dtype=np.int64)

    masked = np.zeros(len(cube['snr']), dtype=bool)
    masked[idxGood] = False
    masked[idxBad]  = True

    return(masked)


def save_spatial_mask(obj, combinedMask, maskedDefunct, maskedSNR, maskedMask): 
    """ Save the mask to disk. """
    outfits = f"output/{obj}/mask.fits"

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with output data
    # This is an integer array! 0 means unmasked, 1 means masked!
    cols = []
    cols.append(fits.Column(name='MASK',         format='I', array=np.array(combinedMask, dtype=int)  ))
    cols.append(fits.Column(name='MASK_DEFUNCT', format='I', array=np.array(maskedDefunct, dtype=int) ))
    cols.append(fits.Column(name='MASK_SNR',     format='I', array=np.array(maskedSNR, dtype=int)     ))
    cols.append(fits.Column(name='MASK_FILE',    format='I', array=np.array(maskedMask, dtype=int)    ))
    tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    tbhdu.name = "MASKFILE"

    # Create HDU list and write to file
    tbhdu.header['COMMENT'] = "Value 0  -->  unmasked"
    tbhdu.header['COMMENT'] = "Value 1  -->  masked"
    HDUList = fits.HDUList([priHDU, tbhdu])
    HDUList.writeto(outfits, overwrite=True)
