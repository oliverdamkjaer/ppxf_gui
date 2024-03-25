import numpy as np
import astropy.io.fits as fits
from tqdm import tqdm
from datetime import datetime

C = 299792.458  # speed of light in km/s

def read_cube(file, z, wave_range, snr_range, crop):

    # Reading the cube
    pbar = tqdm(desc=f"Reading MUSE cube...", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")

    hdu = fits.open(file)
    obj_name = hdu[0].header['OBJECT'].replace(' ', '')
    inst_mode = hdu[0].header['HIERARCH ESO INS MODE']
    #date_time = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    obj = f"{obj_name}_{inst_mode}_test"#{date_time}
    hdr = hdu[1].header
    data = hdu[1].data[:, crop[0]-crop[2]:crop[0]+crop[2]+1, crop[1]-crop[2]:crop[1]+crop[2]+1]
    stat = hdu[2].data[:, crop[0]-crop[2]:crop[0]+crop[2]+1, crop[1]-crop[2]:crop[1]+crop[2]+1]
    pbar.update(30)

    # Transform data into 2-dim array of spectra
    npix = data.shape[0]
    spec = data.reshape(npix, -1) # Create array of spectra [npix, nx*ny]
    varspec = stat.reshape(npix, -1) # Create array of variance spectra [npix, nx*ny]
    wave = hdr['CRVAL3'] + hdr['CD3_3']*np.arange(npix) # Create array of wavelength [npix]
    pixelsize = hdr['CD2_2']*3600.0 # Calculate pixelsize from header information
    pbar.update(20)
    
    # De-redshift spectra
    wave = wave/(1+z)

    # Shorten spectra to required wavelength range
    idx_wave = (wave >= wave_range[0]) & (wave <= wave_range[1])
    spec = spec[idx_wave,:][30:-30]
    varspec = varspec[idx_wave,:][30:-30]
    wave = wave[idx_wave][30:-30]

    # Computing the SNR per spaxel
    idx_snr = (wave >= snr_range[0]) & (wave <= snr_range[1])
    signal = np.nanmedian(spec[idx_snr,:], axis=0)
    noise = np.abs(np.nanmedian(np.sqrt(varspec[idx_snr,:]), axis=0))

    snr = signal / noise
    pbar.update(30)

    # Create coordinates centred on the brightest spectrum
    flux = np.nanmean(spec, 0)
    jm = np.nanargmax(flux)
    row, col = map(np.ravel, np.indices(data.shape[-2:]))
    x = (col - col[jm])*pixelsize
    y = (row - row[jm])*pixelsize
    velscale = C*np.diff(np.log(wave[-2:]))
    pbar.update(20)

    # Create dictionary with relevant output
    cube = {'x':x, 'y':y, 'wave':wave, 'spec':spec, 'varspec':varspec, 'snr':snr, 'data':data, 'row':row, 'col':col,\
            'signal':signal, 'noise':noise, 'pixelsize':pixelsize, 'velscale':velscale, 'object':obj}

    pbar.set_description(f"Reading MUSE cube... Successfully read {int(len(snr))} spectra!")
    save_cube(data, pixelsize, velscale, obj)
    print(f"Output saved to: output/{obj}/cube_out.fits")
    pbar.close()

    return cube

def save_cube(data, pixelsize, velscale, obj):
    
    outfits = f"output/{obj}/cube_out.fits"

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Image HDU with PPXF gas bestfit
    cubeDataHDU = fits.ImageHDU(data)
    cubeDataHDU.name = 'CUBE_DATA'

    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, cubeDataHDU])
    HDUList.writeto(outfits, overwrite=True)
    fits.setval(outfits, "PIXSIZE", value=pixelsize)
    fits.setval(outfits, "VELSCALE", value=velscale[0])
    fits.setval(outfits, "OBJECT", value=obj)