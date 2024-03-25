import numpy as np
import astropy.io.fits as fits
from tqdm import tqdm
import ppxf.ppxf_util as util

def prep_spectra(cube):
    """
    This function performs the following tasks: 
     * Apply spatial bins to linear spectra; Save these spectra to disk
     * Log-rebin all spectra, regardless of whether the spaxels are masked or not; Save all spectra to disk
     * Apply spatial bins to log-rebinned spectra; Save these spectra to disk
    """

    print('\n----- Preparing spectra -----')

    # Read maskfile
    maskfile = f"output/{cube['object']}/mask.fits"
    mask = fits.open(maskfile)[1].data.MASK
    idx_good = np.where(mask == 0)[0]

    # Read binning pattern
    voronoi_output = f"output/{cube['object']}/vorbin_out.fits"
    bin_num = fits.open(voronoi_output)[1].data.BIN_ID[idx_good]
    ubins     = np.unique(bin_num)
    nbins     = len(ubins)

    # Apply spatial bins to linear spectra
    pbar = tqdm(desc='Spatially binning linear spectra', total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    bin_spec, bin_espec, bin_flux = spatial_bin(bin_num, cube['spec'][:,idx_good], cube['varspec'][:,idx_good])
    pbar.update(100)
    pbar.close()

    # Save spatially binned linear spectra
    pbar = tqdm(desc=f"Writing: ../output/{cube['object']}/bin_spectra_lin.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_bin_spec(cube['object'], bin_spec, bin_espec, bin_flux, cube['wave'], cube['velscale'], "lin")
    pbar.update(100)
    pbar.close()

    # Log-rebin spectra
    pbar = tqdm(desc='Log-rebinning spectra', total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    lam_range = [np.min(cube['wave']), np.max(cube['wave'])]
    log_spec, log_lam, _ = util.log_rebin(lam_range, cube['spec'], cube['velscale'])
    log_varspec, _, _ = util.log_rebin(lam_range, cube['varspec'], cube['velscale'])
    pbar.update(100)
    pbar.close()

    # Save log-rebinned spectra
    #pbar = tqdm(desc=f"Writing: ../output/{cube['object']}/all_spectra.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    #save_all_spec(cube['object'], log_spec, log_varspec, log_lam, cube['velscale'][0])
    #pbar.update(100)
    #pbar.close()

    # Apply bins to log spectra
    pbar = tqdm(desc='Spatially binning log spectra', total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    bin_log_spec, bin_log_espec, bin_flux = spatial_bin(bin_num, log_spec[:,idx_good], log_varspec[:,idx_good])
    pbar.update(100)
    pbar.close()

    # Save spatially binned log spectra
    pbar = tqdm(desc=f"Writing: ../output/{cube['object']}/bin_spectra_log.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_bin_spec(cube['object'], bin_log_spec, bin_log_espec, bin_flux, log_lam, cube['velscale'], flag="log")
    pbar.update(100)
    pbar.close()



def spatial_bin(bin_num, spec, varspec):
    """ Spectra belonging to the same spatial bin are added. """
    # Read in output from vorbin

    ubins     = np.unique(bin_num)
    nbins     = len(ubins)
    npix      = spec.shape[0]
    bin_data  = np.zeros([npix, nbins])
    bin_error = np.zeros([npix, nbins])
    bin_flux  = np.zeros(nbins)

    for i in range(nbins):
        k = np.where(bin_num == ubins[i])[0]
        #lenbin = len(k)
        #if lenbin == 1:
        #    av_spec     = spec[:,k]
        #    av_espec = np.sqrt(varspec[:,k])
        #else:
        #    av_spec     = np.nansum(spec[:,k], axis=1)
        #    av_espec = np.sqrt(np.nansum(varspec[:,k], axis=1))
        av_spec = np.nansum(spec[:,k], axis=1)
        av_espec = np.sqrt(np.nansum(varspec[:,k], axis=1))
        
        #av_espec[av_espec <= 0] = 1e6
    
        bin_data[:,i] = np.ravel(av_spec)
        bin_error[:,i] = np.ravel(av_espec)
        bin_flux[i] = np.mean(av_spec, axis=0)

    return bin_data, bin_error, bin_flux


def save_bin_spec(obj, bin_data, bin_error, bin_flux, log_lam, velscale, flag):
    """ Spatially binned spectra and error spectra are saved to disk. """
    outdir = f'output/{obj}/'

    if flag == 'log':
        outfits_spectra = outdir + 'bin_spectra_log.fits'
    elif flag == 'lin':
        outfits_spectra  = outdir + 'bin_spectra_lin.fits'

    npix = len(bin_data)

    # Create primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append(fits.Column(name='SPEC', format=str(npix)+'D', array=bin_data.T))
    cols.append(fits.Column(name='ESPEC', format=str(npix)+'D', array=bin_error.T))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BIN_SPECTRA'

    # Table HDU for BINFLUX
    cols = []
    cols.append(fits.Column(name='BINFLUX', format='D', array=bin_flux))
    binfluxHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    binfluxHDU.name = 'BINFLUX'

    # Table HDU for LOGLAM
    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=log_lam))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = 'LOGLAM'

    # Create HDU list and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, binfluxHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header values
    fits.setval(outfits_spectra, 'VELSCALE', value=velscale[0])
    fits.setval(outfits_spectra, 'CRPIX1', value=1.0)
    fits.setval(outfits_spectra, 'CRVAL1', value=log_lam[0])
    fits.setval(outfits_spectra, 'CDELT1', value=log_lam[1]-log_lam[0])



def save_all_spec(obj, spec, varspec, log_lam, velscale):
    """ Save all logarithmically rebinned spectra to file. """
    outfits_spectra = f'output/{obj}/all_spectra.fits'

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append(fits.Column(name='SPEC', format=str(len(spec))+'D', array=spec.T))
    cols.append(fits.Column(name='VARSPEC', format=str(len(spec))+'D', array=varspec.T))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'SPECTRA'

    # Table HDU for LOGLAM
    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=log_lam))
    ln_lamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    ln_lamHDU.name = 'LOGLAM'
    
    # Create HDU List and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, ln_lamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header keywords
    fits.setval(outfits_spectra, 'VELSCALE', value=velscale)
    fits.setval(outfits_spectra, 'CRPIX1', value=1.0)
    fits.setval(outfits_spectra, 'CRVAL1', value=log_lam[0])
    fits.setval(outfits_spectra, 'CDELT1', value=log_lam[1]-log_lam[0])
