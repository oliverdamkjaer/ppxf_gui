import astropy.io.fits as fits
import numpy as np
from os import path
from tqdm import tqdm

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
from multiprocessing import Process, Queue

C = 299792.458  # speed of light in km/s

def ppxf_emission_fit(templates, log_bin_spec, log_bin_espec, velscale, start, mom, adeg, mdeg, component, gas_component, gas_names, lam_gal, lam_temp, bounds, constr_kinem, i):
    
    pp = ppxf(templates, log_bin_spec, log_bin_espec, velscale, start,
              moments=mom, degree=adeg, mdegree=mdeg, lam=lam_gal, lam_temp=lam_temp,
              component=component, gas_component=gas_component, gas_names=gas_names, 
              bounds=bounds, constr_kinem=constr_kinem, plot=False, quiet=True)

    return pp.bestfit, pp.gas_bestfit, pp.gas_bestfit_templates, pp.gas_flux[:], pp.gas_flux_error[:]


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for templates, log_bin_spec, log_bin_espec, velscale, start, mom, adeg, mdeg, component, gas_component, gas_names, lam_gal, lam_temp, bounds, constr_kinem, i in iter(inQueue.get, 'STOP'):

        bestfit, gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error = \
          ppxf_emission_fit(templates, log_bin_spec, log_bin_espec, velscale, start, mom, adeg, mdeg, component, gas_component, gas_names, lam_gal, lam_temp, bounds, constr_kinem, i)

        outQueue.put((i, bestfit, gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error))


def extract_emission(obj, adeg, mdeg, ncomp_gas, ncpu=8):

    print('\n----- pPXF emission lines -----')
    pbar = tqdm(desc=f"Reading ../output/{obj}/bin_spectra_log.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    # Read data from file
    hdu = fits.open(f'output/{obj}/bin_spectra_log.fits')
    log_bin_spec = np.array(hdu['BIN_SPECTRA'].data['SPEC'].T)
    log_bin_espec  = np.array(hdu['BIN_SPECTRA'].data['ESPEC'].T)
    log_lam_gal   = np.array(hdu['LOGLAM'].data['LOGLAM'])
    velscale = hdu[0].header['VELSCALE']
    
    lam_gal = np.exp(log_lam_gal)
    npix = log_bin_spec.shape[0]
    nbins = log_bin_spec.shape[1]
    pbar.update(100)
    pbar.close()
    
    # Reading optimal stellar templates and starting values for (V, sig)
    hdu = fits.open(f'output/{obj}/ppxf_output.fits')
    
    try:
        velbin0, sigbin0 = hdu['KIN_DATA'].data['MC_SOL_V'], hdu['KIN_DATA'].data['MC_SOL_SIGMA']
    except:
        velbin0, sigbin0 = hdu['KIN_DATA'].data['V'], hdu['KIN_DATA'].data['SIGMA']
        pass

    stars_templates = hdu['OPTIMAL_TEMPLATES'].data['OPTIMAL_TEMPLATES'].T
    log_lam_temp = hdu['LOGLAM_TEMPLATE'].data['LOGLAM_TEMPLATE']
    lam_temp = np.exp(log_lam_temp)

    # Prepare input for ppxf
    lam_range_gal = [np.min(lam_gal), np.max(lam_gal)]
    gas_templates, gas_names, line_wave = util.emission_lines(log_lam_temp, lam_range_gal, 2.62)

    #ncomp_gas = [3]
    
    repeated_templates = []
    repeated_lines = []
    for i in range(len(gas_names)):
        repeated_lines.append(np.repeat(line_wave[i], ncomp_gas[i]))
        repeated_templates.append([gas_templates[:,i]]*ncomp_gas[i])

    line_wave = np.hstack(repeated_lines)
    nlines = line_wave.size
    gas_templates = np.vstack(repeated_templates).T

    new_names = []
    component0 = []
    for i, name in enumerate(gas_names):
        if ncomp_gas[i] == 1:
            new_names.append(f"{name}_1")
            component0.append(1)
        else:
            for j in range(ncomp_gas[i]):
                new_names.append(f"{name}_{j+1}")
                component0.append(j+1)

    gas_names = np.array(new_names)

    component = [0]*1 + component0
    gas_component = np.array(component) > 0

    stars_gas_templates = np.column_stack([stars_templates, gas_templates])

    mom = [-2, 2, 2, 2]
    # [[v0, sig0], [v1, sig1], [v2, sig2], [v3, sig3]]
    sig_diff = 200  # minimum dispersion difference in km/s
    A_ineq = np.array([[0, 0, 0, 1, 0, 0, 0, -1],
                       [0, 0, 0, 1, 0, -1, 0, 0]])      # sigma3 - sigma4 < -sigma_diff
    b_ineq = np.array([-sig_diff, -sig_diff])/velscale  # velocities have to be expressed in pixels
    constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

    vlim = lambda x: 0 + x*np.array([-100, 100])
    bounds = [[vlim(2), [20, 100]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
            [vlim(2), [20, 100]],       # I force the narrow component=1 to lie +/-200 km/s from the stellar velocity
            [vlim(3), [20, 2500]],
            [vlim(3), [20, 2500]]]      # I force the broad component=3 to lie +/-200 km/s from the stellar velocity

    adeg = -1
    mdeg = 8
    
    # Arrays to store result of ppxf    
    ppxf_bestfit = np.zeros((nbins, npix))
    ppxf_gas_bestfit = np.zeros((nbins, npix))
    gas_bestfit_templates = np.zeros((nbins, npix, nlines))
    gas_flux = np.zeros((nbins, nlines))
    gas_flux_error = np.zeros((nbins, nlines))

    # ====================
    # Run PPXF

    # Create Queues
    inQueue  = Queue()
    outQueue = Queue()
    
    # Create worker processes
    ps = [Process(target=workerPPXF, args=(inQueue, outQueue)) for _ in range(ncpu)]
    
    # Start worker processes
    for p in ps: p.start()

    # Fill the queue
    for i in range(nbins):
        start = [[velbin0[i], sigbin0[i]], [velbin0[i], 50], [velbin0[i], 500], [velbin0[i], 500]]
        stars_gas_templates = np.column_stack([stars_templates[:,i], gas_templates])
        inQueue.put((stars_gas_templates, log_bin_spec[:,i], log_bin_espec[:,i], velscale, start, mom, adeg, mdeg, component, gas_component, gas_names, lam_gal, lam_temp, bounds, constr_kinem, i))
    
    # now get the output with indices
    ppxf_tmp = [outQueue.get() for _ in tqdm(range(nbins), desc=f"Running pPXF with {ncpu} cores", total=nbins, bar_format="{percentage:3.0f}% |{bar:10}{r_bar} | {desc:<10}")]
    
    # send stop signal to stop iteration
    for _ in range(ncpu): inQueue.put('STOP')

    # stop processes
    for p in ps: p.join()

    # Get output
    index = np.zeros(nbins)
    for i in range(0, nbins):
        index[i] = ppxf_tmp[i][0]
        ppxf_bestfit[i,:] = ppxf_tmp[i][1]
        ppxf_gas_bestfit[i,:] = ppxf_tmp[i][2]
        gas_bestfit_templates[i, :, :] = ppxf_tmp[i][3]
        gas_flux[i,:] = ppxf_tmp[i][4]
        gas_flux_error[i,:] = ppxf_tmp[i][5]
        
    # Sort output
    argidx = np.argsort(index)
    ppxf_bestfit = ppxf_bestfit[argidx,:]
    ppxf_gas_bestfit = ppxf_gas_bestfit[argidx,:]
    gas_bestfit_templates = gas_bestfit_templates[argidx, :, :]
    gas_flux = gas_flux[argidx,:]
    gas_flux_error = gas_flux_error[argidx,:]
    

    # Save output from ppxf to disk
    pbar = tqdm(desc=f"Writing: ../output/{obj}/ppxf_emission.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_ppxf_emission(obj, ppxf_bestfit, ppxf_gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error, log_lam_gal, log_lam_temp, nbins, gas_names, line_wave)
    pbar.update(100)
    pbar.close()


def save_ppxf_emission(obj, ppxf_bestfit, ppxf_gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error, log_lam_gal, log_lam_temp, nbins, gas_names, line_wave):
    """ Saves all output to disk. """
    # ========================
    # SAVE output
    outfits_ppxf = f'output/{obj}/ppxf_emission.fits'
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name='BESTFIT', format=str(nbins)+'D', array=ppxf_bestfit.T))
    bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    bestfitHDU.name = 'BESTFIT'

    # Table HDU with PPXF gas bestfit
    cols = []
    cols.append(fits.Column(name='GASBESTFIT', format=str(nbins)+'D', array=ppxf_gas_bestfit.T))
    gas_bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gas_bestfitHDU.name = 'GASBESTFIT'

    # Image HDU with PPXF gas bestfit
    gas_bestfit_tempHDU = fits.ImageHDU(gas_bestfit_templates)
    gas_bestfit_tempHDU.name = 'GASBESTFIT_TEMPLATES'

    # Table HDU with PPXF log_lam_gal
    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=log_lam_gal))
    log_lamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    log_lamHDU.name = 'LOGLAM'
    
    # Table HDU with log_lam_templates
    cols = []
    cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=log_lam_temp))
    log_lam_tempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    log_lam_tempHDU.name = 'LOGLAM_TEMPLATE'

    # Table HDU with emission line information
    cols = []
    gas_names = [name.replace("[", "").replace("]", "_") for name in gas_names]
    cols.append(fits.Column(name='GAS_NAMES', format='15A', array=gas_names))
    cols.append(fits.Column(name='LINE_WAVE', format='D', array=line_wave))
    gas_flux_infoHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gas_flux_infoHDU.name = 'GAS_FLUX_INFO'

    # Table HDU with PPXF gas_flux
    cols = []
    for i, lname in enumerate(gas_names):
        cols.append(fits.Column(name=lname, format='D', array=gas_flux[:,i]))
    gas_fluxHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gas_fluxHDU.name = 'GAS_FLUX'

    # Table HDU with PPXF gas_flux
    cols = []
    for i, lname in enumerate(gas_names):
        cols.append(fits.Column(name=lname, format='D', array=gas_flux_error[:,i].T))
    gas_flux_errorHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gas_flux_errorHDU.name = 'GAS_FLUX_ERR'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, bestfitHDU, gas_bestfitHDU, gas_bestfit_tempHDU, log_lamHDU, log_lam_tempHDU, gas_flux_infoHDU, gas_fluxHDU, gas_flux_errorHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)


def extract_line_flux(obj):
    """
    This function reads the output from the ppxf emission line fit and converts the gas fluxes
    from ergs/(cm^2 s Å) to ergs/(cm^2 s) and outputs everything into a neat structure. It does NOT
    split the flux from the doublets into their individual flux contributions. This needs to happen
    afterwards.
    """

    ppxf_emission_hdu = fits.open(f'output/{obj}/ppxf_emission.fits')
    gas_names = ppxf_emission_hdu['GAS_FLUX_INFO'].data['GAS_NAMES']
    line_wave = ppxf_emission_hdu['GAS_FLUX_INFO'].data['LINE_WAVE']
    bestfit = ppxf_emission_hdu['BESTFIT'].data['BESTFIT'].T
    gas_bestfit_templates = ppxf_emission_hdu['GASBESTFIT_TEMPLATES'].data
    nlines = line_wave.size

    spec_hdu = fits.open(f'output/{obj}/bin_spectra_log.fits')
    log_bin_spec = spec_hdu['BIN_SPECTRA'].data['SPEC']
    velscale = spec_hdu[0].header['VELSCALE']
    nbins = log_bin_spec.shape[0]

    gas_flux = np.zeros((nbins, nlines))
    gas_flux_error = np.zeros((nbins, nlines))

    for i in range(nlines):
        gas_flux[:, i] = ppxf_emission_hdu['GAS_FLUX'].data.field(i)
        gas_flux_error[:, i] = ppxf_emission_hdu['GAS_FLUX_ERR'].data.field(i)
        

    an_arr = np.zeros((nbins, nlines))
    flux_arr = np.zeros((nbins, nlines))

    for j in tqdm(range(nbins), desc=f"Extracting line fluxes", total=nbins, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}"):
        rms = robust_sigma(log_bin_spec[j,:] - bestfit[j,:], zero=1)
        dlam = line_wave*velscale/C  # Angstrom per pixel at line wavelength (dlam/lam = dv/c)
        flux = (gas_flux[j, :]*dlam)  # Convert to ergs/(cm^2 s)
        an = np.max(gas_bestfit_templates[j, :, :], axis=0)/rms

        flux_arr[j, :] = flux
        an_arr[j, :] = an
    
    pbar = tqdm(desc=f"Writing: ../output/{obj}/line_flux.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_line_flux(obj, flux_arr, an_arr, gas_names)
    pbar.update(100)
    pbar.close()

def save_line_flux(obj, line_flux, line_an, gas_names):
    """ Saves all output to disk. """
    # ========================
    # SAVE output
    outfits_ppxf = f'output/{obj}/line_flux.fits'
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with PPXF gas_flux
    cols = []
    for i, lname in enumerate(gas_names):
        lname = lname.replace("[", "").replace("]", "_")
        cols.append(fits.Column(name=lname, format='D', array=line_flux[:,i]))
    line_fluxHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    line_fluxHDU.name = 'LINE_FLUX'

    # Table HDU with PPXF gas_flux
    cols = []
    for i, lname in enumerate(gas_names):
        lname = lname.replace("[", "").replace("]", "_")
        cols.append(fits.Column(name=lname, format='D', array=line_an[:,i].T))
    line_anHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    line_anHDU.name = 'LINE_AN'
    
    # Create HDU list and write to file
    HDUList = fits.HDUList([priHDU, line_fluxHDU, line_anHDU])
    HDUList.writeto(outfits_ppxf, overwrite=True)