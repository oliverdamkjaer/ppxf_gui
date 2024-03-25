import astropy.io.fits as fits
import numpy as np
from os import path
from tqdm import tqdm

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from multiprocessing import Process, Queue
from urllib import request

C = 299792.458  # speed of light in km/s

def clip_outliers(galaxy, bestfit, goodpixels, sig_clip):
    """
    Repeat the fit after clipping bins deviant more than 3*sigma
    in relative error until the bad bins don't change any more.
    """
    while True:
        scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
        resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < sig_clip*err)
        if np.array_equal(goodpixels, ok_old):
            break
            
    return goodpixels

def fit_and_clean(templates, log_bin_spec, log_bin_espec, velscale, start, bounds, constr_kinem, goodpixels0, lam_gal, lam_temp, mom, adeg, mdeg, component, gas_component, gas_names, nsims, sig_clip, i):
    
    goodpixels = goodpixels0.copy()
    pp = ppxf(templates, log_bin_spec, log_bin_espec, velscale, start, bounds=bounds, constr_kinem=constr_kinem,
              component=component, gas_component=gas_component, gas_names=gas_names, moments=mom, degree=adeg,
              mdegree=mdeg, lam=lam_gal, lam_temp=lam_temp, goodpixels=goodpixels, plot=False, quiet=True)

    goodpixels = clip_outliers(log_bin_spec, pp.bestfit, goodpixels, sig_clip)

    # Add clipped pixels to the original masked emission line regions and repeat the fit
    goodpixels = np.intersect1d(goodpixels, goodpixels0)

    pp = ppxf(templates, log_bin_spec, log_bin_espec, velscale, start, bounds=bounds, constr_kinem=constr_kinem,
              component=component, gas_component=gas_component, gas_names=gas_names, moments=mom, degree=adeg,
              mdegree=mdeg, lam=lam_gal, lam_temp=lam_temp, goodpixels=goodpixels, plot=False, quiet=True)
    
    optimal_template = templates @ pp.weights
    
    # Correct the formal errors assuming that the fit is good
    formal_error = pp.error[0] * np.sqrt(pp.chi2)
    
    # Array to store (potential) results from MC-Simulations
    mc_dist = np.zeros((nsims, np.abs(mom[0])))

    if nsims != 0:

        for j in range(0, nsims):
            # Add noise to bestfit:
            noisy_bestfit = pp.bestfit + np.random.normal(0, 1, len(log_bin_spec))*log_bin_espec

            mc = ppxf(templates, noisy_bestfit, log_bin_espec, velscale, start, bounds=bounds, constr_kinem=constr_kinem,
                      goodpixels=goodpixels, component=component, gas_component=gas_component, gas_names=gas_names,
                      moments=mom, degree=adeg, mdegree=mdeg, lam=lam_gal, lam_temp=lam_temp, plot=False, quiet=True)
            mc_dist[j,:] = mc.sol[0][:]

        mc_sol = np.median(mc_dist, axis=0)
        mc_err = np.std(mc_dist, axis=0)

    return pp.sol[:], pp.bestfit, pp.gas_bestfit, pp.gas_bestfit_templates, pp.gas_flux[:], pp.gas_flux_error[:], goodpixels, optimal_template, formal_error, mc_dist, mc_sol, mc_err


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for templates, log_bin_spec, log_bin_espec, velscale, start, bounds, constr_kinem, goodpixels0, lam_gal, lam_temp, mom, adeg, mdeg, component, gas_component, gas_names, nsims, sig_clip, i in iter(inQueue.get, 'STOP'):

        sol, bestfit, gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error, goodpixels, optimal_template, formal_error, mc_dist, mc_sol, mc_err = \
          fit_and_clean(templates, log_bin_spec, log_bin_espec, velscale, start, bounds,
                        constr_kinem, goodpixels0, lam_gal, lam_temp, mom, adeg, mdeg,
                        component, gas_component, gas_names, nsims, sig_clip, i)

        outQueue.put((i, sol, bestfit, gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error, goodpixels, optimal_template, formal_error, mc_dist, mc_sol, mc_err))


def kinematics_and_emission(obj, ncpu=8, nsims=10):

    print('\n----- pPXF emission lines -----')
    pbar = tqdm(desc=f"Reading ../output/{obj}/bin_spectra_log.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    # Read data from file
    hdu = fits.open(f'output/{obj}/bin_spectra_log.fits')
    log_bin_spec = np.array(hdu['BIN_SPECTRA'].data['SPEC'].T)
    log_bin_espec = np.array(hdu['BIN_SPECTRA'].data['ESPEC'].T)
    log_lam_gal = np.array(hdu['LOGLAM'].data['LOGLAM'])
    velscale = hdu[0].header['VELSCALE']
    
    lam_gal = np.exp(log_lam_gal)
    npix = log_bin_spec.shape[0]
    nbins = log_bin_spec.shape[1]
    pbar.update(100)
    pbar.close()

    # Prepare templates
    pbar = tqdm(desc=f"Preparing stellar templates", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    ppxf_dir = path.dirname(path.realpath(lib.__file__))
    basename = f"spectra_{'emiles'}_9.0.npz"
    filename = path.join(ppxf_dir, 'sps_models', basename)
    if not path.isfile(filename):
        url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
        request.urlretrieve(url, filename)
    emiles = lib.sps_lib(filename, velscale, fwhm_gal=None, norm_range=[5070, 5950])
    templates, log_lam_temp = emiles.templates, emiles.ln_lam_temp

    reg_dim = templates.shape[1:]
    stars_templates = templates.reshape(templates.shape[0], -1)
    stars_templates /= np.median(stars_templates)
    pbar.update(100)
    pbar.close()

    # Prepare input for ppxf
    lam_range_gal = [np.min(lam_gal), np.max(lam_gal)]
    gas_templates, gas_names, line_wave = util.emission_lines(log_lam_temp, lam_range_gal, 2.62)

    gas_names = [gas_names[0], gas_names[0]+'_b1', gas_names[0]+'_b2']
    line_wave = np.tile(line_wave, 3)
    gas_templates = np.tile(gas_templates, 3)
    nlines = line_wave.size

    stars_gas_templates = np.column_stack([stars_templates, gas_templates])
    component = [0]*stars_templates.shape[1] + [1] + [2] + [3]
    gas_component = np.array(component) > 0

    mom = [2, 2, 2, 2]
    start = [[0, 130], [0, 250], [0, 500], [0, 500]]

    # [[v0, sig0], [v1, sig1], [v2, sig2], [v3, sig3]]
    sig_diff = 200  # minimum dispersion difference in km/s
    A_ineq = np.array([[0, 0, 0, 1, 0, 0, 0, -1],
                       [0, 0, 0, 1, 0, -1, 0, 0]])      # sigma3 - sigma4 < -sigma_diff
    b_ineq = np.array([-sig_diff, -sig_diff])/velscale  # velocities have to be expressed in pixels
    constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

    vlim = lambda x: 0 + x*np.array([-100, 100])
    bounds = [[vlim(2), [20, 300]],      
            [vlim(2), [20, 500]],       # I force the narrow component=1 to lie +/-200 km/s from the stellar velocity
            [vlim(2), [20, 1500]],
            [vlim(2), [20, 1500]]]      # I force the broad component=3 to lie +/-200 km/s from the stellar velocity

    adeg = 8
    mdeg = -1
    goodpixels0 = np.arange(0, lam_gal.size, 1)
    sig_clip = 3

    # Arrays to store result of ppxf
    ppxf_result = np.zeros((nbins, 6))
    ppxf_bestfit = np.zeros((nbins, npix))
    ppxf_gas_bestfit = np.zeros((nbins, npix))
    gas_bestfit_templates = np.zeros((nbins, npix, nlines))
    gas_flux = np.zeros((nbins, nlines))
    gas_flux_error = np.zeros((nbins, nlines))
    ppxf_goodpixels = np.zeros((nbins, npix), dtype=bool)
    optimal_templates = np.zeros((nbins, templates.shape[0]))
    formal_error = np.zeros((nbins, 6))
    mc_dist = np.zeros((nbins, nsims, np.abs(mom[0])))
    mc_sol = np.zeros((nbins, 6))
    mc_err = np.zeros((nbins, 6))


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
        inQueue.put((stars_gas_templates, log_bin_spec[:,i], log_bin_espec[:,i], velscale, start,
                     bounds, constr_kinem, goodpixels0, lam_gal, miles.lam_temp, mom, adeg, mdeg,
                     component, gas_component, gas_names, nsims, sig_clip, i))
    
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
        ppxf_result[i,:np.abs(mom[0])]  = ppxf_tmp[i][1][0]
        ppxf_bestfit[i,:] = ppxf_tmp[i][2]
        ppxf_gas_bestfit[i,:] = ppxf_tmp[i][3]
        gas_bestfit_templates[i, :, :] = ppxf_tmp[i][4]
        gas_flux[i,:] = ppxf_tmp[i][5]
        gas_flux_error[i,:] = ppxf_tmp[i][6]
        ppxf_goodpixels[i,:][ppxf_tmp[i][7]] = True
        optimal_templates[i,:] = ppxf_tmp[i][8]
        formal_error[i,:np.abs(mom[0])] = ppxf_tmp[i][9][0]
        mc_dist[i,:,:] = ppxf_tmp[i][10]
        mc_sol[i,:np.abs(mom[0])] = ppxf_tmp[i][11]
        mc_err[i,:np.abs(mom[0])] = ppxf_tmp[i][12]
        
    # Sort output
    argidx = np.argsort(index)
    ppxf_result = ppxf_result[argidx,:]
    ppxf_bestfit = ppxf_bestfit[argidx,:]
    ppxf_gas_bestfit = ppxf_gas_bestfit[argidx,:]
    gas_bestfit_templates = gas_bestfit_templates[argidx, :, :]
    gas_flux = gas_flux[argidx,:]
    gas_flux_error = gas_flux_error[argidx,:]
    ppxf_goodpixels = ppxf_goodpixels[argidx,:]
    optimal_templates = optimal_templates[argidx,:]
    formal_error = formal_error[argidx,:]
    mc_dist = mc_dist[argidx,:,:]
    mc_sol = mc_sol[argidx,:]
    mc_err = mc_err[argidx,:]
    
    run_config = {"adeg": adeg, "mdeg": mdeg, "start": start, "bounds": bounds, "moments": mom, "sig_clip": sig_clip}

    # Save output from ppxf to disk
    pbar = tqdm(desc=f"Writing: ../output/{obj}/ppxf_output.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_ppxf(obj, ppxf_result, ppxf_bestfit, ppxf_goodpixels, optimal_templates, formal_error,
              mc_dist, mc_sol, mc_err, log_lam_gal, goodpixels0, log_lam_temp, nbins, run_config)
    pbar.update(100)
    pbar.close()

    pbar = tqdm(desc=f"Writing: ../output/{obj}/ppxf_emission.fits", total=100, bar_format="{percentage:3.0f}% |{bar:10}| {desc:<10}")
    save_ppxf_emission(obj, ppxf_bestfit, ppxf_gas_bestfit, gas_bestfit_templates, gas_flux, gas_flux_error,
                       log_lam_gal, log_lam_temp, nbins, gas_names, line_wave)
    
    pbar.update(100)
    pbar.close()

def save_ppxf(obj, ppxf_result, ppxf_bestfit, ppxf_goodpixels, optimal_templates, formal_error,
              mc_dist, mc_sol, mc_err, log_lam_gal, goodpixels0, log_lam_temp, nbins, run_config):
    """ Saves all output to disk. """
    # ========================
    # SAVE output
    outfits_ppxf = f'output/{obj}/ppxf_output.fits'
    
    # Primary HDU
    priHDU = fits.PrimaryHDU()
    
    # Table HDU with PPXF output data
    cols = []
    cols.append(fits.Column(name='V', format='D', array=ppxf_result[:,0]))
    cols.append(fits.Column(name='SIGMA', format='D', array=ppxf_result[:,1]))
    if np.any(ppxf_result[:,2]) != 0: cols.append(fits.Column(name='H3', format='D', array=ppxf_result[:,2]))
    if np.any(ppxf_result[:,3]) != 0: cols.append(fits.Column(name='H4', format='D', array=ppxf_result[:,3]))
    if np.any(ppxf_result[:,4]) != 0: cols.append(fits.Column(name='H5', format='D', array=ppxf_result[:,4]))
    if np.any(ppxf_result[:,5]) != 0: cols.append(fits.Column(name='H6', format='D', array=ppxf_result[:,5]))

    cols.append(fits.Column(name='FORM_ERR_V' ,    format='D', array=formal_error[:,0]))
    cols.append(fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1]))
    if np.any(formal_error[:,2]) != 0: cols.append(fits.Column(name='FORM_ERR_H3', format='D', array=formal_error[:,2]))
    if np.any(formal_error[:,3]) != 0: cols.append(fits.Column(name='FORM_ERR_H4', format='D', array=formal_error[:,3]))
    if np.any(formal_error[:,4]) != 0: cols.append(fits.Column(name='FORM_ERR_H5', format='D', array=formal_error[:,4]))
    if np.any(formal_error[:,5]) != 0: cols.append(fits.Column(name='FORM_ERR_H6', format='D', array=formal_error[:,5]))
    
    cols.append(fits.Column(name='MC_SOL_V', format='D', array=mc_sol[:,0]))
    cols.append(fits.Column(name='MC_SOL_SIGMA', format='D', array=mc_sol[:,1]))
    if np.any(mc_sol[:,2]) != 0: cols.append(fits.Column(name='MC_SOL_H3', format='D', array=mc_sol[:,2]))
    if np.any(mc_sol[:,3]) != 0: cols.append(fits.Column(name='MC_SOL_H4', format='D', array=mc_sol[:,3]))
    if np.any(mc_sol[:,4]) != 0: cols.append(fits.Column(name='MC_SOL_H5', format='D', array=mc_sol[:,4]))
    if np.any(mc_sol[:,5]) != 0: cols.append(fits.Column(name='MC_SOL_H6', format='D', array=mc_sol[:,5]))

    cols.append(fits.Column(name='MC_ERR_V', format='D', array=mc_err[:,0]))
    cols.append(fits.Column(name='MC_ERR_SIGMA', format='D', array=mc_err[:,1]))
    if np.any(mc_err[:,2]) != 0: cols.append(fits.Column(name='MC_ERR_H3', format='D', array=mc_err[:,2]))
    if np.any(mc_err[:,3]) != 0: cols.append(fits.Column(name='MC_ERR_H4', format='D', array=mc_err[:,3]))
    if np.any(mc_err[:,4]) != 0: cols.append(fits.Column(name='MC_ERR_H5', format='D', array=mc_err[:,4]))
    if np.any(mc_err[:,5]) != 0: cols.append(fits.Column(name='MC_ERR_H6', format='D', array=mc_err[:,5]))
    

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'KIN_DATA'

    # Image HDU with arrays of raw MC solutions for every fit parameter in every bin
    if np.any(mc_dist) != 0:
        mc_dist_HDU = fits.ImageHDU(mc_dist)
        mc_dist_HDU.name = 'MC_DIST'

    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name='BESTFIT', format=str(nbins)+'D', array=ppxf_bestfit.T))
    bestfitHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    bestfitHDU.name = 'BESTFIT'

    # Table HDU with PPXF bestfit
    cols = []
    cols.append(fits.Column(name='GOODPIX', format=str(nbins)+'L', array=ppxf_goodpixels.T))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = 'GOODPIX'

    # Table HDU with PPXF goodpixels
    cols = []
    cols.append(fits.Column(name='GOODPIX0', format='L', array=goodpixels0))
    goodpix0HDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpix0HDU.name = 'GOODPIX0'
    
    # Table HDU with PPXF log_lam
    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=log_lam_gal))
    log_lamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    log_lamHDU.name = 'LOGLAM'
    
    # Table HDU with optimal templates
    cols = []
    cols.append(fits.Column(name='OPTIMAL_TEMPLATES', format=str(optimal_templates.shape[1])+'D', array=optimal_templates))
    optimal_tempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    optimal_tempHDU.name = 'OPTIMAL_TEMPLATES'
    
    # Table HDU with log_lam_templates
    cols = []
    cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=log_lam_temp) )
    log_lam_tempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    log_lam_tempHDU.name = 'LOGLAM_TEMPLATE'
    
    # Create HDU list and write to file
    if np.any(mc_dist) != 0:
        HDUList = fits.HDUList([priHDU, dataHDU, mc_dist_HDU, bestfitHDU, goodpixHDU, goodpix0HDU,
                                log_lamHDU, optimal_tempHDU, log_lam_tempHDU])
    else:
        HDUList = fits.HDUList([priHDU, dataHDU, bestfitHDU, goodpixHDU, goodpix0HDU,
                                log_lamHDU, optimal_tempHDU, log_lam_tempHDU])

    HDUList.writeto(outfits_ppxf, overwrite=True)
    fits.setval(outfits_ppxf, "ADEG", value=run_config['adeg'])
    fits.setval(outfits_ppxf, "MDEG", value=run_config['mdeg'])
    fits.setval(outfits_ppxf, "START", value=str(run_config['start']))
    fits.setval(outfits_ppxf, "BOUNDS", value=str(run_config['bounds']))
    fits.setval(outfits_ppxf, "MOM", value=str(run_config['moments']))
    fits.setval(outfits_ppxf, "SIGCLIP", value=run_config['sig_clip'])

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

    # Image HDU with PPXF gas bestfit templates
    gas_bestfit_tempHDU = fits.ImageHDU(gas_bestfit_templates)
    gas_bestfit_tempHDU.name = 'GASBESTFIT_TEMPLATES'

    # Table HDU with PPXF log_lam
    cols = []
    cols.append(fits.Column(name='LOGLAM', format='D', array=log_lam_gal))
    log_lamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    log_lamHDU.name = 'LOGLAM'
    
    # Table HDU with log_lam_templates
    cols = []
    cols.append( fits.Column(name='LOGLAM_TEMPLATE', format='D', array=log_lam_temp) )
    log_lam_tempHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    log_lam_tempHDU.name = 'LOGLAM_TEMPLATE'

    # Table HDU with emission line information
    cols = []
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