import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

from read_data import read_cube
from spatial_masking import generate_spatial_mask
from spatial_binning import calculate_vorbin
from prepare_spectra import prep_spectra
from stellar_kinematics import extract_kinematics
from kinematics_and_emission import kinematics_and_emission
from emission_lines import extract_emission, extract_line_flux
from multiprocessing import cpu_count

file = '/Users/oddam/Downloads/MUSE/agn/NGC3783_MUSE_WFM.fits'
file = '/Users/oddam/Downloads/MUSE/quiescent/HCG91c_2014-08-20_WFM-NOAO.fits'
#file = "C:/Users/oddam/Dropbox/MUSE/quiescent/HCG91c_2014-08-20_WFM-NOAO.fits"
z = 0.00973
z = 0.023983
crop = (156, 158, 50) # NGC3783
crop = (123, 146, 30) # HCG91c

adeg = -1
mdeg = 8
ncomp_gas = [3]
ncomp_gas = [3, 3, 1, 1, 1, 3, 1, 1, 1]

if __name__ == "__main__":

    cube = read_cube(file=file, z=z, wave_range=[4000, 10000], snr_range=[8580, 8630], crop=crop)

    try:
        os.mkdir(f"output/{cube['object']}")
    except:
        pass

    generate_spatial_mask(cube, 1, crop)
    calculate_vorbin(cube, target_sn=30)
    prep_spectra(cube)
    extract_kinematics(cube['object'], ncpu=6, nsims=1)
    #kinematics_and_emission(cube['object'], ncpu=cpu_count(), nsims=30)
    extract_emission(cube['object'], adeg=adeg, mdeg=mdeg, ncomp_gas=ncomp_gas, ncpu=cpu_count())
    extract_line_flux(cube['object'])