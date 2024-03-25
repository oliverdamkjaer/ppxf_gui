# Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import astropy.io.fits as fits
import os

# Third party imports
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QCheckBox, QGroupBox
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import QMainWindow, QWidget, QFileDialog, QStatusBar, QLabel
from PySide6 import QtGui

class ViewerTab(QWidget):
    def __init__(self):
        super().__init__()

        # self.dir_path = QFileDialog.getExistingDirectory(self, "Open Directory",
        # "/Users/oddam/Dropbox/Universitet/ppxf_viewer/output",
        # QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        self.dir_path = "/Users/oddam/Dropbox/Universitet/ppxf_viewer/output/HCG91c_WFM-NOAO-N_test"

        self.load_data()
        self.create_figures()
        self.create_layout()

    def create_button(self, label, callback):
            button = QPushButton(label)
            button.clicked.connect(callback)
            return button
    
    def create_checkbox(self, label, callback, checked=True):
            checkbox = QCheckBox(label)
            checkbox.toggled.connect(callback)
            checkbox.setChecked(checked)
            return checkbox
    
    def add_widgets_to_layout(self, layout, widgets):
            for widget in widgets:
                layout.addWidget(widget)

    def create_group_box(self, layout):
        group_box = QGroupBox()
        group_box.setLayout(layout)
        return group_box

    def create_layout(self):

        # Create the buttons above the map plot
        flux_button = self.create_button("Flux", self.flux_button_clicked)
        vel_button = self.create_button("V", self.vel_button_clicked)
        vel_err_button = self.create_button("V error", self.vel_err_button_clicked)
        sig_button = self.create_button("Sigma", self.sig_button_clicked)
        sig_err_button = self.create_button("Sigma error", self.sig_err_button_clicked)
        original_button = self.create_button("Original", self.original_button_clicked)

        # Create horizontal layout for the buttons
        buttons = [flux_button, vel_button, vel_err_button, sig_button, sig_err_button, original_button]
        button_layout = QHBoxLayout()
        self.add_widgets_to_layout(button_layout, buttons)

        # Create checkboxes for the fit plot
        checkbox_data = self.create_checkbox("Data", self.checkbox_data_clicked)
        checkbox_kin_bestfit = self.create_checkbox("Kin bestfit", self.checkbox_kin_bestfit_clicked)
        checkbox_emission_bestfit = self.create_checkbox("Emission bestfit", self.checkbox_emission_bestfit_clicked)

        # Create vertical layout for the checkboxes
        checkboxes = [checkbox_data, checkbox_kin_bestfit, checkbox_emission_bestfit]
        checkbox_layout = QVBoxLayout()
        self.add_widgets_to_layout(checkbox_layout, checkboxes)
        
        # Group buttons, toolbar and map in a single box
        map_layout = QVBoxLayout()
        map_layout.addLayout(button_layout, 1)
        map_layout.addWidget(NavigationToolbar(self.canvas_map, self), 1)
        map_layout.addWidget(self.canvas_map, 30)
        map_group_box = self.create_group_box(map_layout)

        fit_layout = QVBoxLayout()
        fit_layout.addWidget(NavigationToolbar(self.canvas_fit, self))
        fit_layout.addWidget(self.canvas_fit)
        fit_group_box = self.create_group_box(fit_layout)

        mc_layout = QVBoxLayout()
        mc_layout.addWidget(self.canvas_mc)
        mc_group_box = self.create_group_box(mc_layout)

        # Define the left layout consisting of the map_group_box and mc_group_box
        left_layout = QVBoxLayout()
        left_layout.addWidget(map_group_box, 2)
        left_layout.addWidget(mc_group_box, 1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addWidget(fit_group_box, 2)
        main_layout.addLayout(checkbox_layout)

        self.setLayout(main_layout)

    def create_figures(self):

        # Create matplotlib canvas widgets for all plots
        self.fig_map = Figure(layout="constrained")
        self.fig_map.patch.set_facecolor("None")
        self.canvas_map = FigureCanvas(self.fig_map)
        self.canvas_map.setStyleSheet("background-color:transparent;")
        
        self.fig_fit = Figure(layout="constrained")
        self.fig_fit.patch.set_facecolor("None")
        self.canvas_fit = FigureCanvas(self.fig_fit)
        self.canvas_fit.setStyleSheet("background-color:transparent;")

        self.fig_mc = Figure(layout="constrained")
        self.fig_mc.patch.set_facecolor("None")
        self.canvas_mc = FigureCanvas(self.fig_mc)
        self.canvas_mc.setStyleSheet("background-color:transparent;")

        # Axes for the map plot
        self.ax_map = self.canvas_map.figure.subplots()
        self.ax_map.set_xlabel('Offset (arcsec)')
        self.ax_map.set_ylabel('Offset (arcsec)')

        self.img = self.binshow(self.x, self.y, self.bin_num_long, np.log(self.signal_bin), label='Flux')
        self.ax_map.set_xlabel('Offset (arcsec)')
        self.ax_map.set_ylabel('Offset (arcsec)')
        self.canvas_map.mpl_connect('button_press_event', self.onclick)

        # Axes for the fit plot and residuals
        self.ax_fit = self.canvas_fit.figure.subplots(nrows=2, sharex=True, height_ratios=[2,1])
        self.ax_fit[0].get_xaxis().set_visible(False)
        self.ax_fit[0].set_xlim(min(self.lam_gal), max(self.lam_gal))
        self.ax_fit[0].set_ylabel(r'Flux (erg/s/cm$^2$/Å)')

        self.line_data, = self.ax_fit[0].step([], [], color='k', where='mid', lw=2)
        self.line_emission_bestfit, = self.ax_fit[0].plot([], [], color='g', lw=2)
        self.line_kin_bestfit, = self.ax_fit[0].plot([], [], color='r', lw=2)
        
        self.line_res, = self.ax_fit[1].plot([], [], color='C0', lw=2)

        self.ax_fit[1].axhline(3, color='k', ls='--', label=r'$\pm$3 $\sigma$')
        self.ax_fit[1].axhline(-3, color='k', ls='--')
        self.ax_fit[1].axhline(0, color='k')
        self.ax_fit[1].set_xlim(min(self.lam_gal), max(self.lam_gal))
        self.ax_fit[1].set_xlabel('Wavelength (Å)')
        self.ax_fit[1].set_ylabel('Residuals ($\sigma$)')
        self.ax_fit[1].legend(loc='upper left')

        # Keep y-label coordinates of ax1 and ax2 frozen
        self.ax_fit[0].yaxis.set_label_coords(-0.1, 0.5)
        self.ax_fit[1].yaxis.set_label_coords(-0.1, 0.5)

        self.ax_mc = self.canvas_mc.figure.subplots(ncols=2)
        self.canvas_map.draw()

    def onclick(self, event):
        if event.button == 3:
            self.mx, self.my = event.xdata, event.ydata
            self.get_voronoi_bin()

    def get_voronoi_bin(self):
        idx = np.where(np.logical_and(np.abs(self.x-self.mx) < self.pixelsize, np.abs(self.y-self.my) < self.pixelsize))[0]
        if len(idx) == 1:
            final_idx = idx[0]
        elif len(idx) == 4:
            xmini = np.argsort(np.abs(self.x[idx]-self.mx))[:2]
            ymini = np.argmin(np.abs(self.y[idx[xmini]]-self.my))
            final_idx = idx[xmini[ymini]]
        else:
            return None

        self.idx_bin = np.abs(self.bin_num_long[final_idx])
        self.plot_data()


    def plot_data(self):
        # Return all the indices that belong to the same bin
        all_idx_bin = np.where(np.abs(self.bin_num_long) == self.idx_bin)[0]

        # Remove all collections from the Axes object
        for collection in self.ax_map.collections:
            collection.remove()

        # Plot solid rectangle patches at all the indices that belong to the same bin
        rectangles = np.empty(len(all_idx_bin), object)

        for j, i in enumerate(all_idx_bin):
            rectangles[j] = (Rectangle((self.x[i]-self.pixelsize*0.5, self.y[i]-self.pixelsize*0.5),
                            self.pixelsize, self.pixelsize))

        collection = PatchCollection(rectangles, hatch="////////", facecolor='None', lw=0)
        self.ax_map.add_collection(collection)

        ##############################
        self.line_data.set_data(self.lam_gal, self.spec[self.idx_bin,:])

        ymax = np.max(self.emission_bestfit[self.idx_bin,:])
        ymin = np.min(self.spec[self.idx_bin,:])
        self.line_emission_bestfit.set_data(self.lam_gal, self.emission_bestfit[self.idx_bin,:])
        temp_maxy = np.min(self.gas_bestfit_templates[self.idx_bin,:,:])
        dist = np.abs(ymin-temp_maxy)*0.8

        self.stars_bestfit = self.emission_bestfit[self.idx_bin,:] - self.gas_bestfit[:,self.idx_bin]

        self.ax_fit[0].set_ylim(0.8*dist, 1.2*ymax)

        res = (self.spec[self.idx_bin,:] - self.emission_bestfit[self.idx_bin,:])/self.espec[self.idx_bin,:]
        self.line_kin_bestfit.set_data(self.lam_gal, self.kin_bestfit[self.idx_bin,:])
        
        self.line_res.set_data(self.lam_gal, res)
        
        self.ax_fit[0].set_title(f'BIN: {self.idx_bin}')

        max_res = np.max(np.abs(res))
        self.ax_fit[1].set_ylim(-1.2*max_res, 1.2*max_res)
        
        # Try to plot the mc distributions
        self.ax_mc[0].cla()
        self.ax_mc[1].cla()

        self.ax_mc[0].hist(self.mc_output[self.idx_bin,:,0], bins=20, edgecolor='k', linewidth=1, alpha=0.6)
        self.ax_mc[0].set_xlabel("V (km/s)")
        self.ax_mc[0].set_ylabel("Counts")
        self.ax_mc[0].set_title(f'V = {self.mc_sol_vel[self.idx_bin]:.1f} $\pm$ {self.mc_err_vel[self.idx_bin]:.1f} km/s', fontsize=8)
        
        self.ax_mc[1].hist(self.mc_output[self.idx_bin,:,1], bins=20, edgecolor='k', linewidth=1, alpha=0.6)
        self.ax_mc[1].set_xlabel("Sigma (km/s)")
        self.ax_mc[1].set_ylabel("Counts")
        self.ax_mc[1].set_title(f'Sigma = {self.mc_sol_sig[self.idx_bin]:.1f} $\pm$ {self.mc_err_sig[self.idx_bin]:.1f} km/s', fontsize=8)

        # Update the canvases with all the changes
        self.canvas_map.draw()
        self.canvas_fit.draw()
        self.canvas_mc.draw()

    def checkbox_data_clicked(self, checked):
        if checked:
            self.line_data.set_visible(True)
        else:
            self.line_data.set_visible(False)
        self.canvas_fit.draw()

    def checkbox_kin_bestfit_clicked(self, checked):   
        if checked:
            self.line_kin_bestfit.set_visible(True)
        else:
            self.line_kin_bestfit.set_visible(False)
        self.canvas_fit.draw()

    def checkbox_emission_bestfit_clicked(self, checked):
        if checked:
            self.line_emission_bestfit.set_visible(True)
        else:
            self.line_emission_bestfit.set_visible(False)
        self.canvas_fit.draw()

    def flux_button_clicked(self):
        self.cbar.remove()
        self.ax_map.cla()
        self.binshow(self.x, self.y, self.bin_num_long, np.log(self.signal_bin), label="Flux")
        self.canvas_map.draw()

    def vel_button_clicked(self):
        self.cbar.remove()
        self.ax_map.cla()
        self.binshow(self.x, self.y, self.bin_num_long, self.mc_sol_vel, label=r'$V$ (km/s)')
        self.canvas_map.draw()

    def vel_err_button_clicked(self):
        self.cbar.remove()
        self.ax_map.cla()
        self.binshow(self.x, self.y, self.bin_num_long, self.mc_err_vel, label=r'$\Delta V$ (km/s)')
        self.canvas_map.draw()

    def sig_button_clicked(self):
        self.cbar.remove()
        self.ax_map.cla()
        self.binshow(self.x, self.y, self.bin_num_long, self.mc_sol_sig, label=r'$\sigma_*$')
        self.canvas_map.draw()

    def sig_err_button_clicked(self):
        self.cbar.remove()
        self.ax_map.cla()
        self.binshow(self.x, self.y, self.bin_num_long, self.mc_err_sig, label=r'$\Delta\sigma_*$')
        self.canvas_map.draw()

    def original_button_clicked(self):
        self.cbar.remove()
        self.ax_map.cla()

        xmin, xmax = np.min(self.x), np.max(self.x)
        ymin, ymax = np.min(self.y), np.max(self.y)
        extent = [xmin-self.pixelsize/2, xmax+self.pixelsize/2, ymin-self.pixelsize/2, ymax+self.pixelsize/2]
        img = self.ax_map.imshow(np.log(np.nanmean(self.cube_data, axis=0)), origin='upper', interpolation='nearest', extent=extent, cmap='RdBu_r')
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar = plt.colorbar(img, cax=cax)
        self.cbar.set_label("log(Flux)")
        self.ax_map.set_xlabel("Offset (arcsec)")
        self.ax_map.set_ylabel("Offset (arcsec)")
        self.ax_map.minorticks_on()
        self.ax_map.tick_params(length=10, width=1, which='major')
        self.ax_map.tick_params(length=5, width=1, which='minor')

        self.canvas_map.draw()

    def binshow(self, x, y, bin_num, val, label, **kwargs):

        x, y, val = map(np.ravel, [x, y, val[bin_num]])

        vmin, vmax = np.percentile(val, [1, 99])

        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        x1 = (x - xmin)/self.pixelsize
        y1 = (y - ymin)/self.pixelsize
        nx = int(round((xmax - xmin)/self.pixelsize) + 1)
        ny = int(round((ymax - ymin)/self.pixelsize) + 1)
        mask = np.ones((nx, ny), dtype=bool)
        img0 = np.empty((nx, ny))
        j = np.round(x1).astype(int)
        k = np.round(y1).astype(int)

        mask[j, k] = 0
        img0[j, k] = val
        img0 = np.ma.masked_array(img0, mask)

        extent = [xmin-self.pixelsize/2, xmax+self.pixelsize/2, ymin-self.pixelsize/2, ymax+self.pixelsize/2]
        self.img = self.ax_map.imshow(np.rot90(img0), interpolation='nearest',
                        origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        extent=extent, **kwargs)
        
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar = plt.colorbar(self.img, cax=cax)
        self.cbar.set_label(label)
        
        
        self.ax_map.set_xlabel("Offset (arcsec)")
        self.ax_map.set_ylabel("Offset (arcsec)")

        self.ax_map.minorticks_on()
        self.ax_map.tick_params(length=10, width=1, which='major')
        self.ax_map.tick_params(length=5, width=1, which='minor')

    def load_data(self):

        self.kinematics = os.path.isfile(f"{self.dir_path}/ppxf_output.fits")
        self.emission = os.path.isfile(f"{self.dir_path}/ppxf_emission.fits")
        self.cube_data = fits.open(f"{self.dir_path}/cube_out.fits")[1].data
        self.cube_data = np.rot90(self.cube_data, k=-2)

        # Load vorbin output
        self.vorbin_out = fits.open(f"{self.dir_path}/vorbin_out.fits")
        self.idx_inside = np.where(self.vorbin_out['VORBIN'].data['BIN_ID'] >= 0)[0]
        self.x = np.array(self.vorbin_out['VORBIN'].data['X'][self.idx_inside])
        self.y = np.array(self.vorbin_out['VORBIN'].data['Y'][self.idx_inside])
        self.signal = np.array(self.vorbin_out['VORBIN'].data['SIGNAL'][self.idx_inside])
        self.bin_num_long = np.array(self.vorbin_out['VORBIN'].data['BIN_ID'][self.idx_inside])
        self.ubins = np.unique(np.abs(np.array(self.vorbin_out['VORBIN'].data['BIN_ID'])))
        self.pixelsize = self.vorbin_out[0].header['PIXSIZE']
        try:
            self.target_sn = self.vorbin_out[0].header['TARGETSN']
            self.nbins = self.vorbin_out[0].header['NBINS']
        except:
            pass

        # Load spectra
        self.binspec_hdu = fits.open(f"{self.dir_path}/bin_spectra_log.fits")
        self.spec = self.binspec_hdu['BIN_SPECTRA'].data['SPEC']
        self.espec = self.binspec_hdu['BIN_SPECTRA'].data['ESPEC']
        self.flux = np.array(self.binspec_hdu['BINFLUX'].data['BINFLUX'])
        self.loglam = self.binspec_hdu['LOGLAM'].data['LOGLAM']
        self.lam_gal = np.exp(self.loglam)

        # Load pPXF output
        if self.kinematics:
            print("Kinematic data found")
            self.ppxf_hdu = fits.open(f"{self.dir_path}/ppxf_output.fits")
            self.kin_bestfit = self.ppxf_hdu['BESTFIT'].data['BESTFIT'].T
            self.kin_apoly = self.ppxf_hdu['APOLY'].data['APOLY'].T
            self.kin_mpoly = self.ppxf_hdu['MPOLY'].data['MPOLY'].T
            self.goodpixels = self.ppxf_hdu['GOODPIX'].data['GOODPIX']
            self.velbin = self.ppxf_hdu['KIN_DATA'].data['V']
            self.sigbin = self.ppxf_hdu['KIN_DATA'].data['SIGMA']
            self.adeg = self.ppxf_hdu[0].header['ADEG']
            self.mdeg = self.ppxf_hdu[0].header['MDEG']
            self.sig_clip = self.ppxf_hdu[0].header['SIGCLIP']
            self.start = self.ppxf_hdu[0].header['START']

            # Load MC data if available
            try:
                self.mc_output = self.ppxf_hdu['MC_DIST'].data
                self.mc_sol_vel = self.ppxf_hdu['KIN_DATA'].data['MC_SOL_V']
                self.mc_sol_sig = self.ppxf_hdu['KIN_DATA'].data['MC_SOL_SIGMA']
                self.mc_err_vel = self.ppxf_hdu['KIN_DATA'].data['MC_ERR_V']
                self.mc_err_sig = self.ppxf_hdu['KIN_DATA'].data['MC_ERR_SIGMA']
            except:
                print("Warning: No MC data available.")
                pass

        # Load emission line output
        if self.emission:
            print("Emission line data found")
            self.ppxf_emission_hdu = fits.open(f"{self.dir_path}/ppxf_emission.fits")
            self.emission_bestfit = self.ppxf_emission_hdu['BESTFIT'].data['BESTFIT'].T
            self.gas_bestfit = self.ppxf_emission_hdu['GASBESTFIT'].data['GASBESTFIT']
            self.gas_bestfit_templates = self.ppxf_emission_hdu['GASBESTFIT_TEMPLATES'].data
        else:
            self.gas_bestfit = None

        self.signal_bin = np.zeros(self.ubins.size)
        for j in range(len(self.ubins)):
            w = (self.bin_num_long == j)
            self.signal_bin[j] = self.signal[w][0]