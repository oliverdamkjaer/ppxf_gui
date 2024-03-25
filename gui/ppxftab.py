from PySide6.QtWidgets import QWidget, QFileDialog, QStatusBar, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QGroupBox
from PySide6 import QtGui
import astropy.io.fits as fits
import requests
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class PpxfTab(QWidget):
    def __init__(self):
        super().__init__()

        self.dir_path = "/Users/oddam/Dropbox/Universitet/ppxf_viewer/output/HCG91c_WFM-NOAO-N_test"

        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.create_source_input()
        self.create_crop_input()

        self.create_crop_map()
        self.create_right_layout()

        self.main_layout.addLayout(self.left_layout, 1)
        self.main_layout.addLayout(self.right_layout, 2)

        self.setLayout(self.main_layout)
        
    def create_source_input(self):

        select_file_button = QPushButton("Select file")
        select_file_button.clicked.connect(self.select_file_button_clicked)
        self.file_name_label = QLabel("No file selected")

        redshift_label = QLabel('Redshift:')
        self.redshift_line_edit = QLineEdit()
        self.redshift_line_edit.setValidator(QtGui.QDoubleValidator(0.0, 9.999, 3))
        self.redshift_line_edit.textEdited.connect(self.redshift_line_edit_text_edited)

        file_selection_layout = QHBoxLayout()
        file_selection_layout.addWidget(select_file_button)
        file_selection_layout.addWidget(self.file_name_label)

        redshift_layout = QHBoxLayout()
        redshift_layout.addWidget(redshift_label)
        redshift_layout.addWidget(self.redshift_line_edit)

        source_input_layout = QVBoxLayout()
        source_input_layout.addLayout(file_selection_layout)
        source_input_layout.addLayout(redshift_layout)

        source_input_group = QGroupBox("Source information")
        source_input_group.setLayout(source_input_layout)

        self.left_layout.addWidget(source_input_group)
    
    def create_crop_input(self):
        x_label = QLabel("x:")
        y_label = QLabel("y:")
        width_label = QLabel("width:")

        x_spin_box = QSpinBox()
        y_spin_box = QSpinBox()
        width_spin_box = QSpinBox()

        lower_wave_range_line_edit = QLineEdit()
        lower_wave_range_line_edit.setPlaceholderText("Lower bound")
        upper_wave_range_line_edit = QLineEdit()
        upper_wave_range_line_edit.setPlaceholderText("Upper bound")

        lower_snr_range_line_edit = QLineEdit()
        lower_snr_range_line_edit.setPlaceholderText("Lower bound")
        upper_snr_range_line_edit = QLineEdit()
        upper_snr_range_line_edit.setPlaceholderText("Upper bound")

        crop_layout = QHBoxLayout()
        crop_layout.addWidget(x_label)
        crop_layout.addWidget(x_spin_box)
        crop_layout.addWidget(y_label)
        crop_layout.addWidget(y_spin_box)
        crop_layout.addWidget(width_label)
        crop_layout.addWidget(width_spin_box)

        wave_range_layout = QHBoxLayout()
        wave_range_layout.addWidget(lower_wave_range_line_edit)
        wave_range_layout.addWidget(upper_wave_range_line_edit)

        snr_range_layout = QHBoxLayout()
        snr_range_layout.addWidget(lower_snr_range_line_edit)
        snr_range_layout.addWidget(upper_snr_range_line_edit)

        crop_group = QGroupBox("Crop")
        crop_group.setLayout(crop_layout)

        wave_range_group = QGroupBox("Wavelength region (Å):")
        wave_range_group.setLayout(wave_range_layout)

        snr_range_group = QGroupBox("SNR region (Å):")
        snr_range_group.setLayout(snr_range_layout)

        self.left_layout.addWidget(crop_group)
        self.left_layout.addLayout(wave_range_layout)
        self.left_layout.addLayout(crop_layout)
        self.left_layout.addWidget(wave_range_group)
        self.left_layout.addWidget(snr_range_group)

    def create_right_layout(self):
        crop_map_layout = QVBoxLayout()
        crop_map_layout.addWidget(NavigationToolbar(self.canvas_crop_map, self), 1)
        crop_map_layout.addWidget(self.canvas_crop_map)
        self.right_layout.addLayout(crop_map_layout)

        self.toggle_button = QPushButton('Activate Selector', self)
        self.toggle_button.clicked.connect(self.toggle_selector)
        self.right_layout.addWidget(self.toggle_button)

    def get_redshift_from_ned(self):
        query = {"name": {"v": self.object_name}}
        response = requests.post('https://ned.ipac.caltech.edu/srs/ObjectLookup', json=query)
        data = response.json()
        self.redshift = data['Preferred']['Redshift']['Value']
        self.redshift_line_edit.setText(str(self.redshift))

    def select_file_button_clicked(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select file", "", "FITS files (*.fits)")
        if file:
            self.file_name = file.split("/")[-1]
            self.file_name_label.setText(self.file_name)
            hdu = fits.open(file)
            self.object_name = hdu[0].header['OBJECT'].replace(' ', '')
            self.cube_data = hdu[1].data
            
            # Update the crop_map_img with the new data
            self.ax_crop_map.cla()
            data = np.nanmean(self.cube_data, axis=0)
            vmin, vmax = np.nanpercentile(data, [1, 99])
            self.crop_map_img = self.ax_crop_map.imshow(data, origin='upper', interpolation='nearest', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            self.canvas_crop_map.draw()

            try:
                self.get_redshift_from_ned()
            except:
                print("Failed to obtain redshift from NED/IPAC.")

    def redshift_line_edit_text_edited(self):
        self.redshift = float(self.redshift_line_edit.text())

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        self.rs_x1, self.rs_y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        self.rs_x2, self.rs_y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        print(f"({self.rs_x1}, {self.rs_y1}) --> ({self.rs_x2}, {self.rs_y2})")
        print(f"The button you used were: {eclick.button} {erelease.button}")

    # In your toggle_selector method
    def toggle_selector(self):
        if self.RS.active:
            self.RS.set_active(False)
            self.toggle_button.setText('Activate Selector')
        else:
            self.RS.set_active(True)
            self.toggle_button.setText('Deactivate Selector')

    def create_crop_map(self):
        self.fig_crop_map = Figure(layout="constrained")
        self.fig_crop_map.patch.set_facecolor("None")
        self.canvas_crop_map = FigureCanvas(self.fig_crop_map)
        self.canvas_crop_map.setStyleSheet("background-color:transparent;")

        self.ax_crop_map = self.canvas_crop_map.figure.subplots()
        self.ax_crop_map.set_xlabel('Offset (arcsec)')
        self.ax_crop_map.set_ylabel('Offset (arcsec)')

        placeholder_data = np.zeros((100, 100))  # Create an array of zeros
        self.crop_map_img = self.ax_crop_map.imshow(placeholder_data, origin='upper', interpolation='nearest', cmap='RdBu_r')
        #self.crop_map_img = self.ax_crop_map.imshow(np.log(np.nanmean(self.cube_data, axis=0)), origin='upper', interpolation='nearest', cmap='RdBu_r')

        self.ax_crop_map.set_xlabel("Offset (arcsec)")
        self.ax_crop_map.set_ylabel("Offset (arcsec)")
        self.ax_crop_map.minorticks_on()
        self.ax_crop_map.tick_params(length=10, width=1, which='major')
        self.ax_crop_map.tick_params(length=5, width=1, which='minor')

        self.ax_crop_map.set_xlabel('Offset (arcsec)')
        self.ax_crop_map.set_ylabel('Offset (arcsec)')
        
        self.RS = RectangleSelector(self.ax_crop_map, self.line_select_callback,
                                           useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
        # Deactivate the RectangleSelector
        self.RS.set_active(False)
        self.canvas_crop_map.mpl_connect('key_press_event', self.toggle_selector)

        self.canvas_crop_map.draw()

        