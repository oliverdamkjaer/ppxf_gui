from PySide6.QtWidgets import QApplication
import sys

from mainwindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QGroupBox { font-size: 14px; }")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())