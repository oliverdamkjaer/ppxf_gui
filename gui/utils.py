from PySide6.QtWidgets import QFrame

class Separator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setFixedWidth(10)