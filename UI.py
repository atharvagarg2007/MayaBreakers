from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QFont
import sys

from final_detector import is_duplicate   

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Duplicate Detector")
        self.setGeometry(50, 50, 1100, 1300)

        self.file_path = None

        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")

        about_action = QtWidgets.QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        self.layout = QVBoxLayout()
        #Image layout
        self.image_label = QLabel("Load an image")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed gray;")
        self.image_label.setFixedSize(1000, 1000)
        self.layout.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)

        self.result_label = QLabel("")
        self.result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        font = QFont("Arial", 15)

        # buttons area
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setFont(font)
        self.load_btn.clicked.connect(self.load_image)
        self.layout.addWidget(self.load_btn)

        self.check_btn = QPushButton("Check Image")
        self.check_btn.setFont(font)
        self.check_btn.clicked.connect(self.click_image)
        self.layout.addWidget(self.check_btn)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def load_image(self):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )

        if not self.file_path:
            return

        pixmap = QPixmap(self.file_path)
        pixmap = pixmap.scaled(
            1000, 1000,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("border: none;")

    def click_image(self):
        if not self.file_path:
            self.result_label.setText("Please load an image first.")
            self.result_label.setStyleSheet("color: orange; font-size: 16px;")
            return

        is_dup, similarity, matched = is_duplicate(self.file_path)

        if is_dup:
            self.result_label.setText(
                f"DUPLICATE\nMatched with: {matched}\nSimilarity: {similarity:.3f}"
            )
            self.result_label.setStyleSheet("color: red; font-size: 30px; font-weight: bold;")
        else:
            self.result_label.setText(
                f"UNIQUE IMAGE\nBest Similarity: {similarity:.3f}"
            )
            self.result_label.setStyleSheet("color: green; font-size: 30px; font-weight: bold;")

    def show_about(self):
        QtWidgets.QMessageBox.about(
            self,
            "About",
            "Duplicate Image Detector\n\n"
            "Developed by: Team MayaBreakers\n"
            "MNNIT Allahabad\n\n"
            "Contact: atharva090307@gmail.com"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
