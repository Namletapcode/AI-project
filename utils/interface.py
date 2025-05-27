from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QCheckBox, QLabel, QPushButton
)
import sys
from PySide6.QtCore import Qt
from configs import bot_config

class SettingsWindow(QWidget):
    def __init__(self, share_state):
        super().__init__()
        self.share_state = share_state
        self.setWindowTitle("AI Agent Settings")
        self.setFixedSize(200, 200)
        self.init_ui()
        self.set_dark_theme()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("⚙️ Agent Controls")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        
        # Add bot type button
        self.bot_type_button = QPushButton("Parameter Bot")
        self.bot_type_button.clicked.connect(self.toggle_bot_type)
        self.bot_type_button.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                border: 2px solid #3a9ff5;
                border-radius: 4px;
                padding: 5px;
                color: #dddddd;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)

        self.checkbox_vision = QCheckBox("Show Bot Vision")
        self.checkbox_vision.setChecked(bot_config.bot_draw)
        self.checkbox_vision.stateChanged.connect(self.update_bot_draw)

        layout.addWidget(title)
        layout.addWidget(self.bot_type_button)
        layout.addSpacing(10)  # Add some space between button and checkbox
        layout.addWidget(self.checkbox_vision)

        self.setLayout(layout)
        
    def update_bot_draw(self, state):
        self.share_state.bot_draw = bool(state)
    
    def toggle_bot_type(self):
        self.share_state.is_vision = not self.share_state.is_vision
        self.bot_type_button.setText("Vision Bot" if self.share_state.is_vision else "Parameter Bot")
        
    def set_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #dddddd;
                font-family: Segoe UI, Roboto, sans-serif;
            }
            QCheckBox {
                spacing: 10px;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #888;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                image: url("./utils/Checked.svg");
                background-color: #2b2b2b;
                border: 2px solid #3a9ff5;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.show()
    app.exec()