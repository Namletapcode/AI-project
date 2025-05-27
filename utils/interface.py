from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QCheckBox, QLabel
)
import sys
from PySide6.QtCore import Qt
from configs import bot_config

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Agent Settings")
        self.setFixedSize(300, 200)
        self.init_ui()
        self.set_dark_theme()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("⚙️ Agent Controls")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")

        self.checkbox_vision = QCheckBox("Show Bot Vision")
        self.checkbox_vision.setChecked(bot_config.bot_draw)
        self.checkbox_vision.stateChanged.connect(self.update_bot_draw)

        self.checkbox_param = QCheckBox("Use Param Agent")
        self.checkbox_param.setChecked(True)

        self.checkbox_vision_agent = QCheckBox("Use Vision Agent")
        self.checkbox_vision_agent.setChecked(False)

        layout.addWidget(title)
        layout.addWidget(self.checkbox_vision)
        layout.addWidget(self.checkbox_param)
        layout.addWidget(self.checkbox_vision_agent)

        self.setLayout(layout)
        
    def update_bot_draw(self, state):
        bot_config.bot_draw = bool(state)

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