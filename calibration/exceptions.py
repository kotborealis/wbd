class UnsupportedCalibrationMode(Exception):
    def __repr__(self):
        return """UnsupportedCalibrationMode('choose from: "right" or "left" mode')"""
