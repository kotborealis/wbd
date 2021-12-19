class UnsupportedBoardMode(Exception):
    def __repr__(self):
        return """UnsupportedBoardMode('choose from: "right" or "left" mode')"""
