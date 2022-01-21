import json

class CheckersBoard:
    def __init__(self):
        self.white = {"men": [], "kings": []}
        self.black = {"men": [], "kings": []}

    def add_white(self, field, piece_type="men"):
        self.white[piece_type].append(field)

    def add_black(self, field, piece_type="men"):
        self.black[piece_type].append(field)

    def plain(self):
        return {"white": self.white, "black": self.black}

    def toJSON(self):
        return json.dumps(self.plain())