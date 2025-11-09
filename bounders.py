import cv2


class Bounders:
    def __init__(self, image):
        self.image = image
        self.line = []
        self.pins = []

    def set_pin_bounds(self):
        try:
            self.pins = cv2.selectROIs("Select pins in order", self.image, showCrosshair=True)
            cv2.destroyWindow("Select pins in order")
        except:
            print('[ERR] Pins location failed')

    def set_line_bounds(self):
        try:
            self.line = cv2.selectROI('Select Line', self.image, showCrosshair=True)
            cv2.destroyWindow('Select Line')
        except:
            print('[ERR] Pins location failed')
