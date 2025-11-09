import cv2


class Camera:
    def __init__(self, cam_source=0):
        self.cam_source = cam_source
        self.cap = cv2.VideoCapture(self.cam_source)
        print('[STATUS] Camera turning on...')
        if self.cap.isOpened():
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame = None
        else:
            raise ValueError("[ERR] Camera is not open/plug.")

    def video_read(self, show=True, gray=False):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret and show:
                if gray:
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Source', self.frame)
                if cv2.waitKey(25) & 0xFF == ord('s'):
                    self.take_image()
                return self.frame
            elif ret and not show:
                return self.frame
            else:
                raise ValueError("[ERR] Can't get image from source.")
        else:
            raise ValueError("[ERR] Camera is not open.")

    def video_stop(self):
        if self.cap.isOpened():
            self.cap.release()
            print('[STATUS] Camera stopped.')
        else:
            print('[STATUS] Camera not yet turned on.')

    def take_image(self):
        try:
            cv2.imwrite('snapshot.jpg', self.frame)
            print('[STATUS] Image saved.')
        except:
            print("[ERR] Failed to save image.")

    def resolution(self):
        print('Resolution = ' + str(self.width) + 'x' + str(self.height))
