import cv2
import numpy as np


class Sensor:
    def __init__(self, frame, source_image, pin_bounds=None, line_bound=None, threshold=50, average1=40, average2=200):
        self.frame = frame
        self.source_image = source_image
        self.pin_bounds = pin_bounds
        self.line_bound = line_bound
        self.threshold = threshold
        self.average1 = average1
        self.average2 = average2
        self.pins = np.ones(shape=(10, 1))
        self.is_ball_passed = False

    def ball_detection(self):
        try:
            if self.line_bound is not None:
                source_image_roi = self.source_image[self.line_bound[1]:self.line_bound[1] + self.line_bound[3],
                                   self.line_bound[0]:self.line_bound[0] + self.line_bound[2]]
                source_image_roi_gray = cv2.cvtColor(source_image_roi, cv2.COLOR_BGR2GRAY)

                frame_roi = self.frame[self.line_bound[1]:self.line_bound[1] + self.line_bound[3],
                            self.line_bound[0]:self.line_bound[0] + self.line_bound[2]]
                frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)

                diff = cv2.absdiff(source_image_roi_gray, frame_roi_gray)
                thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]
                average = np.mean(thresh)

                if average > self.average1:
                    cv2.putText(self.frame, "Ball Passed", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.is_ball_passed = True

                else:
                    cv2.putText(self.frame, "Ball not Passed", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.is_ball_passed = False
            else:
                raise ValueError('[ERR] Please enter line bound')
                raise Exception
        except:
            print("[ERR] Ball detection can't start")

    def pin_detection(self):
        try:
            if self.pin_bounds is not None:
                for (index, (x, y, dx, dy)) in enumerate(self.pin_bounds):
                    roi = self.frame[y:y + dy, x:x + dx]
                    thresh = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY)[1]
                    average2 = np.mean(thresh)

                    if average2 > self.average2:
                        cv2.rectangle(self.frame, (x, y), (x + dx, y + dy), (0, 255, 0), 3)
                        self.pins[index] = 1

                    else:
                        cv2.rectangle(self.frame, (x, y), (x + dx, y + dy), (0, 0, 255), 3)
                        self.pins[index] = 0

                    cv2.putText(self.frame, "Ball Passed", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(self.frame, f"{int(np.sum(self.pins))} pin-s down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255), 2)
            else:
                raise ValueError('[ERR] Please enter pin bounds')
                raise Exception
        except:
            print("[ERR] Pin detection can't start")
