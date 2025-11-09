import numpy as np
import cv2
import time
import pyodbc
import pickle


def take_image():
    global lane_image
    global lane_image_gray
    if cap.isOpened():
        ret, lane_image = cap.read()
        if ret:
            lane_image_gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
        else:
            print("Error opening video stream or file")
    else:
        print("Not Found")


def set_pin_bounds():
    global pinBoundingBoxes
    pinBoundingBoxes = cv2.selectROIs("Select pins in order", lane_image, showCrosshair=True)
    cv2.destroyWindow("Select pins in order")


def set_line_bounds():
    global lineBoundBox
    lineBoundBox = cv2.selectROI('Select Line', lane_image, showCrosshair=True)
    cv2.destroyWindow('Select Line')


def ball_detection():
    global isBallPassed
    if cap.isOpened():
        print("OK11")
        lane = lane_image_gray
        print(lane)
        return lane
    else:
        print("Error opening video stream or file")


def pin_detection():
    global pins
    print("bagirmiyib?")    
    pins = np.ndarray(shape=(10, 1))
    print("sayde")
    cap = cv2.VideoCapture('C:/BowlingApp/Line2/lane_image_gray.jpg')     
    ret, lane = cap.read()
    print("Galde")
    if ret:
        index = 0
        print("Girde")
        for x, y, dx, dy in pinBoundingBoxes:

            roi = lane[y:y + dy, x:x + dx]
            _, thresh = cv2.threshold(roi, 90, 255, cv2.THRESH_BINARY)

            mean = np.mean(thresh)

            if mean > 200:
                cv2.rectangle(lane, (x, y), (x + dx, y + dy), (0, 255, 0), 3)
                pins[index] = 1

            else:
                cv2.rectangle(lane, (x, y), (x + dx, y + dy), (0, 0, 255), 3)
                pins[index] = 0

            print("firrande"+ str(index))
            index += 1

            cv2.putText(lane, "Ball Passed", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(lane, f"{10 - int(np.sum(pins))} pin-s down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        if cv2.waitKey(23) & 0xFF == ord('q'):
            pass

        print("chikhde")
        return lane
    else:
        print("Error opening video stream or file")



device_name = 'Line2'
lane_number = 2

cap = cv2.VideoCapture('C:/BowlingApp/Line2/lane_image_gray.jpg') 
print(cap)
take_image()

result = 'pin_ok'
if result is None:
    connection = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server= 192.168.10.11;Port=1433"
        "Database=bowling-db;"
        "UID=sa;"
        "PWD=J@han6898;"
        "Trusted_Connection=no"
    )
    cursor = connection.cursor()
    print(connection)

    set_pin_bounds()
    pin = 1
    for x, y, dx, dy in pinBoundingBoxes:
        sql = f"UPDATE [bowling-db].configuration.pin_location SET pin{pin}xydxdy = '{x},{y},{dx},{dy}' WHERE lane = '{lane_number}'"
        cursor.execute(sql)
        connection.commit()
        pin += 1
    with open('C:/BowlingApp/Line2/pinBoundingBoxes/pinBoundingBoxes', 'wb') as f:
        pickle.dump(pinBoundingBoxes, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('C:/BowlingApp/Line2/pinBoundingBoxes/pinBoundingBoxes', 'rb') as f:
        pinBoundingBoxes = pickle.load(f)


result = 'line_ok'
if result is None:
    connection = pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server= 192.168.10.11;Port=1433"
        "Database=bowling-db;"
        "UID=sa;"
        "PWD=J@han6898;"
        "Trusted_Connection=no"
    )
    cursor = connection.cursor()
    print(connection)
    
    set_line_bounds()
    sql = f"""UPDATE [bowling-db].configuration.ball_line_location SET line_box = 
        '{lineBoundBox[0]},{lineBoundBox[1]},{lineBoundBox[2]},{lineBoundBox[3]}' 
         WHERE lane = '{lane_number}'"""

    cursor.execute(sql)
    connection.commit()
    with open('C:/BowlingApp/Line2/lineBoundBox/lineBoundBox', 'wb') as f:
        pickle.dump(lineBoundBox, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('C:/BowlingApp/Line2/lineBoundBox/lineBoundBox', 'rb') as f:
        lineBoundBox = pickle.load(f)


shot = [None]
isBallPassed = False
print("OK1")

print("OK2")
lane = ball_detection()
cv2.imshow('image', lane_image_gray)
cv2.imshow('Lane', lane)
print("OK3")
#timeout = int(round(time.time() * 1000)) + 350
#while isBallPassed and int(round(time.time() * 1000)) < timeout:
lane = pin_detection()
cv2.imshow('Lane', lane)
print("OK4")

shot[0] = 10 - int(np.sum(pins))

f = open('C:/BowlingApp/Line2/Shots/shot.hmd', "w")
f.write(str(shot[0]))
f.close()

print(shot[0])
#cv2.destroyAllWindows()
