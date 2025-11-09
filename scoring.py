class Error(Exception):
    """Base class for other exceptions"""
    pass

class WrongShotScoring(Error):
    """Raised when the shot score is too much"""
    pass

class WrongTotalScoring(Error):
    """Raised when the total score is too much"""
    pass


def pointCalculator(frame, shot1, shot2, shot3=0):
    try:
        if shot1+shot2>10:
            raise WrongTotalScoring

        elif frame<10:
            if shot1 == 10:
                print('Strike \n')
            elif shot2 == 10 or (shot1 + shot2 == 10):
                print('Spare \n')
            elif 0 < (shot1+shot2) < 10:
                print('Point: {} \n'.format(shot1+shot2))
            else:
                print('Gutter \n')

    except WrongTotalScoring:
        print('Wrong total point scoring \n')


frame_number = [1, 2, 3, ..., 10]

while True:
    try:
        shot1input = int(input('Shot1 = '))

        if shot1input > 10 or shot1input < 0:
            raise WrongShotScoring
            break

        else:
            if shot1input < 10:
                shot2input = int(input('Shot2 = '))

                if shot2input > 10 or shot2input < 0:
                    raise WrongShotScoring
                    break

            else:
                shot2input = 0

        pointCalculator(frame_number[2], shot1input, shot2input)

    except WrongShotScoring:
        print('Wrong shot point, please enter between 0-10 \n')
