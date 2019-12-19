###################################################
###     Create The matching points manually     ###
###################################################
import cv2
import numpy as np
import scipy.io

global source, dest, match_p_dst, match_p_src, frame, frame2

source = cv2.imread('src.jpg')
dest = cv2.imread('dst.jpg')

frame = source
frame2 = dest

match_p_dst = []
match_p_src = []


def selectROI(event, x, y, flags, param):
    global match_p_dst, match_p_src, frame, frame2
    # if we are in points selection mode, the mouse was clicked,
    # and we do not already have 25 points, then update the
    # list of  points with the (x, y) location of the click
    # and draw the circle

    if event == cv2.EVENT_LBUTTONDOWN and len(match_p_src) < 25 and param == 1:
        match_p_src.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
    elif event == cv2.EVENT_LBUTTONDOWN and len(match_p_dst) < 25 and param == 2:
        match_p_dst.append((x, y))
        cv2.circle(frame2, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame2", frame2)


cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800, 600)
cv2.setMouseCallback("frame", selectROI, param=1)
cv2.imshow("frame", frame)

cv2.namedWindow("frame2", cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame2', 800, 600)
cv2.setMouseCallback("frame2", selectROI, param=2)
cv2.imshow("frame2", frame2)

# keep looping until 50 points have been selected
while (len(match_p_src) + len(match_p_dst)) < 50:
    print('Press any key when finished marking the points!! ')
    cv2.waitKey(0)

match_p_src = np.array(match_p_src, dtype=float)
match_p_dst = np.array(match_p_dst, dtype=float)
cv2.destroyAllWindows()
matches_test = {'match_p_dst': match_p_dst.T, 'match_p_src': match_p_src.T}
scipy.io.savemat("matches.mat", matches_test)
print('saved to matches.mat')
