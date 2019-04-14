import numpy as np
import cv2

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def makeHologram(original):

    scale=0.5
    scaleR=4

    holo = cv2.resize(original, (480, 480), interpolation = cv2.INTER_CUBIC)
    
    top_side = holo.copy()
    bottom_side = rotate_bound(holo.copy(),180)
    right_side = rotate_bound(holo.copy(), 90)
    left_side = rotate_bound(holo.copy(), 270)
    
    hologram = np.zeros([max(holo.shape)*scaleR,max(holo.shape)*scaleR,3], holo.dtype)
    
    center_x = int((hologram.shape[0])/2)
    center_y = int((hologram.shape[1])/2)
    
    vert_x = int((top_side.shape[0])/2)
    vert_y = int((top_side.shape[1])/2)

    hologram[0:top_side.shape[0], center_x-vert_x:center_x+vert_x] = top_side
    hologram[ hologram.shape[1]-bottom_side.shape[1]:hologram.shape[1] , center_x-vert_x:center_x+vert_x] = bottom_side

    hori_x = int((right_side.shape[0])/2)
    hori_y = int((right_side.shape[1])/2)

    hologram[ center_x-hori_x : center_x-hori_x+right_side.shape[0] , hologram.shape[1]-right_side.shape[0] : hologram.shape[1]] = right_side
    hologram[ center_x-hori_x : center_x-hori_x+left_side.shape[0] , 0 : left_side.shape[0] ] = left_side
    
    return hologram


def video(video):
    cap = cv2.VideoCapture(video)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    holo = None
    ret = False
    while(not ret):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640), interpolation = cv2.INTER_CUBIC)
            holo = makeHologram(frame)
    out = cv2.VideoWriter('hologram1.mp4',fourcc, 15, (holo.shape[0],holo.shape[1]))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    while(True):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640), interpolation = cv2.INTER_CUBIC)
            holo = makeHologram(frame)
            out.write(holo)
            count += 1
        if(count>=total_frames-1):
            break
    
    cap.release()
    out.release()
    return

cap = cv2.VideoCapture(0)
cap.set(3, 640) #width
cap.set(4, 480) #height

if cap.isOpened():
    _,frame = cap.read()
    cap.release() #releasing camera immediately after capturing picture
    if _ and frame is not None:
        top_side = frame
        holo_img = makeHologram(frame)
        cv2.imwrite('holo.jpg', holo_img)
        video('test.mov')








