from cv2 import blur
import tensorflow as tf
import tensorflow.keras.utils as utils
import numpy as np
import cv2
import time
import imutil
import mahotas as mh

# model = tf.keras.models.load_model('results')
# # img = tf.io.read_file('data/test/rock/testrock01-18.png')
# # tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
# # tensor = tf.image.resize(tensor, (120, 120))
# # input_tensor = tf.expand_dims(tensor, axis=0)
# class_name = ['paper', 'rock', 'scissors']

mirror=True
cam = cv2.VideoCapture(0)

pTime = 0
while True:
     # read webcam
    ret_val, img = cam.read()

    # flip image if mirrored
    if mirror: 
        img = cv2.flip(img, 1)

    # resize original image for displaying
    img = imutil.resize(img.copy(), width = 500, height = 500)

    # image processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    T= mh.thresholding.otsu(blurred)
    blurred[blurred>T] = 255
    blurred[blurred<255] = 0


    edged = cv2.Canny(blurred, 30, 150)
    cv2.line(edged, (1, 498), (498, 498), 255, thickness=1)
    
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort the contours from the largest to smallest
    # only taking the 10 largest one 
    if len(cnts) > 0:
        cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[0]

        mask = np.zeros(img.shape[:2], np.uint8)

        cv2.fillPoly(mask, pts =[cnts], color=255)

        img = cv2.bitwise_and(img, img, mask=mask)
        # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
        black_pixels = np.where(
            (img[:, :, 0] == 0) & 
            (img[:, :, 1] == 0) & 
            (img[:, :, 2] == 0)
        )

        # set those pixels to white
        img[black_pixels] = [255, 255, 255]


    # convert colour back 
    edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

    blurred = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    # frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(edged, f'FPS:{int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    # stacking original and edged
    display= np.hstack([edged, blurred, img])

    cv2.imshow("display", display)

    if cv2.waitKey(1) == 27: 
        break  # esc to quit


# test = utils.image_dataset_from_directory(
#     'data/test',
#     label_mode = 'categorical',
#     class_names = None,
#     image_size = (300, 300),
#     shuffle = False,
# )

# x = model.predict(test)
# print(class_name[np.argmax(x[1])])
# print(class_name[np.argmax(model(input_tensor))])

