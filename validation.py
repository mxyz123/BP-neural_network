import tensorflow as tf
import keras as kr
import numpy as np
from PIL import Image
import cv2 as cv

model = kr.models.load_model("NoEx-whole_model.keras")
model.compile()
model.summary()

def get_pred_arr(path):
    #path  = np.asarray(Image.open(path).resize(size=(60, 60)), dtype=np.uint8)
    path = tf.expand_dims(path, 0)
    pred_ = model.predict(path)
    #print(pred_)
    out_ = np.argmax(pred_)
    #print(out_)
    match int(out_):
        case 0:
            print("down")
            return
        case 1:
            print("front")
            return
        case 2:
            print("left")
            return
        case 3:
            print("none")
            return
        case 4:
            print("right")
            return
        case 5:
            print("up")
            return
      
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (60, 60))
    print(get_pred_arr(img))
    # Display the resulting frame
    cv.imshow('frame', img)
    if cv.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

'''
get_pred_arr("./organized_images/front/20250310004443.png")
get_pred_arr("./organized_images/front/20250310004535.png")
get_pred_arr("./organized_images/front/20250312181913.png")
get_pred_arr("./organized_images/front/capture_2.jpg")
get_pred_arr("./organized_images/front/capture_27.jpg")
get_pred_arr("./organized_images/front/capture_67.jpg")
get_pred_arr("./organized_images/none/images_original_images_original_20250312182709.png_fcc74cb5-1bf9-45a1-b1ba-a401057e1e0b.png_9fd7a6d1-7e22-45d4-b253-b948d569dbf5.png")
get_pred_arr("./organized_images/up/capture_59.jpg")
get_pred_arr("./organized_images/up/capture_104.jpg")
get_pred_arr("./organized_images/up/20250312183149.png")
'''