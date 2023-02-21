import cv2, os, math, time, argparse
from flask import Flask, request, render_template
import numpy as np

root_dir = "./model_files/"    # relative - root directory for model files
faceProto = root_dir +  "opencv_face_detector.pbtxt"      #config-file for face detection
faceModel = root_dir +  "opencv_face_detector_uint8.pb"   #pre-trained model for face detection
ageProto = root_dir +  "age_deploy.prototxt"              #config-file for age detection
ageModel = root_dir + "age_net.caffemodel"    #pre-trained model for age detection
genderProto = root_dir +  "gender_deploy.prototxt"   #config-file for gender detection
genderModel = root_dir +  "gender_net.caffemodel"    #pre-trained model for gender detection
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # mean values for gender detection
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
genderList = ['Male', 'Female']

ageNet = cv2.dnn.readNet(ageModel,ageProto)   # first parameter is used to store training weights, while the second is used to save network
genderNet = cv2.dnn.readNet(genderModel, genderProto) # first parameter is used to store training weights, while the second is used to save network
faceNet = cv2.dnn.readNet(faceModel, faceProto) # first parameter is used to store training weights, while the second is used to save network

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'   #this folder will contain all uploaded photos as well as default photo
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB upper limit for file size

def resize_image(raw_img):
    frame = cv2.imread(raw_img)  # orginal image is also stored for future use in the following code lines
    # resizing it for faster image processing time
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # fx=0.5, fy=0.5 means the image is being scaled down to half its original size
    return small_frame, frame

def getFaceBox(net, frame, conf_threshold = 0.75):     #confidence threshold for face detection is 75% by default.
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # scaling and normalization is done here for image processing
    # parameters are: the image, the scaling factor, the size of the image after scaling, the mean value of the image,flags
    blob = cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),[104, 117, 123], True, False)
    net.setInput(blob)   #dnn is fed with input image (scaled and normalized version)
    detections = net.forward() # neural network's learned weights are applied to the input data and produce a prediction
    bboxes = []    #detected images' boundaries(4 points) will be added to this list

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]* frameWidth)
            y1 = int(detections[0,0,i,4]* frameHeight)
            x2 = int(detections[0,0,i,5]* frameWidth)
            y2 = int(detections[0,0,i,6]* frameHeight)
            bboxes.append([x1,y1,x2,y2])

    return frameOpencvDnn , bboxes

imagename = "default.jpg"    # default image that will be shown to the user when you load homepage
image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagename)   # flask needs to know exact path for the default image

@app.route('/', methods=['GET', 'POST'])

def upload():
    global imagename, image_path
    if request.method == 'POST':     # this function is triggered as soon as "upload" button is clicked
        if 'file' not in request.files:
            return 'No file part in the request'
        file = request.files['file']   #getting the image file which is supplied by user
        imagename = file.filename     # image name is update here to replace "default.jpg" text
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagename)   # flask needs to know where the uploaded file is

        if file and file.content_length > app.config['MAX_CONTENT_LENGTH']:   # checking if the file is too large
            return 'File size exceeded the limit of 16 MB'   # error page will be returned if the file is oversized
        if imagename == '':     #if no image file is selected, name should be reverted to the default
            imagename = "default.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagename)
        else:
            file.save(image_path)   # saving the image into the "uploads" folder

    return render_template('index.html', image_path=image_path, imagename=imagename)

@app.route('/detect', methods=['GET', 'POST'])
def detect():    #this function is triggered as soon as "Detect Gender" button is clicked
    padding = 20
    warning = ""
    gender = age = ""
    global imagename, image_path
    if request.method == 'POST':
        small_frame, frame = resize_image(image_path)
        frameFace, bboxes = getFaceBox(faceNet, small_frame)   # detecting face(s) here
        if not bboxes:
            warning = "No Face Detected"  # you can modify "conf_threshold" parameter if there is face but not detected.
        else:
            if len(bboxes) > 1:
                warning = "More Than 1 Person Detected"   # the model detects all faces by default. we show error here if more than 1
            else:         # if there is only 1 face detected in the image file
                for bbox in bboxes:
                    # cropping the face portion from the whole photo in order to focus gender detection
                    face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)   #dnn is fed with face photo (scaled and normalized version) to detect gender
                    genderPreds = genderNet.forward()  # neural network's gender weights are applied to the input data and produce a gender prediction
                    gender = genderList[genderPreds[0].argmax()]  #argmax is used to obtain the index of the highest gender probability value
                    #print(f"Gender : {gender}, conf = {genderPreds[0].max()}")

                    ageNet.setInput(blob)   #dnn is fed with face photo (scaled and normalized version) to detect age range
                    agePreds = ageNet.forward()   # neural network's age weights are applied to the input data and produce a age prediction
                    age = "Age: "+ ageList[agePreds[0].argmax()]    #argmax is used to obtain the index of the highest age probability value
                    warning = u'\u2713'   # This is ✓   character. Shows that the process is completed successfully.
                    #print(f"Age : {age}, conf = {agePreds[0].max()}")
    return render_template('index.html', image_path=image_path, gender=gender, age=age, warning=warning,  imagename=imagename)

if __name__ == '__main__':
    app.run()
