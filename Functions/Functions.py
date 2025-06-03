import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np
import cv2 as cv
import mediapipe as mp
import numpy as np
from keras.utils import to_categorical
from scipy.interpolate import InterpolatedUnivariateSpline


#This function is to extract all landmarks from an image
#image_pose is the image that needs to be analysed
#pose is the pose function in mediapipe
def detectPose(image_pose, pose):
    
    #create copy of the original image
    original_image = image_pose.copy()
    
    #process the image to get landmark data
    resultant = pose.process(original_image)
    #list of landmarks
    landmarks = resultant.pose_landmarks.landmark

    return landmarks



#This function takes the path of a video
#it returns the angles between "hip to shoulder", "elbow to shoulder", "wrist to elbow" and the horizontal
def extract_angles(path):
    cap = cv.VideoCapture(path)
    frames = []
    #the following loop extracts each frame as an RGB array and appends it to the currently empty frames list.
    while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
            break
          else:
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
    #Measure the fps of the video
    fps = np.round(cap.get(cv.CAP_PROP_FPS))
    
    #Make sure that the data is at 15fps
    if fps==30:
        frames = frames[::2]
    elif fps==60:
        frames = frames[::4]
    else:
        print('The frame rate of the video is not as expected. FPS = ', fps)
        
    # Initialize mediapipe pose class.
    mp_pose = mp.solutions.pose
    # Setup the Pose function for images - independently for the images standalone processing.
    pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1)
    
    #initialize a tensor of zeros that will contain the X and Y coordinates of all 33 landmarks for every frame in frames
    keypoints = np.zeros((len(frames), 33, 2))
    
    #loop through each frame and assign the landmarks to the corressponding location in the keypoints tensor
    for i in range(len(frames)):
        landmarks = detectPose(frames[i], pose_image)
        XY = np.array([(landmark.x, landmark.y) for landmark in landmarks])
        keypoints[i] = XY
#         print([i])
    
    #The not all landmarks are relevant.
    #Extract only relevant landmarks by creating a boolean mask.
    useful_landmarks = [11,13,15,23]
    mask = np.zeros((33),dtype=bool)
    for i in useful_landmarks:
        mask[i] = True
    #Apply the mask to the keypoints. This is now the X and Y coordinates of the relevant landmarks for each frame.
    keypoints = keypoints[:,mask,:]
    
    #Change the permutation of the landmarks. Currently it is Shoulder, Elbow, Wrist, Hip.
    #We want to change to Hip, Shoulder, Elbow, Wrist
    perm = [3,0,1,2]
    keypoints = keypoints[:,perm]
    
    #Now create a tensor that represents the vectors from "hip to shoulder", "shoulder to elbow", "elbow to wrist".
    joints = keypoints[:,1:,:]-keypoints[:,:-1,:]
    
    #it makes more sense for all vectors to be pointing in the same direction when a person is standing upright with arms at the side, therefore multiply the second and third vecotrs by -1. This results in vector "hip to shoulder", "elbow to shoulder", "wrist to elbow".
    joints[:,1:,:] = joints[:,1:,:]*-1
    
    #Now turn each vector into a unit vector, this allows for a dot product to only be a function of the angle between vectors.
    magnitude = np.sqrt(np.sum(joints**2,axis=-1))
    unit_vec = np.copy(joints)
    unit_vec[:,:,0], unit_vec[:,:,1] = joints[:,:,0]/magnitude, joints[:,:,1]/magnitude
    
    #now find the angle between each vector and the horizontal (I will use arctan2 which gives the vector in the range of -pi to pi).
    angle = np.zeros((unit_vec.shape[0],3))
    for i in range(unit_vec.shape[0]):
        angle[i,0] = np.arctan2(unit_vec[i,0,1],unit_vec[i,0,0])
        angle[i,1] = np.arctan2(unit_vec[i,1,1],unit_vec[i,1,0])
        angle[i,2] = np.arctan2(unit_vec[i,2,1],unit_vec[i,2,0])
        
    #Now normalise the angles so that they are between -1 and 1, we can do this by dividing by pi.
    angle = angle/np.pi
    
    #now return the tensor that contains the angles for each frame.
    return angle



#This function takes a list of numbers x and returns a moving average with window of size window_size.
#Note that there are some logical rules applied to the end of the 
def moving_average(x, window_size): #make sure that the window size is an odd number, this means that the returned array is the same size as the input.
    #first I need to pad the data
    padded = np.pad(x,(window_size//2,window_size//2),mode='edge')
    window = np.ones(window_size)/window_size
    return np.convolve(padded, window, mode='valid')


#This function takes in the path of a video, pre-processes the information, and then makes a prediction on all frames on if a concentric contraction, eccentric contraction, or no curl is happening.
def Bicep_Curl_Classification(path,model_name):
    #firstly import the GRU model
    model_gru = tf.keras.models.load_model(f"C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 2\\Classification\\Models\\{model_name}.h5")
    #Extract the relevant angles for each frame in the video.
    Angles = extract_angles(path)
    
    #Next, perform some smoothing of the Angles data
    window_size = 7
    angles = np.copy(Angles)
    angles[:,0], angles[:,1], angles[:,2] = moving_average(Angles[:,0], window_size), moving_average(Angles[:,1], window_size), moving_average(Angles[:,2], window_size)
    
    #Now, create the windows of size X that a prediction will be made about.
    dt = 1/15 #15 is the frame rate
    X = 19
    sequences = np.zeros((angles.shape[0]-X+1, X, 3))
    for i in range(X,angles.shape[0]):
        sequences[i-X] = angles[i-X:i]
    
    #Now make a prediction with the imported model
    result = model_gru.predict(sequences)
    
    #Now take a moving average. I have found that this ensures that the class prediction isn't too sensitive and doesn't change abruptly for a small number of frames just to change back. I use a windoe size of 7, this is arbitrary, but I don't want too large of a window, just large enough to get rid of any high frequency changes (7 works well).
    Window_Size = 7
    result[:,0],result[:,1],result[:,2]= moving_average(result[:,0],Window_Size),moving_average(result[:,1],Window_Size),moving_average(result[:,2],Window_Size)
    
    ###################################################################################
    ##Now apply logical rules to the output of the model to acquire the final output.##
    ###################################################################################
    #Extract the class prediction for each frame
    prediction = to_categorical(np.argmax(result,axis=1),3)
    
    #Find the indexes where the class changes
    change = prediction[1:]*prediction[:-1]
    C = np.where(np.sum((change == np.array([0,0,0])), axis=1) == 3)[0]
    
    #Create a list representing the class of each chunk
    class_chunk = [np.argmax(result[i+1]) for i in C-1]

    #create a list that represents the previous class chunk corresponding to the current class chunk.
    #Note that I add class 0 ("Not Curl") to the beginning since there is not an actual previous chunk
    prev_chunk = [0]+class_chunk[:-1]

    #Now change the prediction vector based on the rule that if the previous class chunk was not a "concentric contraction", then the current cannot be an "eccentric contraction"
    for i in range(len(class_chunk)):
        if class_chunk[i] == 2 and prev_chunk[i] != 1:
            ind1 = C[i-1]+1
            ind2 = C[i]+2
            
            #If the first chunk in the result is an eccentric contraction, then we need to change "ind1" to equal 0.
            if i==0:
                ind1 = 0
            prediction[ind1:ind2] = np.array([1,0,0])
                
    #Now return the prediction in addition to the Angles array
    return prediction, Angles



#This function takes the angular data stored in Angles from the function extract_angles in addition to normalised time that ranges from 0 to 1 (just use np.linspace(0,1,len(Angles))) and the desired number of points that we want to sample. The function returns the same trajectory of angles, however now it is represented by the custom amount of points (sequence_length).
def equal_lengths(Data, Time, sequence_length):
    #The data is three dimensional, therefore interpolate each dimension separately.
    X = Data[:,0]
    Y = Data[:,1]
    Z = Data[:,2]
    
    X_interp = InterpolatedUnivariateSpline(Time, X, k=3)
    Y_interp = InterpolatedUnivariateSpline(Time, Y, k=3)
    Z_interp = InterpolatedUnivariateSpline(Time, Z, k=3)
    
    #Now sample the data at time intervals dt then re-combine
    t = np.linspace(0,1,sequence_length)
    x = X_interp(t)
    y = Y_interp(t)
    z = Z_interp(t)
    equal_length_data = np.stack((x,y,z),axis=1)
    
    #Return the data and the normalised time
    return equal_length_data