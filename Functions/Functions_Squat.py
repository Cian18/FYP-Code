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
from tensorflow.keras.models import load_model







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










#this function is to extract the ratio of "the direct distance from head to toe" to the "height" of the person. This will be small if the person is crouching.
#This is for the squat classification in its simplest form.
def height_ratio(path):
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
        
        
    #Acquire the approximate total height of the person. Adding the average length of each shin, the average length of each thigh and the length of the torso and the length of the neck.
    shins = np.mean((np.linalg.norm(keypoints[:,25,:]-keypoints[:,27,:],axis=1),np.linalg.norm(keypoints[:,26,:]-keypoints[:,28,:],axis=1)))
    thighs = np.mean((np.linalg.norm(keypoints[:,23,:]-keypoints[:,25,:],axis=1),np.linalg.norm(keypoints[:,24,:]-keypoints[:,26,:],axis=1)))
    torso = np.linalg.norm(np.mean((keypoints[:,11,:],keypoints[:,12,:]),axis=0) - np.mean((keypoints[:,23,:],keypoints[:,24,:]),axis=0), axis=1)
    neck = np.linalg.norm(keypoints[:,0,:]-np.mean((keypoints[:,11,:],keypoints[:,12,:])),axis=1)
    height = shins+thighs+torso+neck
    
    head_toe = np.linalg.norm(keypoints[:,0,:]-np.mean((keypoints[:,27,:],keypoints[:,28,:]),axis=0),axis=1)
    
    ratio = moving_average(head_toe/np.mean(height),7)
    
    return ratio










#This function takes the path of a video
#it returns the relevant angles for analysis of a squat
def extract_landmarks(path):
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
    
    return keypoints
      
    
    
    
    
    
    
    
    
    
    
#This function uses the extract_landmarks function on a given video path to extract all features that I have deemed to be relevant. There are two types of features, angles of vectors relative to the horizontal and scalar quantities.
#The feature vector angles are: 'Left Shin', 'Right Shin', 'Left Thigh', 'Right Thigh', 'Torso/Spine'.
#The feature scalar values are: 'Knee Horizontal Separation','Foot Separation', 'Shoulders to hip vertical distance', 'Hips to knees vertical distance'
def features_front(path):
    keypoints = extract_landmarks(path)
    
    #Create two label lists, vector and scalar features that will be extracted.
    vec_label = ['Torso/Spine']
    scal_label = ['Knee Horizontal Separation','Foot Separation', 'Hips to knees vertical distance']
    
    #Create an array that contains all the feature vectors for each frame.
    feat_vec = (keypoints[:,12,:]+keypoints[:,11,:])/2-(keypoints[:,24,:]+keypoints[:,23,:])/2
    
    #Now normalise each vector to a unit vector.
    feat_vec = feat_vec/np.linalg.norm(feat_vec,axis=1,keepdims=True)
    
    #now find the angle between each vector and the horizontal (I will use arctan2 which gives the vector in the range of -pi to pi).
    angle = np.zeros(feat_vec.shape[0])
    #loop through each frame
    for i in range(feat_vec.shape[0]):
        #each angle will be in radians
        angle[i] = np.arctan2(feat_vec[i,1],feat_vec[i,0])
    
    #Create an array that contain all the feature scalars for each frame.
    feat_scal = np.array([np.abs(keypoints[:,26,0]-keypoints[:,25,0]), np.abs(keypoints[:,28,0]-keypoints[:,27,0]), ((keypoints[:,23,:]+keypoints[:,24,:])/2-(keypoints[:,25,:]+keypoints[:,26,:])/2)[:,1]])
    
    #Normalise all scalars relative to the average horzontal distance between the shoulders
    feat_scal = feat_scal/np.mean(np.abs(keypoints[:,11,0]-keypoints[:,12,0]))

    #Finally concatenate the vector angles and the scalars to make one array containing all relevant features.
    features = np.concatenate((angle.reshape(1,-1), feat_scal),axis=0)
    
    return features.T









#This function uses the extract_landmarks function on a given video path to extract all features that I have deemed to be relevant. There are two types of features, angles of vectors relative to the horizontal and scalar quantities.
#The feature vector angle is: 'Torso/Spine'
#The feature scalar values are: 'Heel vertical position (relative to the toe vertical position)', 'Hips to knees vertical distance'
def features_side_R(path):
    keypoints = extract_landmarks(path)
        
    #Create two label lists, vector and scalar features that will be extracted.
    vec_label = ['Torso/Spine']
    scal_label = ['Heel Vertical Position', 'Hips to knees vertical distance']
    
    #Create an array that contains all the feature vectors for each frame.
    feat_vec = keypoints[:,12,:]-keypoints[:,24,:]
    
    #Now normalise each vector to a unit vector.
    feat_vec = feat_vec/np.linalg.norm(feat_vec,axis=1,keepdims=True)
    
    #now find the angle between each vector and the horizontal (I will use arctan2 which gives the vector in the range of -pi to pi).
    angle = np.zeros(feat_vec.shape[0])
    #loop through each frame
    for i in range(feat_vec.shape[0]):
        #Each angle will be in radians
        angle[i] = np.arctan2(feat_vec[i,1],feat_vec[i,0])
    
    #Create an array that contain all the feature scalars for each frame.
    feat_scal = np.array([keypoints[:,30,1]-keypoints[:,32,1], keypoints[:,24,1]-keypoints[:,26,1]])
    
    #Normalise the scalar quanitites by the length of the right thigh
    feat_scal = feat_scal/np.mean(np.linalg.norm((keypoints[:,24,:]-keypoints[:,26,:]),axis=1))

    #Finally concatenate the vector angles and the scalars to make one array containing all relevant features.
    features = np.concatenate((angle.reshape(1,-1), feat_scal),axis=0)
    
    return features.T
    
    
    
    
    
    
    
    

#This function takes a list of numbers x and returns a moving average with window of size window_size.
#Note that there are some logical rules applied to the end of the 
def moving_average(x, window_size): #make sure that the window size is an odd number, this means that the returned array is the same size as the input.
    #first I need to pad the data
    padded = np.pad(x,(window_size//2,window_size//2),mode='edge')
    window = np.ones(window_size)/window_size
    return np.convolve(padded, window, mode='valid')










#This function will extract and normalise the landmarks of a person. This will then be ran through a model that predicts the camera angle.
#The function outputs both the normalise landmark data and the orientation of the person relative to the camera (can be one of the following classes: "Forwards", "Side Right", or "Side Left").
#Note that this function will operate on a single image. The reason being that in a video, only one camera angle is expected, so no need to operate on every single frame.
def camera_angle(image, model_path):
    # Initialize mediapipe pose class.
    mp_pose = mp.solutions.pose
    
    # Setup the Pose function for images - independently for the images standalone processing.
    pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1)
    landmarks = detectPose(image, pose_image)
    XY = np.array([(landmark.x, landmark.y) for landmark in landmarks])

    #Now normalize the data. Normalize x coord with respect to the maximum and minimum x coord. Then the same for y.
    #Define the maximum and minimum value for x and y coordinates from all landmarks.
    xmax = np.max(XY[:,0])
    xmin = np.min(XY[:,0])
    ymax = np.max(XY[:,1])
    ymin = np.min(XY[:,1])
    
    #normalize the landmarks
    X = (XY[:,0]-xmin)/(xmax-xmin)
    Y = (XY[:,1]-ymin)/(ymax-ymin)
    data = (np.stack((X,Y)).T).reshape(1,66)
    
    #now put the data through the model to make a prediction
    model = load_model(model_path)
    result = np.argmax(model.predict(data))
    if result == 0: 
        cam_ang = 'Front'
    elif result == 1:
        cam_ang = 'Side Right'
    else:
        cam_ang = 'Side Left'
    
    return data, cam_ang










#This function takes in a list of arrays which may vary in length. These arrays represent the trajectory of each relevant joint angle in a concentric or eccentric contraction.
#Another input is the desired sequence length.
#The output is an array that contains all trajectories, however, now they are all of equal lengths having been interpolated with cubic splines.
def equal_lengths(Data, sequence_length):
    #Acquire the number of repetitions
    rep_num = len(Data)
    
    #Acquire the number of joint angles that are to be interpolated
    joint_num = Data[0].shape[1]
    
    #Initiallize and empty array that will be filled with the interpolate trajectories
    Data_interp = np.zeros((rep_num,sequence_length,joint_num))
    
    #loop through each repetition
    for i in range(rep_num):
        #create a normalised time scale that matches the length of the current data. (this is required for the interpolation).
        time = np.linspace(0,1,len(Data[i]))
        
        #loop through each joint
        for j in range(joint_num):
            interp = InterpolatedUnivariateSpline(time, Data[i][:,j])
            
            #create a normalised time scale that matches the desired length of the data.
            t = np.linspace(0,1,sequence_length)
            
            #Acquire the trajectory as an array of the desired length. Fill this into the corresponding location in the empty array
            Data_interp[i,:,j] = interp(t)
    
    #Return the modified data
    return Data_interp








#This function takes the video path of a person performing a squat as an input. It then extracts landmarks, then segments the video by extracting the chunks of frames that contain concentric contractions and eccentric contractions. Next, it detects the orientation of the person relative to the camera. Based on the orientation of the person it extracts the relevant joint angle trajectories for all eccentric and concentric contractions. Also based on the orientation, it imports the relevant "correct technique" regression model. Next, it finds the to what degree the repetition is complete; based on this it scales the trajectory of each of the otherFinally, it finds the signed error of the actual angle compared to the "correct" angle at N equally spaced points throughout each contraction.
#The output of the model is two arrays, one representing the signed error at all N points for each concentric contraction, and the other array is the same for the eccentric contraction.

#To avoid inconvenience, the model paths will not be arguments in the function, therefore, if the path of the models changes, it will need to be changed directly within the function here.
def signed_error(N, video_path):
    
    
    ###########
    ##Step 1 ## I will run the height_ratio() function, then I will assemble this data into windows spanning 19 frames.
    ###########
    height = height_ratio(video_path)
    window = 19
    sequences = np.zeros((height.shape[0]-window+1, window))
    for i in range(window,height.shape[0]+1):
        sequences[i-window] = height[i-window:i]
        
    #Load the classification model
    squat_class = load_model("C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 3\\Classification\\Models\\Final Model.h5")
    
    #now I input this data into the model that classifies each frame as one of three classes: "No Squat", "Concentric", or "Eccentric"
    class_pred = to_categorical(np.argmax(squat_class.predict(sequences),axis=1), 3)
    
    
    ###########
    ##Step 2 ## Take the first frame of the video, make a prediction on the camera angle.
    ###########
    cap = cv.VideoCapture(video_path)
    
    # Read all frames
    ret, frame = cap.read()
    
    #make an orientation prediction on the 11th frame (random choice, no reason for it).
    _, orientation = camera_angle(cv.cvtColor(frame, cv.COLOR_BGR2RGB), "C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 3\\Camera Angle\\Models\\Cam Angle Model.h5")
    
    
    ###########
    ##Step 3 ## Use the camera orientation and the frame-wise classification to extract relevant joint angles for the relevant frames in
    ########### each concentric and eccentric contraction.
    
    #Obtain the frames at which the class changes. Also obtain the class corresponding to each of these changes.
    #Identify the locations of the class changes
    change = class_pred[1:]*class_pred[:-1]
    C = np.where(np.sum((change == np.array([0,0,0])), axis=1) == 3)[0]

    #Create a list representing the class of each chunk
    class_chunk = [np.argmax(class_pred[i+1]) for i in C-1]
    
    if orientation == 'Front':
        print('Front')
        Angles = features_front(video_path)
        
        #Initiate lists that will contain the array of angles for each concentric and eccentric contraction, respectively.
        concentric = []
        eccentric = []
        
        #Loop through each class chunk
        for i in range(len(class_chunk)):
            #define the beginning of the range of frames that represent the current chunk. We define the second index later
            #Note that we need to add 9 to the value value in C, this is because a prediction isn't made on the first 9 frames.
            ind0 = C[i-1]+1+9

            #if i==0 then the range of frames is from the beginning (the abouve definition for ind0 doesn't apply in this case)
            if i==0:
                ind0 = 0

            #if not a bicep curl at all
            if class_chunk[i] == 0:
                pass

            #if concentric contraction
            elif class_chunk[i] == 1:
                #if this is the final chunk
                if i==len(class_chunk)-1:
                    concentric.append(Angles[ind0:])
                else:
                    ind1 = C[i]+1+9
                    concentric.append(Angles[ind0:ind1])

            #if eccentric contraction
            elif class_chunk[i]==2:
                #if this is the final chunk
                if i==len(class_chunk)-1:
                    eccentric.append(Angles[ind0:])
                else:
                    ind1 = C[i]+1+9
                    eccentric.append(Angles[ind0:ind1])
        
        
    elif orientation == 'Side Right':
        print('Side')
        Angles = features_side_R(video_path)
        
        #Initiate lists that will contain the array of angles for each concentric and eccentric contraction, respectively.
        concentric = []
        eccentric = []
        
                #Loop through each class chunk
        for i in range(len(class_chunk)):
            #define the beginning of the range of frames that represent the current chunk. We define the second index later
            #Note that we need to add 9 to the value value in C, this is because a prediction isn't made on the first 9 frames.
            ind0 = C[i-1]+1+9

            #if i==0 then the range of frames is from the beginning (the above definition for ind0 doesn't apply in this case)
            if i==0:
                ind0 = 0

            #if not a bicep curl at all
            if class_chunk[i] == 0:
                pass

            #if concentric contraction
            elif class_chunk[i] == 1:
                #if this is the final chunk
                if i==len(class_chunk)-1:
                    concentric.append(Angles[ind0:])
                else:
                    ind1 = C[i]+1+9
                    concentric.append(Angles[ind0:ind1])

            #if eccentric contraction
            elif class_chunk[i]==2:
                #if this is the final chunk
                if i==len(class_chunk)-1:
                    eccentric.append(Angles[ind0:])
                else:
                    ind1 = C[i]+1+9
                    eccentric.append(Angles[ind0:ind1])
                    
                    
    ###########
    ##Step 4 ## Use cubic spline interpolation to split each concentric and eccentric contraction into equal length sequences of length N.
    ########### Note that I have this as a separate function above, so just call that.
    
    con_interp = equal_lengths(concentric, N)
    ecc_interp = equal_lengths(eccentric, N)
    
    
    ###########
    ##Step 5 ## Based on the camera orientation, obtain the "correct technique" model and create an array that matches the size of the 
    ########### concentric and eccentric arrays. Then subtract one from the other to acquire the signed error of the angles.
    
    if orientation == 'Front':
        #load the models
        con_model = np.load("C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 3\\Technique Modelling\\Models\\Front Concentric.npz")
        ecc_model = np.load("C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 3\\Technique Modelling\\Models\\Front Eccentric.npz")
        
        #create empty arrays of the same size as con_interp and ecc_interp
        corr_con = np.zeros_like(con_interp)
        corr_ecc = np.zeros_like(ecc_interp)
        
        #the models are dictionaries, it will be easier to index them numerically, therefore create a list of the keys. It will be the same for both models.
        keys = list(con_model.keys())
        
        #create a normalised time scale that will be used to find the values of the angles
        time = np.linspace(0,1,N)
        
        #loop through each joint angle
        for i, key in enumerate(keys):
            #extract the current coefficients for concentric and eccentric
            coeff_con = con_model[key]
            coeff_ecc = ecc_model[key]
            
            #first deal with the angular input
            if i==0:
                T = np.stack((np.ones(N),time))
                traj_con = np.arctan2(coeff_con[1]@T, coeff_con[0]@T)
                traj_ecc = np.arctan2(coeff_ecc[1]@T, coeff_ecc[0]@T)
                
            elif i == 1:
                T = np.stack((np.ones(N),time,time**2))
                traj_con = coeff_con@T
                traj_ecc = coeff_ecc@T
                
            #Now deal with all other linear models
            elif i==2:
                T = np.stack((np.ones(N),time))
                traj_con = coeff_con@T
                traj_ecc = coeff_ecc@T
             
            #Now deal with the final feature which is modelled as quadratic
            elif i == 3:
                T = np.stack((np.ones(N),time,time**2))
                traj_con = coeff_con@T
                traj_ecc = coeff_ecc@T
            
            #Now fill in the empty arrays (note that the trajectory for each repetition will be identical).
            corr_con[:,:,i] = traj_con
            corr_ecc[:,:,i] = traj_ecc
        
        #calculate the signed error and return it for concentric and eccentric
        con_err = con_interp-corr_con
        ecc_err = ecc_interp-corr_ecc
    
        return con_err, con_interp, corr_con, ecc_err, ecc_interp, corr_ecc
            
    elif orientation == 'Side Right':
        #load the models
        con_model = np.load("C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 3\\Technique Modelling\\Models\\Side Concentric.npz")
        ecc_model = np.load("C:\\Users\\Cian\\1 FYP Code\\Code_with_git\\Prototype 3\\Technique Modelling\\Models\\Side Eccentric.npz")
        
        #create empty arrays of the same size as con_interp and ecc_interp
        corr_con = np.zeros_like(con_interp)
        corr_ecc = np.zeros_like(ecc_interp)
        
        #the models are dictionaries, it will be easier to index them numerically, therefore create a list of the keys. It will be the same for both models.
        keys = list(con_model.keys())
        
        #create a normalised time scale that will be used to find the values of the angles
        time = np.linspace(0,1,N)
        
        #loop through each joint angle
        for i, key in enumerate(keys):
            #extract the current coefficients for concentric and eccentric
            coeff_con = con_model[key]
            coeff_ecc = ecc_model[key]
            
            if i==0:
                #create a T variable that will be a vector that will multiply by the coefficients. Note that this regression model is quadratic.
                T = np.stack((np.ones(N),time,time**2))

                #create an array that represents the trajectory of each joint angle for each timepoint, for concentric and eccentric.
                traj_con = np.arctan2(coeff_con[1]@T, coeff_con[0]@T)
                traj_ecc = np.arctan2(coeff_ecc[1]@T, coeff_ecc[0]@T)
            elif i==1:
                T = np.stack((np.ones(N),time))
                traj_con = coeff_con@T
                traj_ecc = coeff_ecc@T
            else:
                T = np.stack((np.ones(N),time,time**2))
                traj_con = coeff_con@T
                traj_ecc = coeff_ecc@T
                
            #Now fill in the empty arrays (note that the trajectory for each repetition will be identical).
            corr_con[:,:,i] = traj_con
            corr_ecc[:,:,i] = traj_ecc
             
        #calculate the signed error and return it for concentric and eccentric
        con_err = con_interp-corr_con
        ecc_err = ecc_interp-corr_ecc

        return con_err, con_interp, corr_con, ecc_err, ecc_interp, corr_ecc