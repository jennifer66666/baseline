import os
import numpy as np
from .preprocess import FAN
from plyfile import PlyData, PlyElement

class OnePersonOneEmotion:
    def __init__(self,person_emotion,prefix,test=False):
        self.person_emotion = person_emotion
        self.fa = FAN()
        self.input_output(prefix,test)

    def input_output(self,prefix,test):
        path = prefix+self.person_emotion
        r,f,l,stack = self.three_angles(path)
        # only train data have labels
        if not test:
            txt_path = path+"/Images/fusion/nicp_106_pts.txt"
            label_landmarks = np.loadtxt(txt_path, delimiter=' ')
            np.savetxt(path+"/Images/"+'/label_landmarks.csv', label_landmarks, delimiter=',')

    def three_angles(self,path):
        r_landmarks,f_landmarks,l_landmarks,stack = None,None,None,None
        three_angles_path = path+"/Images/"#0/t.jpg"
        # if there is no a .csv file beside the image
        if len(os.listdir(three_angles_path+"0"))<2:
            #preprocess the img to output .ply file and save it beside the img
            r = self.fa(three_angles_path+'0/')
        else:
            # already have csv file
            r = np.loadtxt(three_angles_path+'0/'+'landmarks.csv', delimiter=',')
        if len(os.listdir(three_angles_path+"1"))<2:
            f = self.fa(three_angles_path+"1/")
        else:
            f = np.loadtxt(three_angles_path+'1/'+'landmarks.csv', delimiter=',')
        if len(os.listdir(three_angles_path+"2"))<2:
            l = self.fa(three_angles_path+"2/")
        else: 
            l = np.loadtxt(three_angles_path+"2/"+'landmarks.csv', delimiter=',')
        # readin .ply and convert to 68x3 np
        if (r is not None) and (f is not None) and (l is not None):
            stack = np.concatenate([r,f,l],axis=1)
        if stack is not None:
            np.savetxt(path+'/Images/stack_landmarks.csv', stack, delimiter=',')
        return r,f,l,stack

    def read_csv(self,person_path,emotion):
        file_path = person_path+"/"+emotion+"/Images/"
        file_path_right = file_path + "0/landmarks.csv"
        file_path_front = file_path + "1/landmarks.csv"
        file_path_left = file_path + "2/landmarks.csv"
        # might be warnings when empty data
        r = np.loadtxt(file_path_right, delimiter=',')
        f = np.loadtxt(file_path_front, delimiter=',')
        l = np.loadtxt(file_path_left, delimiter=',')
        return r,f,l

if __name__ == "__main__":
    prefix = "dataset/validation_arrange/"
    test = True
    person_emotions = [i for i in os.listdir(prefix)]
    for person_emotion in person_emotions:
        one_person = OnePersonOneEmotion(person_emotion,prefix,test=test) 
    


