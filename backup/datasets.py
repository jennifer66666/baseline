# readin landmarks and labels
# stack landmarks from three different angles and give weight
# stacked landmakrs regress to labels
import os
import numpy as np
from preprocess import FAN
from plyfile import PlyData, PlyElement

class OnePerson:
    def __init__(self,person,prefix):
        self.person = person
        self.fa = FAN()
        self.input_output = self.input_output_one_person(prefix)

    def input_output_one_person(self,prefix):
        one_person = []
        person_path = prefix+self.person
        emotions =[e for e in os.listdir(person_path) if '0'<=e[0]<='9']
        for emotion in emotions:
            r,f,l,stack = self.three_angles(person_path,emotion)
            txt_path = person_path+"/"+emotion+"/Images/fusion/nicp_106_pts.txt"
            label_landmarks = np.loadtxt(txt_path, delimiter=' ')
            one_person.append({'person':self.person,'emotion':emotion,'r_landmarks':r,'f_landmarks':f,'l_landmarks':l,"stacked_landmarks":stack,"label_landmarks":label_landmarks})
        return one_person

    def three_angles(self,person_path,emotion):
        r_landmarks,f_landmarks,l_landmarks,stack = None,None,None,None
        three_angles_path = person_path+"/"+emotion+"/Images/"#0/t.jpg"
        # if there is no a .csv file beside the image
        if len(os.listdir(three_angles_path+"0"))<2:
            #preprocess the img to output .ply file and save it beside the img
            r_success = self.fa(three_angles_path+"0/")
        else:
            # already have csv file
            r_success = True
        if len(os.listdir(three_angles_path+"1"))<2:
            f_success = self.fa(three_angles_path+"1/")
        else:
            f_success = True
        if len(os.listdir(three_angles_path+"2"))<2:
            l_success = self.fa(three_angles_path+"2/")
        else: 
            l_success = True
        # readin .ply and convert to 68x3 np
        if r_success and f_success and l_success:
            r_landmarks,f_landmarks,l_landmarks = self.read_csv(person_path,emotion)
            stack = np.concatenate([r_landmarks,f_landmarks,l_landmarks],axis=1)
            #else:
                # if there is problem detecting landmarks, we abandon it 
                #stack = None

        return r_landmarks,f_landmarks,l_landmarks,stack

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
    one_person = OnePerson("00194","dataset/try/") 
    # for some persons, emotions are less then seven
    assert len(one_person.input_output) == 7
    assert one_person.input_output[0]["r_landmarks"].shape == (68,3)
    assert one_person.input_output[5]["f_landmarks"].shape == (68,3)
    assert one_person.input_output[3]["l_landmarks"].shape == (68,3)
    assert one_person.input_output[3]["stacked_landmarks"].shape == (68,9)
    assert one_person.input_output[0]["label_landmarks"].shape == (106,3)


