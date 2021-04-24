# process images into 68 3D landmarks and save
import skimage
import numpy as np
import face_alignment

class FAN:
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

    def __call__(self,dir_path):
        image_path = dir_path+"t.jpg"
        input = skimage.io.imread(image_path)
        preds = self.fa.get_landmarks(input)
        preds = np.array(preds)
        preds = np.squeeze(preds)
        if preds.shape[0]>1:
            #more than 1 person,abandon
            return False
        # prevent generate empty detection
        if preds == None:
            return False
        np.savetxt(dir_path+'landmarks.csv', preds, delimiter=',')
        print("successfully stored "+dir_path+'landmarks.csv')
        return True
