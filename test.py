# inference on validation
import torch
from dataset import LandmarksDataset_validation
from model import Model

def main(model_path,batch_size=32):
    #-----------------------load model---------------------------------
    device = torch.device("cuda:0")
    print(device)
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path))
    # remember that you must call model.eval() to set dropout and batch 
    # normalization layers to evaluation mode before running inference.
    model.eval()
    #-----------------------prepare ds---------------------------------
    validationset = LandmarksDataset_validation(root_dir="dataset/validation_FAN")
    validation_size = len(validationset)
    validation_loader = torch.utils.data.DataLoader(validationset,batch_size=batch_size,shuffle=False,num_workers=4)
    with open(target_path, "w") as f:
        for i,data in enumerate(validation_loader,0):
            inputs = data[0].to(device)
            # some batch are less than batch_size
            outputs = model(inputs.float()).reshape([-1,106,3])
            # go over the batch
            # every batch is not necessarily = batch_size
            for idx,sample in enumerate(data[1]):
                person_emotion = sample
                person = person_emotion[:5]
                # xx or other-x
                emotion = person_emotion[6:]
                f.write("subject_"+person+"/"+"expression_"+emotion+"\n")
                outputs_sample = outputs[idx].tolist()
                for j in range(106):
                    f.write(str(outputs_sample[j][0])+", "+str(outputs_sample[j][1])+", "+str(outputs_sample[j][2])+"\n")
    f.close()
        
if __name__=="__main__":
    model_path = "regression_net.pth"
    target_path = "submission.txt"
    main(model_path)
