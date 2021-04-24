# arrange train process
# input person,emotion:stack_landmarks 68x9
# output same_person,same emotion:landmarks 106x3
from datasets import *
from model import *
import torch
import torch.optim as optim
import random

def main(split,batch_size,epochs):
    #--------------------------------train config---------------------------------------
    # CAUTION: readin all people object at once will casue OOM
    # so we seperate batch indexes first; and make object later inside batch
    # for regression
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    print(device)
    #---------------------------------prepare ds-----------------------------
    all_people_index = os.listdir("dataset/train_bakup/")
    num_all_people = len(all_people_index)
    for epoch in range(epochs):
        running_loss = 0.0
        # random.shuffle() changes the x list in place
        random.shuffle(all_people_index)
        ds_train = all_people_index[:int(num_all_people*split)]
        ds_test = all_people_index[int(num_all_people*split):]
        batches_train = seperate_batch(ds_train,batch_size)
        batches_test = seperate_batch(ds_test,batch_size)
        for batch_idx,batch_train in enumerate(batches_train):
            batch_people = []
            # zero the parameter gradients every batch
            optimizer.zero_grad()
            for person_index in batch_train:
                batch_people.append(OnePerson(person_index,"dataset/train/"))
            # take each person's each emotion's landmarks as one sample
            labels = []
            inputs = []
            for person in batch_people:
                for emotion in person.input_output:
                    if emotion["stacked_landmarks"] is None:
                        # if there is problem to landmark a pic, we skip
                        continue
                    else:
                        inputs.append(emotion["stacked_landmarks"])
                        labels.append(emotion["label_landmarks"])
            inputs = torch.FloatTensor(inputs).to(device)
            labels = torch.FloatTensor(labels).to(device)
            #-------------------------start train one batch-----------------------------
            # forward + backward + optimize
            # input must be tensor rather than np.ndarray
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            #-------------------------one batch finish----------------------------------
            # print out every 100 batches
            if batch_idx == 100:
                # an epoch finish
                print('[%d, %5d] 100-batches loss: %.3f' %
                    (epoch + 1, batch_idx + 1, running_loss / 2000))
        # one epoch finish
        print('[%d] epoch loss: %.3f' %
            (epoch + 1, running_loss / 2000))
    print('Finished Training')

def seperate_batch(ds,batch_size):
    batches = []
    for i in range(len(ds)//batch_size):
        if (i+1)*batch_size<len(ds):
            batches.append(ds[i*batch_size:(i+1)*batch_size])
        else:
            batches.append(ds[i*batch_size:])
    return batches

if __name__ == "__main__":
    split = 0.8
    batch_size = 32
    epochs = 30
    main(split,batch_size,epochs)