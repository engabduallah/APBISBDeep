""" This code is written by Abduallah Damash 2281772
     at 05/06/2021 for CNG483 course to present
  "Age Prediction based on Iris Biometric Data"
Middle East Technical University, All Right Saved"""
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from torch.utils.tensorboard import writer
############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs"
writer = SummaryWriter('runs/mnist1')
###################################################
import matplotlib.pyplot as plt

#Nuarla netwrok with Zero Hidden Layer
class NuralNetwoek0HL0T(nn.Module ):
    def __init__(self):
           # 0 meant type is texture
          super(NuralNetwoek0HL0T, self).__init__()
          self.fc1 = nn.Linear(9600, 4800)  # Input Layer
          self.fc4 = nn.Linear(4800, 3)     # Output Layel
      # super(NuralNetwoek, self).__init__()
      # self.fc1 = nn.Linear(9600, 4800)    #Input Layer
      # self.fc2 = nn.Linear(4800, 2400)    #First Hidden Layer
      # self.fc3 = nn.Linear(2400, 128)     #Second Hidden Layer
      # self.fc4 = nn.Linear(128, 3)        #Output Layer

    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc4(x)
      output = F.log_softmax(x.float(), dim=1)        #optimizeing the loss using log_softmax
      return output
class NuralNetwoek0HL1G(nn.Module):
  def __init__(self):
    # 1 means type is gemotric
    super(NuralNetwoek0HL1G, self).__init__()
    self.fc1 = nn.Linear(5, 4)  # Input Layer
    self.fc4 = nn.Linear(4, 3)  # Output Layer

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc4(x)
    output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
    return output
class NuralNetwoek0HL2GT(nn.Module):
  def __init__(self):
    # means type is both
    super(NuralNetwoek0HL2GT, self).__init__()
    self.fc1 = nn.Linear(9605, 4800)  # Input Layer
    self.fc4 = nn.Linear(4800, 3)  # Output Layer

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc4(x)
    output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
    return output

#Nuarla netwrok with One Hidden Layer
class NuralNetwoek1HL0T(nn.Module):
  def __init__(self):
    super(NuralNetwoek1HL0T, self).__init__()
    self.fc1 = nn.Linear(9600, 4800)  # Input Layer
    self.fc2 = nn.Linear(4800, 2400)  # Hiddein Layer #1
    self.fc4 = nn.Linear(2400, 3)  # Output Layer
  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc4(x)
    output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
    return output
class NuralNetwoek1HL1G(nn.Module):
  def __init__(self):
    # 1 means type is gemotric
    super(NuralNetwoek1HL1G, self).__init__()
    self.fc1 = nn.Linear(5, 4)  # Input Layer
    self.fc2 = nn.Linear(4, 3)  # Hidden Layer #1
    self.fc4 = nn.Linear(3, 3)  # Output Layer

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc4(x)
    output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
    return output
class NuralNetwoek1HL2GT(nn.Module):
  def __init__(self):
    super(NuralNetwoek1HL2GT, self).__init__()
    self.fc1 = nn.Linear(9605, 4800)  # Input Layer
    self.fc2 = nn.Linear(4800, 2400)  # Hidden layer #1
    self.fc4 = nn.Linear(2400, 3)  # Output Layer

  def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc4(x)
      output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
      return output

#Nuarla netwrok with Two Hidden Layer

#Nuarla netwrok with Two Hidden Layer
class NuralNetwoek2HL0T(nn.Module):
    def __init__(self):
      # 0 meant type is texture
      super(NuralNetwoek2HL0T, self).__init__()
      self.fc1 = nn.Linear(9600, 4800)  # Input Layer
      self.fc2 = nn.Linear(4800, 2400)  # Hidden Layer #1
      self.fc3 = nn.Linear(2400, 128)  # Hidden Layer  #2
      self.fc4 = nn.Linear(128, 3)  # Output Layer

    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      x = F.relu(x)
      x = self.fc4(x)
      output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
      return output
class NuralNetwoek2HL1G(nn.Module):
  def __init__(self):
    super(NuralNetwoek2HL1G, self).__init__()
    self.fc1 = nn.Linear(5, 5)  # Input Layer
    self.fc2 = nn.Linear(5, 4)  # Hidden Layer  #1
    self.fc3 = nn.Linear(4, 3)  # Hidden Layer  #2
    self.fc4 = nn.Linear(3, 3)  # Output Layer


  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    x = self.fc3(x)
    x = F.relu(x)
    x = self.fc4(x)
    output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
    return output
class NuralNetwoek2HL2GT(nn.Module):
    def __init__(self):
      super(NuralNetwoek2HL2GT, self).__init__()
      self.fc1 = nn.Linear(9605, 4800)  # Input Layer
      self.fc2 = nn.Linear(4800, 2400)  # Hidden Layer #1
      self.fc3 = nn.Linear(2400, 128)  # Hidden Layer #2
      self.fc4 = nn.Linear(128, 3)  # Output Layer
    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      x = F.relu(x)
      x = self.fc4(x)
      output = F.log_softmax(x.float(), dim=1)  # optimizeing the loss using log_softmax
      return output

class Dataloder(Dataset):
  def __init__(self,features_,labels_):
    self.features_ = torch.from_numpy(features_)
    self.labels_ = torch.from_numpy(labels_)
    self.n_samples = features_.shape[0]
    print('LOADED:',self.n_samples)

  def __len__(self):
    return self.n_samples

  def __getitem__(self, index):
    return self.features_[index], self.labels_[index]

def acquireGeometrictxt (filename):
  # Acquire Geometric Features and spilt into each array for each class
  openfile = open (filename, 'r')
  features = openfile.read().replace("\n", " ").replace(",", " ").split(" ")
  features = np.array(features[19:])      #Resize the array to ignore the written words
  features = list(map(float, features))   #Transfer it from a string to float numbers
  featureClass =[]
  lableClass =[]
  # spelt the features into arry N X 5 for each class
  for i in range(5, len(features), 6):
    if (features[i] == 1):
      featureClass.append(features[i - 5:i])
      lableClass.append(0)
    elif (features[i] == 2):
      featureClass.append(features[i - 5:i])
      lableClass.append(1)
    else:
      featureClass.append(features[i - 5:i])
      lableClass.append(2)
  featureClass = np.array(featureClass)
  lableClass = np.array(lableClass)
  print('Geometric Feature Array shape is',featureClass.shape, 'Geometric Labal Array shape is', lableClass.shape)
  return featureClass, lableClass

def acquireTexturetxt (filename):
  # Acquire Texture Features and spilt into each array for each class
  openfile = open (filename, 'r')
  features = openfile.read().replace("\n", " ").replace(",", " ").split(" ")
  features = np.array(features[19209:])      #Resize the array to ignore the written words
  features = list(map(int, features))        #Transfer it from a string to float numbers
  featureClass = []
  lableClass = []
  # spelt the features into arry N X 5 for each class
  for i in range(9600, len(features), 9601):
    if (features[i] == 1):
      featureClass.append(features[i - 9600:i])
      lableClass.append(0)
    elif (features[i] == 2):
      featureClass.append(features[i - 9600:i])
      lableClass.append(1)
    else:
      featureClass.append(features[i - 9600:i])
      lableClass.append(2)
  # Transfer them into array
  featureClass = np.array(featureClass)
  lableClass = np.array(lableClass)
  print('Texture Feature Array shape is', featureClass.shape, 'Texture Labal Array shape is', lableClass.shape)
  return featureClass, lableClass

def mergetowArray (array1, array2, array1labal, array2lable):
  # This Function see if the label for each input is the same for each array of feature,
  # it will merge them toghther as a new array
  global mergeArray, mergeArrayLabal2D, mergeArrayLabal1D
  equal = 0
  noteual = 0
  for i in range(0, len(array1labal)):
    if (array1labal[i] == array2lable[i]):
      equal += 1
    else:
      noteual += 1
      print('here you have conflict',i, array1labal[i], array2lable[i])
  if (noteual ==0):
    mergeArray = np.column_stack((array1, array2))
    mergeArrayLabal2D = np.column_stack((array1labal, array2lable))
    mergeArrayLabal1D = array1labal
  return mergeArray, mergeArrayLabal2D, mergeArrayLabal1D

def modeltype_hiddenlayer (NumHiddenL, FeatureType):
  if (NumHiddenL==0 and FeatureType==0):
    model = NuralNetwoek0HL0T()
  elif (NumHiddenL==0 and FeatureType==1):
    model = NuralNetwoek0HL1G()
  elif (NumHiddenL==0 and FeatureType==2):
    model = NuralNetwoek0HL2GT()
  elif (NumHiddenL==1 and FeatureType==0):
    model = NuralNetwoek1HL0T()
  elif (NumHiddenL==1 and FeatureType==1):
    model = NuralNetwoek1HL1G()
  elif (NumHiddenL==1 and FeatureType==2):
    model = NuralNetwoek1HL2GT()
  if (NumHiddenL==2 and FeatureType==0):
    model = NuralNetwoek2HL0T()
  elif (NumHiddenL==2 and FeatureType==1):
    model = NuralNetwoek2HL1G()
  elif (NumHiddenL==2 and FeatureType==2):
    model = NuralNetwoek2HL2GT()
  return model

def trainModeal (TrainArray, TrainLabal, NumHiddenL, FeatureType, epochs,batch):
  # Decide on Number of hidden layer nad feature type
  model = modeltype_hiddenlayer(NumHiddenL,FeatureType)
  # Model Parameter for Training the Nural Network #
  opti = optim.RMSprop(model.parameters(), lr=0.0001)
  criteria = nn.CrossEntropyLoss()
  # Hyper Parmeter #
  num_epochs = epochs
  batch_size = batch
  # Seeting the Dataset and loaded to be ready for Training
  train_Tex_loader = Dataloder(TrainArray, TrainLabal)
  # Resizing the Dataset
  train_loader = DataLoader(dataset=train_Tex_loader,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
  # Setting The paramter for training each Epoch
  total_samples = len(train_loader)
  n_iterations = math.ceil(total_samples / batch_size)
  print(f'Total Samples Data {total_samples},Epcoh {num_epochs}, Batch Size {batch_size}, Number of Iteration {n_iterations}')
  print(f'Hidden Layer {NumHiddenL}, Feature Type {FeatureType} [0 Texture, 1 Geometric, 2 both')
  LOSS_HISTORY = []
  ACC_HISTORY = []
  for epoch in range(num_epochs):
    COUNTER = 0
    correct = 0
    total_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
      train_input, train_label = Variable(features), Variable(labels)

      train_output = model(train_input.float())
      loss = criteria(train_output.float(), train_label.long())

      opti.zero_grad()
      loss.backward()
      opti.step()
      total_loss += loss.item()
      _, predicted = torch.max(train_output.data, 1)

      correct += (predicted == train_label).sum().item()
      COUNTER += batch_size

    total_loss = total_loss / COUNTER
    correct = correct / COUNTER
    print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss}, Acc: {correct}')

    LOSS_HISTORY.append(total_loss)
    ACC_HISTORY.append(correct)
  # Plot the accercy and loss and save them
  plt.plot(range(epochs), LOSS_HISTORY)
  plt.savefig('loss_'+str(uuid.uuid4())+'.png')
  plt.figure()
  plt.plot(range(epochs), ACC_HISTORY)
  plt.savefig('acc_'+str(uuid.uuid4())+'.png')
  return 1

def testModeal (TestArray,TestLabal, NumHiddenL, FeatureType, epochs,batch) :
  # Test the model
  # In test phase, we don't need to compute gradients (for memory efficiency)
  class_labels = []
  class_preds = []
  # Decide on Number of hidden layer nad feature type
  model = modeltype_hiddenlayer(NumHiddenL,FeatureType)
  # Seeting the Dataset and loaded to be ready for Testing
  Test_Tex_loader = Dataloder(TestArray, TestLabal)
  # Resizing the Dataset
  Test_loader = DataLoader(dataset=Test_Tex_loader,
                            batch_size=batch,
                            shuffle=True,
                            num_workers=0)
  with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for i, (features, labels) in enumerate(Test_loader):
        test_input, test_label = Variable(features), Variable(labels)
        test_output = model(test_input.float())
        values, predicted = torch.max(test_output.data, 1)
        n_samples += test_label.size(0)
        n_correct += (predicted == labels).sum().item()
        class_probs_batch = [F.softmax(output, dim=0) for output in test_output]
        class_preds.append(class_probs_batch)
        class_labels.append(predicted)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 456 test images with  number of Hidden layers {NumHiddenL}: {acc} %')

    # # Plot the accercy and loss and save them
    # plt.plot(range(epochs), class_preds)
    # plt.savefig('class_preds_' + str(uuid.uuid4()) + '.png')
    # plt.figure()
  return 1

# Load All the subjects of Data:
TrainGem, TrainGemlable = acquireGeometrictxt ('IrisGeometicFeatures_TrainingSet.txt')
TrainTex, TrainTexlable = acquireTexturetxt ('IrisTextureFeatures_TrainingSet.txt')
TestGem, TestGemlable = acquireGeometrictxt ('IrisGeometicFeatures_TestingSet.txt')
TestTex, TestTexlable = acquireTexturetxt ('IrisTextureFeatures_TestingSet.txt')
TrainGemTexClass, TrainGemTexlable2D, TrainGemTexlable = mergetowArray(TrainGem,TrainTex, TrainGemlable, TrainTexlable)
TestGemTexClass, TestGemTexlable2D, TestGemTexlable = mergetowArray(TestGem,TestTex, TestGemlable, TestTexlable)

# Training The Geometric Feature:
#trainModeal(TrainGem,TrainGemlable, 0, 1 ,15,5)
#trainModeal(TrainGem,TrainGemlable, 1, 1 ,15,5)
#trainModeal(TrainGem,TrainGemlable, 2, 1 ,15,5)
# Training The Texteure Features:
# trainModeal(TrainTex,TrainTexlable, 0, 0 ,10,10)
# trainModeal(TrainTex,TrainTexlable, 1, 0 ,10,10)
# trainModeal(TrainTex,TrainTexlable,2, 0 ,10,10)
# Training The Geometric and Texteure Features:
# trainModeal(TrainGemTexClass,TrainGemTexlable, 0, 2 ,10,10)
# trainModeal(TrainGemTexClass,TrainGemTexlable, 1, 2 ,10,10)
# trainModeal(TrainGemTexClass,TrainGemTexlable,2, 2 ,10,10)
#
# testModeal(TestGemTexClass,TestGemTexlable,2, 2 ,10,10)
trainModeal(TestGemTexClass,TestGemTexlable,2, 2 ,10,10)

# trainModeal(TestTex,TestTexlable, 0, 0 ,10,10)
# trainModeal(TestTex,TestTexlable, 1, 0 ,10,10)
# trainModeal(TestTex,TestTexlable, 2, 0 ,10,10)

### Testing The Geometric Feature:###
# testModeal(TestGem,TestGemlable, 0, 1 ,15,5)
# testModeal(TestGem,TestGemlable, 1, 1 ,15,5)
# testModeal(TestGem,TestGemlable, 2, 1 ,15,5)
# ### Training The Texteure Features: ###
# testModeal(TestTex,TestTexlable, 0, 0 ,10,10)
# testModeal(TestTex,TestTexlable, 1, 0 ,10,10)
# testModeal(TestTex,TestTexlable,2, 0 ,10,10)
### Training The Geometric and Texteure Features:###
# testModeal(TestGemTexClass,TestGemTexlable, 0, 0 ,10,10)
# testModeal(TestGemTexClass,TestGemTexlable, 1, 0 ,10,10)
# testModeal(TestGemTexClass,TestGemTexlable,2, 0 ,10,10)


#### End of THe program, HOPE it serves its porupose, BYE BYE ####
