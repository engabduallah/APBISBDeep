# APBISBDeep
![image](https://user-images.githubusercontent.com/87785000/126638319-ed14d9b4-b687-4ec3-b871-fe6bb7122316.png)

This Project is done by Eng. Abduallah Damash, and assigned to me by Asst. Prof. Dr. Meryem Erbilek as part of CNG483 (Int. to Comp. Vision) course.

If you have any issues or comments on the project, contact me on Linkedin (https://www.linkedin.com/in/engabduallah). I also provided the dataset that I used in this project so that you can try it by yourself.

Insight about the project:

Implement an age prediction system based on fully-connected neural networks (NN) with rectified linear unit (ReLU) as nonlinearity 
function between layers and train it with RMSprop optimizer using the provided feature vectors, and useing softmax (cross-entropy loss) function to minimize the difference between actual age group and the estimated one. Then, evaluate it with a BioSecure Multimodal Database (BMDB) that consists of 200 subjects includes four eye images (two left and two right) for people within the age range of 18-73. Using a high-level language program, particularly Python, and common libraries such as PyTorch, OpenCV, Matplotlib, Pandas, and Numpy.

Dataset: 

The commercially available data Set 2 (DS2) of the BioSecure Multimodal Database (BMDB) is utilised 
for this project. Four eye images (two left and two right) were acquired in two different sessions with 
a resolution of 640*480 pixels. The 200 subjects providing the samples contained in this database are 
within the age range of 18-73. 

The training and the testing sets were formed to be person-disjoint sets. Approximately 72% of the 
subjects in each age group are used for training and the remaining subjects used as a testing set. The 
available number of subjects in the testing and the training sets for each age group is shown in the 
following Table.

![image](https://user-images.githubusercontent.com/87785000/126638009-dc261dad-0329-4e24-9e37-ff347b5619a0.png)

For this project three different types of iris biometric features will be used: 

• Texture features: 

These are features which describe the pattern of the iris available only 
from the overall finished output of the acquisition, segmentation, normalisation and feature 
extraction process respectively.

• Geometric features: 

These are features which describe the shape (physical appearance) of 
the iris, and are thus available only from the output of the acquisition and segmentation 
process respectively.

• Both geometric and texture features: 

simply is the combination (concatanation) of both 
feature types.

First two types of features are given to you in a seperate text files. You need to read features 
from these files and also form the third type feature set. File description is as follows:

![image](https://user-images.githubusercontent.com/87785000/126638147-6583505a-4c93-447c-8de9-4616da0e044a.png)

Enjoy it.

All rights saved, if you want to use any piece of the code, mentioning my name will be enough.
