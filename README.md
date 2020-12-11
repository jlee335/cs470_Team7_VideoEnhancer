# cs470_Team7_VideoEnhancer

###### Course: CS470 
###### Team: Team 7
###### Members: 20170151 김재현, 20170432 윤웅노, 20170831 이현재
###### Language: Python (Pytorch)


#Requirements 
This code is specialized on Google Colab environment. To re-create our results, please clone this repository to Google Drive.

#File Structure
cs470_Team7_VideoEnhancer/
├── Data/
│   ├── Test_Data/
│   │   └──  ** Add constant-frame encoded mp4 files to be used as validation data set.
│   │
│   └── Train_Data/
│       └──  ** Add constant-frame encoded mp4 files to be used as training data set.
│    
│  
├── Models/  All four ipynb files contain code for the models and their corresponding training codes.       
│   └──  ** Intermediate and Final trained models are stored here. We have provided our own trained model by default.
|   
├── Test/
|    ├── Before/
|    |   └── ** Add video you want to increase quality here.
|    └── After/
|        └── ** High-quality video will appear here after code run.
|                       
├── Attempt1.ipynb              
├── Attempt2.ipynb              
├── Attempt3.ipynb   
├── Frame_interpolation.ipynb  
|
├── improve_framerate.ipynb   <- This notebook increases framerate by frame_interp network.
└── improve_resolution.ipynb  <- This notebook increases resolution using SR network.
  
#Training
  1. Add .mp4 files to Train_data in order to increase training dataset. 
  2. Open Models folder and open .ipynb file which you like to train
  3. Run the ipynb folder and train the model.

#Converting Video
  1. Place mp4 file you want to convert to in 'Test/Before/' folder
  1. Run improve_framerate or improve_resolution ipynb
  2. Result video file will be outputted in 'Test/After/' folder.




