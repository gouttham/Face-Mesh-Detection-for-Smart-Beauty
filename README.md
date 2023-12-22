# Face-Mesh-Detection-for-Smart-Beauty

<div align="center">
<p>
<img src="./readme_stuffs/demo.gif" width="600" height="400"/>
</p>
<br>
</div>

## Introduction

This repository contains the python code utilizing Face Landmark detection using the Mediapipeline. Kindly refer [here](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python) for the documentation.

## Installation

This repository contains setup.sh that will install the virtual enviroinment and install all the required dependencies.
Run the below commands.
```
. setup.sh
```

## Inference

To run the face landmark detection

```
python run.py
```

## Settings

 S.no     | Name  | Description                                                     |
----------|-------|-----------------------------------------------------------------|
1. | Mesh     | Enable/Disable Mesh                                             |                                            
2. | Iris     | Enable/Disable Iris                                             |                                             
3. | Eyes     | Enable/Disable Eyes                                             |                                            
4. | Eyebrows | Enable/Disable Eyebrows                                         |                                        
5. | Lips     | Enable/Disable Lips                                             |                                            
6. | Outline  | Enable/Disable Face Contour                                     |                                    
7. | Iris_dist | Enable/Disable the distance between the Iris                    |                   
8. | Sketch   | Enable/Disable the video feed                                   |                   
9. | Record   | Records the video and the output is stored in `./videos/*.mp4`  | 