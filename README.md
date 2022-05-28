# CV2022Final_Face_Landmark_Detection
---
## 05/23 -- update by YS
- [Reference](https://github.com/deepinsight/insightface/tree/master/alignment/synthetics/datasets)
    - In src folder
        - dataset.py : Prepare data for training
        - aug.py : Do border augmentation
    - main.py
        - Setting for training or testing
- Todo:
    - Test part in main.py  

## 05/26 -- update by YS
- [Reference](https://github.com/deepinsight/insightface/tree/master/alignment/synthetics/datasets)
    - In src folder
        - dataset.py : Reorder the Augmentation. Do resize first.
    - main.py
        - Replace Tensorboardlogger with wandb
- Todo:
    - Test part in main.py  

## 05/27 -- update by YS
- [Reference](https://github.com/deepinsight/insightface/tree/master/alignment/synthetics/datasets)
    - In src folder
        - dataset.py
            - When testing, return original image.
            - Coordinate enhancement
    - main.py
        - Finish testing part
- In script folder
    - train.sh : simple training script
- In models folder
    - models_select : You can add entrypoints of your models 

## 05/28 -- update by YS
- [Reference](https://github.com/deepinsight/insightface/tree/master/alignment/synthetics/datasets)
    - main.py
        - Add adaption training and testing
        - Improve the presentation of testing result
- In script folder
    - train.sh : Fix some bug
    - adapt.sh : For training of label adaption
- In models folder
    - models_select.py : Add endtrypoint for label adaption