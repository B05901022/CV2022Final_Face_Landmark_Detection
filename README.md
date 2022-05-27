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
        - dataset.py : When testing, return original image.
    - main.py
        - Finish testing part
