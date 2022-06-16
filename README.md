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

## 06/01 -- update by YS
- main.py
    - Update for generate solution.txt for online server.
    - Separate the fcn from backone model.
- In script folder
    - gen_result.sh : For gerenating solution.txt (Only without label adaption)
- 
# Todos
1. Generate solution.txt for label adaption


## 06/06 -- update by YS
- [Mobilenet_v2_CA](https://github.com/Andrew-Qibin/CoordAttention)
- In models folder
    - models_mobilenet_v2_ca.py : Mobilenet_v2 with coordinate attention

## Visualization
Our work supports the visualization from [pytorch/captum](https://github.com/pytorch/captum) and [Captum/GradientShap](https://captum.ai/api/gradient_shap.html).
There are five modes you can choose,
0. Default
1. Face Silhouette
2. Eyes
3. Nose
4. Mouth
To visualize your model of an image,
```
./script/gen_visualize.sh
```