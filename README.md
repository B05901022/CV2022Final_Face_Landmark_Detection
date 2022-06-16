# CV2022Final_Face_Landmark_Detection
## Outline
- [Preparing](#Preparing)
- [Image resoultion](#Image_resoultion)
- [Final setup](#setup)
- [Script](#Script)
- [Taining](#Training)
- [Testing](#Testing)
- [Average models](#Average)
- [Generate solution.txt](#Generate)
- [Inference one image](#Inference_one)
- [Visualization](#Visualization)

<h2 id = "Preparing"> Preparing </h2>

### Enviorment

- python 3.7

<br>

### Clone this repository

```
$ git clone https://github.com/B05901022/CV2022Final_Face_Landmark_Detection.git
$ cd CV2022Final_Face_Landmark_Detection
```
<br>

### Install requirements
```
pip install -r requirements.txt
```
<br>

### File directory

Under the path you save your dataset, you should make sure that you have included these folders
```
CV2022Final_Face_Landmark_Detection/
└─── main.py
└─── ...

data/
└─── synthetics_train/
        └─── annot.pkl
└─── aflw_val/
        └─── annot.pkl
└─── aflw_test/
```
<br>

### Wandb

Login your wandb.
Enter this command in shell
```
wandb login
```
In **main.py** change your project and entity
![image alt](./pic/1.png) <br>
<br>

<h2 id = "Image_resoultion"> Image resoultion </h2>

In **main.py**, change the image resolution,
```
input_resolution=384 
```
<br>

<h2 id = "setup"> Final setup </h2>

<table>
  <tr style=" border-top: 1px solid white;">
    <th style="text-align:center">Backbone</th>
    <th style="text-align:center">Loss</th>
    <th style="text-align:center">Optimizer</th>
    <th style="text-align:center">Learning Rate</th>
    <th style="text-align:center">Weight Decay</th>
    <th style="text-align:center">Momentum</th>
    <th style="text-align:center">LR Scheduler</th>
    <th style="text-align:center">Epoch</th>
  </tr>
  <tr style=" border-bottom: 1px solid white;">
    <td style="text-align:center">MobileNetV2(1.25)</td>
    <td style="text-align:center">L1</td>
    <td style="text-align:center">SGD</td>
    <td style="text-align:center">0.01</td>
    <td style="text-align:center">0.00001</td>
    <td style="text-align:center">0.9</td>
    <td style="text-align:center">Disable</td>
    <td style="text-align:center">240</td>
  </tr>
</table>
<br>

<h2 id = "Script"> Script </h2>

```
<script Directory>  
└─── train.sh
└─── test.sh
└─── gen_result.sh
└─── gen_visualize.sh
└─── inference_one.sh
└─── adapt.sh
└─── adapt_test.sh
```
<br>

<h2 id = "Training"> Training </h2>

Enter the command
```
./script/train.sh
```

- In **train.sh**
    - Ensure that setting is what you want
    - There are some flag you can use for training
        - `--use_sam` 
            - optimizer + SAM
        - `--use_swa` 
            - Use SWA
        - `--cood_en` 
            - Coordconv
        - `--lr_nosch`
            - Disable learning scheduler
<br>

<h2 id = "Testing"> Testing </h2>

Enter the command
```
./script/test.sh
```

- In **test.sh**
    - Ensure that backone is the same as training <br>
    - Select the ckpt you want to test <br>
    ![image alt](./pic/2.png) <br>

    - There are some flag you can use for testing
        - `--cood_en`
            - Coordconv (Only for Coordconv)
        - `--use_shift`
            - Shift the image along x-axis, y-axis and average the all detections
<br>

<h2 id = "Average"> Average models </h2>

Enter the command
```
python soup.py
```

- In **soup.py** 
    - select the chekpoints you want to use
![image alt](./pic/4.png) <br>

    - They should come from smae backbone model

<br>

<h2 id = "Generate"> Generate solution.txt </h2>

Enter the command
```
./script/gen_result.sh
```

- In **gen_result.sh**
    - Ensure that backone is the same as training <br>
    - Select the ckpt you want to generate solution <br>

    - There are some flag you can use for testing
        - `--use_shift`
            - Shift the image along x-axis, y-axis and average the all detections

        **<font color=#FF0000>※ gen_result do not support Coordconv </font>**
<br>

<h2 id = "Inference_one"> Inference and plot one image </h2>

Enter the command
```
./script/inference_one.sh
```

- In **inference_one.sh**
    - Select the ckpt you want to generate solution <br>
    - Specify the image you want to inference <br>
    - Remember giving the file name for result <br>

    ![image alt](./pic/3.png) <br>

    **<font color=#FF0000>※ Inference_one do not support Coordconv and 25_shift iamges </font>**

<br>

<h2 id = "Visualization"> Visualization </h2>

To visualize your model of an image,
```
./script/gen_visualize.sh
```

Our work supports the visualization from [pytorch/captum](https://github.com/pytorch/captum).
There are five modes you can choose,  

- In **gen_visualize.sh**
    0. Default
    1. Face Silhouette
    2. Eyes
    3. Nose
    4. Mouth

**<font color=#FF0000>※ This operation requires a lot of RAM.</font> It is better to use cpu rather than GPU and makce sure that <font color=#FF0000>RAM > 50 GB </font>is available.**
