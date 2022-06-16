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
- CUDA 10.1

### Clone this repository

```
$ git clone https://github.com/B05901022/CV2022Final_Face_Landmark_Detection.git
$ cd CV2022Final_Face_Landmark_Detection
```

### Install requirements
```
pip install -r requirements.txt
```

**<font color=#FF0000>※ Check torch.version.cuda is same as that from nvidia-smi </font>** <br>
If you have any problem, you can refer to [here](https://pytorch.org/get-started/previous-versions/)

### File directory

The default file directory is as follows, which can be modified to customized directory in train.sh and test.sh.
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

### Wandb

Log in your wandb.
Enter this command in shell
```
wandb login
```
In **main.py** change your project and entity
![image alt](./pic/1.png)
<h2 id = "Image_resoultion"> Image resoultion </h2>

To change the image resolution for faster training, please modify main.py as follows:
```
input_resolution=384 
```

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

<h2 id = "Script"> Script </h2>

```
scrip/
└─── train.sh
└─── test.sh
└─── gen_result.sh
└─── gen_visualize.sh
└─── inference_one.sh
└─── adapt.sh
└─── adapt_test.sh
```

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
            - Enable SWA
        - `--cood_en` 
            - Coordconv
        - `--lr_nosch`
            - Disable learning scheduler

<h2 id = "Testing"> Testing </h2>

Enter the command
```
./script/test.sh
```

- In **test.sh**
    - Ensure that backone is the same as training 
    - Select the checkpoint you want to test <br>
    ![image alt](./pic/2.png)
    - There are some flag you can use for testing
        - `--cood_en`
            - Need to be enabled if `--cood_en` flag is enabled in training
        - `--use_shift`
            - Shift the image along x-axis, y-axis and average the all detections

<h2 id = "Average"> Average models </h2>

Enter the command
```
python soup.py
```

- In **soup.py** 
    - Add the checkpoints to be averaged in MODEL_checkpointS list<br>
    ![image alt](./pic/4.png)
    - Ensure all selected come from smae backbone model


<h2 id = "Generate"> Generate solution.txt </h2>

Enter the command
```
./script/gen_result.sh
```
- In **gen_result.sh**
    - Ensure that backone is the same as training
    - Select the checkpoint you want to generate solution
    - There are some flag you can use for testing
        - `--use_shift`

    **<font color=#FF0000>※ gen_result does not support Coordconv </font>**

<h2 id = "Inference_one"> Inference and plot one image </h2>

Enter the command
```
./script/inference_one.sh
```
- In **inference_one.sh**
    - Select the checkpoint you want to generate solution 
    - Specify the image you want to inference 
    - Remember giving the file name for result<br> 
    ![image alt](./pic/3.png)
    
    **<font color=#FF0000>※ Inference_one does not support Coordconv and 25_shift iamges </font>**


<h2 id = "Visualization"> Visualization </h2>

To visualize your model of an image,
```
./script/gen_visualize.sh
```
Our work supports the visualization from [pytorch/captum](https://github.com/pytorch/captum).
There are five modes you can choose,  
In **gen_visualize.sh** 

1. Default (for testing if pytorch captum works)
2. Face Silhouette
3. Eyes
4. Nose
5. Mouth

**<font color=#FF0000>※ This operation requires a lot of RAM.</font> It is better to use CPU rather than GPU and make sure that `RAM > 50 GB` is available.**