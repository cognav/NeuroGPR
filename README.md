# Brain-inspired multimodal hybrid neural network for robot place recognition

## Table of Contents
1. About
2. Structure
3. Usage
4. Datasets
5. Citation
6. License
7. Acknowledgments
8. Note

## About
This library implements a brain-inspired multimodal hybrid neural network (MHNN) for robot place recognition. The MHNN encodes and integrates multimodal cues from both conventional and neuromorphic sensors. Specifically, to encode different sensory cues, we build various neural networks of spatial view cells, place cells, head direction cells, and time cells. To integrate these cues, we design a multiscale liquid state machine that can process and fuse multimodal information effectively and asynchronously by using diverse neuronal dynamics and bio-inspired inhibitory circuits. 

## Structure
> * src
>> *	model: contains the MHNN model
>> *	tools: contains the utils 
>> *	config: contains the configure files 
>> *	main: contains the demo 

## Usage
* Step 1, setup the running environments
* Step 2, download the datasets 
* Step 3, download the code: <br>
git clone https://github.com/cognav/neurogpr.git
* Step 4, run the demo in the folder of ‘main’ :  <br> 
python main_mhnn.py --config_file ../config/corridor_setting.ini

## Requirements
The following libraries are needed for running the demo. 
* Python 3.7.4
* Torch 1.11.0
* Torchvision 0.12.0
* Numpy 1.21.5
* Scipy 1.7.1

## Datasets
The datasets include four groups of data collected in different environments. 
* The Room, Corridor, and THU-Forest datasets are available on Zenodo (https://zenodo.org/record/7827108#.ZD_ke3bP0ds). 
* The public Brisbane-Event-VPR dataset is available on Zenodo (https://zenodo.org/record/4302805#.ZD8puXbP0ds). 

## Citation
Fangwen Yu, Yujie Wu, Songchen Ma, Mingkun Xu, Hongyi Li, Huanyu Qu, Chenhang Song, Taoyi Wang, Rong Zhao and Luping Shi. Brain-inspired multimodal hybrid neural network for robot place recognition. Science Robotics (accepted). 

## License
MIT License

## Acknowledgments 
If you have any questions, please contact us. 

## Note
If the parameter settings are different, the results may be not consistent. You may need to modify the settings or adjust the code properly.

