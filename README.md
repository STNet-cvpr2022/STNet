# STNet
pytorch code for submitted cvpr2022 papar.

The code is built on Siamfcpp (PyTorch) and tested on Ubuntu 18.04 environment with an NVIDIA RTX3090 GPU.

##  Test on FE240hz dataset
1. Download our preprocessing test dataset of FE240hz. (you can download the whole FE240hz dataset from [here](https://zhangjiqing.com/publication/iccv21_fe108_tracking/)).

2. Download the [pretrained model](https://drive.google.com/file/d/1xD-d24TRoMHRAQKIxE7CxMhI2UffSiUG/view?usp=sharing), and put it into ./snapshots/stnet.

3. Change your own data path in videoanalyst/engine/tester/tester_impl/eventdata.py.

4. run ``` python main/test.py --config experiments/test/fe240/fe240.yaml ``` the predicted bbox will be saved in logs/EVENT-Benchmark/. 

##  Test on VisEvent dataset
1. Download our preprocessing test dataset of VisEvent. (you can download the whole VisEvent dataset from [here](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)).

2. Download the [pretrained model](https://drive.google.com/file/d/1xD-d24TRoMHRAQKIxE7CxMhI2UffSiUG/view?usp=sharing), and put it into ./snapshots/stnet.

3. Change your own data path in videoanalyst/engine/tester/tester_impl/eventdata.py, change  _pretrain_model_path_ in experiments/test/fe240/fe240.yaml.

4. run ``` python main/test.py --config experiments/test/fe240/fe240.yaml ``` the predicted bbox will be saved in logs/EVENT-Benchmark/. 
