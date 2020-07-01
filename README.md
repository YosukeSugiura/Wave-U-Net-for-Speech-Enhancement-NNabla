# Wave-U-Net for Speech-Enhancement (NNabla)

Implement of [Improved Speech Enhancement with the Wave-U-Net](https://arxiv.org/abs/1811.11307).

##  Requrement

### Python

  - Python 3.6
  - CUDA 10.1 & CuDNN 7.6
    + Please choose the appropriate CUDA and CuDNN version to match your [NNabla version](https://github.com/sony/nnabla/releases) 

### Packages

Please install the following packages with pip.
(If necessary, install latest pip first.)

  - nnabla  (over v1.1)
  - nnabla-ext-cuda  (over v1.1)
  - scipy 
  - numba  
  - joblib  
  - pyQT5  
  - pyqtgraph  (after installing pyQT5)
  - pypesq (see ["install with pip"](https://github.com/ludlows/python-pesq#install-with-pip) on offical site)  
    In the latest version, the package name is changed to `pesq`.
    If you install `pesq`, change 'pypesq' to `pesq` in 25th line in `wave-u-net.py` as
    ```@Python
    from pesq import pesq as pypesq
    ```
  
  ## Contents

  - **wave-u-net.py**  
      Main source code. Run this.
  
  - **data.py**  
      This is for creating Batch Data. Before runnning, please download wav dataset as seen below.
      
  - **settings.py**  
      This includes setting parameters.
      
  ## Download & Create Database
  
   1.   Download ```wave-u-net.py```, ```settings.py```, ```data.py``` and save them into the same directory.
   
   2.  In the directory, make three folders  ```data```, ```pkl```, ```params``` .
   
        - ```data```  folder :  save wav data.
        - ```pickle``` folder  :  save pickled database "~.pkl".
        - ```params``` folder  :  save parameters including network models.

   3.   Download  the following 4 dataset, and unzip them from  
        
        Noisy speech database for training speech enhancement algorithms and TTS models  
        https://datashare.is.ed.ac.uk/handle/10283/2791

        [ Direct Links ]
          - [clean_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_wav.zip)
          - [noisy_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_wav.zip)
          - [clean_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_testset_wav.zip)
          - [noisy_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_testset_wav.zip)

   4. Move those unzipped 4 folders into ```data```  folder.

   5.  Convert the sampling frequency of all the wav data to 16kHz.
         For example, [this site](https://online-audio-converter.com/) is useful.
         After converting, you can delete the original wav data. 
    
### Train & Predict

If train, in `wave-u-net.py`, 

```
 Train = 1
```

If predict, in `wave-u-net.py`, 

```
 Train = 0
```
