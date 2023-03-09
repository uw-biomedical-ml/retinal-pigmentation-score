# retinal-pigmentation-score
Calculate the retinal pigmentation from a color fundus image using deep learning to segment the vasculature and nerve, then find the median pixel value of retinal background in the Lab colorspace.

## Credit
A significant portion of this pipeline was built from code from Automorph. Thank you!

https://github.com/rmaphoh/AutoMorph/blob/main/README.md

```
@article{zhou2022automorph,
  title={AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline},
  author={Zhou, Yukun and Wagner, Siegfried K and Chia, Mark A and Zhao, An and Xu, Moucheng and Struyven, Robbert and Alexander, Daniel C and Keane, Pearse A and others},
  journal={Translational vision science \& technology},
  volume={11},
  number={7},
  pages={12--12},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```

### Dataset Format

### Input
  
  All color fundus photos need to be in a single directory and in png format.

### Output
    
  All results will be dumped into a folder that you specify: *ie* `/data/arajesh/rps/results`
  
  The results directory will have the subsequent structure

    .
    |-- M0/ Pre-processing outputs
    |-- M1/ Image Quality Assesment
    |   |--Good_quality/image_list.csv **(list of good quality images)**
    |-- M2/
    |   |--binary_vessel/raw_binary/ **(binary vessel segmentation masks)**
    |       |--vessel_seg1.png
    |       |--vessel_seg2.png
    |   |--disc_cup/optic_disc_cup/ **(disc segmentation masks)**
    |       |--disc_seg1.png
    |       |--disc_seg2.png
    |-- retinal_pigmentation_score.csv **(csv with median a,b values and 'pigmentation' a.k.a RPS score)**
    |-- RPS_representative_images.png **(representative figure giving context for what RPS scores are with regards to RGB colors)**


# Execution instructions

1. Clone this code repository:

  `git clone https://github.com/arajesh17/retinal-pigmentation-score.git`

2. Pull the docker container and start it.

    The docker container is stored here: https://hub.docker.com/r/arajesh17/rps

    Please pull the repo 
    
      `docker pull arajesh17/rps`
    
    Now in your terminal after you finished the pull, you can start the container with an interactive shell
    
      `docker run -it -v <your images path:/home/images/> -v <your results path>:/home/results/ -v <your rps repo path>:/home/retinal-pigmentation-score/ --gpus all arajesh17/rps /bin/bash`

3. Activate the python environment in the docker container

    `conda activate automorph`

4. Run the main.py

    `python /home/retinal-pigementation-score/src/main.py`
    
5. Figure out optimal worker number and batch_size

  Run main.py() and also `nvidia-smi` in another window on the same machine to look at GPU memory usage while executing the code. If you are not using all of your card's memory, increase the batch size until you are using nearly all of this. Batch size can be modified in the `src/config.py` file of this repo.


### To-Do

