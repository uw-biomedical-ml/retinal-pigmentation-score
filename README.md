# retinal-pigmentation-score
Calculate the retinal pigmentation from a color fundus image

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

# Dataset Format

**Input**
  
  All color fundus photos need to be in a single directory and in png format.

**Output**

  All results will be dumped into a folder that you specify

The output will have multiple directories:

  1. M0: preprocessing outputs
  
  2. M1: image quality steps - results will be stored in M1/Good_quality/image_list.csv
  
  3a. M2 disc_cup: optic disc and cup segmentation
  
  3b. M2 binary_vessel: binary vessel segmentation
  
  3c. M2 artery_vein: artery and vein multiclass segmentation
  
  *retinal_background_lab_values.csv* : csv file with the median retinal background for each image extracted with retinal pigmentation extraction script
  

# Execution instructions

1. Pull the docker container and start it.

    The docker container is stored here: https://hub.docker.com/r/arajesh17/rps

    Please type the two commands:
    
      `docker pull arajesh17/rps`
    
    Now in your terminal after you finished the pull, you can start the container with an interactive shell
    
      `docker run -it -v <your images path:/home/images/> -v <your results path>:/home/results/ -v <your rps repo path>:/home/retinal-pigmentation-score/ --gpus all arajesh17/rps /bin/bash`

2. Activate the python environment in the docker container

    `conda activate automorph`

3. Run the main.py

    `python /home/retinal-pigementation-score/src/main.py`


### To-Do
- [X] Modularize Automorph
- [ ] Figure out minimum docker requirements for automorph
- [ ] Port the retinal pigmentation extraction code
- [ ] Reduce the file size specifically removing the extra model state dicts that are stored for the models in automorph

