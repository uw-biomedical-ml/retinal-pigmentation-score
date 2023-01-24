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


# Execution instructions

`docker pull arajesh/rps`

`docker exec -it -v <mount file system> --gpus all arajesh/rps /bin/bash`

*within the docker container*
  
`conda activate automorph`
  
*change the config.py to have variables relevant to you*

`python main.py`


### To-Do
- [X] Modularize Automorph
- [ ] Figure out minimum docker requirements for automorph
- [ ] Port the retinal pigmentation extraction code
- [ ] Reduce the file size specifically removing the extra model state dicts that are stored for the models in automorph

