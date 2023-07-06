# Linking data separation, visual separation, and classifier performance

A Python implementation of the proposed paper [1] and its further contribution [2]. In [1], we showed the correlation of the concepts of data separation (DS), visual separation (VS), and classifier performance (CP) through pseudolabeling and constrastive learning -- that is, high DSleads to high VS and next high CP. For this, we explored a  embedded pseudo-labeling approach (DeepFA) by using a non-linear projection (t-SNE) from a feature space of the DNN to a 2D space, followed by semi-supervised label propagation using a connectivity-based method (OPFSemi). We used contrastive learning to produce the latent space for DeepFA by three methods (SimCLR, SupCon, and their combination).

In [2], we proposed to evaluate ten projection techniques for the same pipeline, and we identified two classes of projection techniques – one leading to poor VS and next poor CS regardless of the available DS, and the other showing a good DS-VS-CP correlation. We argue that this last group of projections is a useful instrument in classifier engineering tasks.

Example of our proposed approach on a small subset of MNIST dataset is provided in this code. The percentage of the initial supervised samples is randomly chosen.

[1] Benato, B. C., Falcão, A. X., Telea, A. C. "Linking Data Separation, Visual Separation, and Classifier Performance Using Pseudo-labeling by Contrastive Learning". VISIGRAPP 2023. ([pdf](https://webspace.science.uu.nl/~telea001/uploads/PAPERS/VISAPP23/paper.pdf))

## Installing

DeepFA is based on OPFSemi, and its running depends on OPFSemi implementation. A precompiled file of PyIFT (see details [here](https://github.com/JoOkuma/PyIFT). A binary file is given in DeepFA/dist directory. We tested on Ubuntu 18.04 and Python 3.8. For installation purposes, a Dockerfile is provided. You can follow below DeepFA installation via Dockerfile.

### Dockerfile 

First, we need to download the Dockerfile provided in this project.
```
curl -H 'Authorization: token ACCESS_TOKEN ' -H 'Accept: application/vnd.github.v3.raw' -O -L https://raw.github.com/barbarabenato/linking_ds_vs_cp_proj/main/Dockerfile
```

Then, we build the image from the Dockerfile in the same folder we downloaded the Dockerfile and run the container. 

```
docker build -t linking_img . && docker run -it linking_img 
```

You can check the PyIFT installation by:
```
python
>>> import pyift.pyift as ift
>>>
```


### Linking DS, VS, and CP
After running a container with PyIFT installed and linking_ds_vs_cp_proj downloaded, we install our package by changing to the main directory and installing its dependencies:
```
cd linking_ds_vs_cp/ && python -m pip install .
```

You can check the installation by:
```
python
>>> import linking_ds_vs_cp
>>>
```

## Running
A simple example of DeepFA is provided on a small subset of MNIST. The usage of "run_example.py" and its parameters are provided below.
```
usage run_example.py <model name [simclr, supcon, both]> <perc of sup samples> <dimensionality reduction> <iterations for contrst learn>
```

You can run it by changing to deepfa directory and executing, for example:
```
python run_example.py simclr 0.5 umap 50 
```

After its running, you should be able to see the generated feature learning curves (with training and validation losses/accuracies), 2D projection, propagation accuracy/kappa, and classification accuracy/kappa. Colored points in the 2D projection represent different classes, circled red points are the supervised points. A list of possible dimensionality reduction algorithms can be accessed in [2] and linking_ds_vs_cp/dimensionality_reduction.py

## Citation
If you use the provided code, please cite the following article:
```
@conference{Benato:visapp23,
    author={Bárbara Benato. and Alexandre Falcão. and Alexandru{-}Cristian Telea.},
    title={Linking Data Separation, Visual Separation, and Classifier Performance Using Pseudo-labeling by Contrastive Learning},
    booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2023) - Volume 5: VISAPP},
    year={2023},
    pages={315-324},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0011856300003417},
    isbn={978-989-758-634-7},
    issn={2184-4321},
}
```

