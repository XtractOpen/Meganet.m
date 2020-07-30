## GNvpro
A Matlab implementation of a trust region Newton-Krylov variable projection scheme for efficient training of deep neural networks.

### Associated Publication
***Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection***

Please cite as:


```latex
@article{newman2020train,
	title={Train Like a (Var)Pro: Efficient Training of Neural Networks with Variable Projection},
	author={Elizabeth Newman and Lars Ruthotto and Joseph Hart and Bart van Bloemen Waanders},
	year={2020},
	journal={arXiv preprint arxiv.org/abs/2007.13171},
}
```

### Setup
GNvpro is a subdirectory of Meganet.m repository. Download or clone from below:

[https://github.com/XtractOpen/Meganet.m](https://github.com/XtractOpen/Meganet.m)

Before running the scripts, set the Matlab path to include all of the Meganet.m repository.  Change the working directory to Meganet.m and run the following command:

```matlab
addpath(genpath('.'))
```

The scripts used in the paper are located in the subdirectory **examples/GNvpro** and are listed below.  The scripts include the hyperparameters used in the paper as well as code to visualize the results. 

###Toy Example

Train a DNN on a binary classification problem determining if points are contained inside an ellipse or not.

```matlab
EVP_Circle_GNvpro.m
EVP_Circle_GN.m
```

### PDE Surrogate Modeling

Train a DNN as a surrogate model of a PDE.  The data is available for download [here](url).

#### Convection Diffusion Reaction (CDR)


```matlab
EVP_CDR_GNvpro.m
EVP_CDR_GN.m
EVP_CDR_SGD.m
EVP_CDR_LBFGSvpro.m
```

#### Direct Current Resistivity (DCR)

```matlab
EVP_DCR_GNvpro.m
EVP_DCR_GN.m
EVP_DCR_SGD.m
EVP_DCR_LBFGSvpro.m
```

We generated the data in Matlab using a Finite Volume approach.  The code used to generate the data is available in the subdirectory **data/DCR**.

```matlab
driverGenerateDCRData.m % script to generate DCR data
setupDCROperators.m 		% function to setup finite volume operators
solveDCR.m 							% function to solve the PDE
```

### Classification and Image Segmentation

#### Indian Pines
Train a DNN to segment the [Indian Pines dataset](http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes).  The data consists of 145 x 145 images corresponding to 220 spectral wavelengths.  The goal is to classify the material or crop located at each pixel.

```
EVP_IndianPines_GNvpro.m
EVP_IndianPines_GN.m
EVP_IndianPines_SGD.m
EVP_IndianPines_LBFGSvpro.m
```

### Ackowledgments

Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government. Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE- NA-0003525. SAND2020-7339 O.