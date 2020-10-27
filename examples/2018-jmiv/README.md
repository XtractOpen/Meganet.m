# Examples from Deep Neural Networks Motivated by PDEs

These drivers solve image classification problem with the STL-10, CIFAR-10, 
and CIFAR-100 datesets, as described in 

'''
@article{Ruthotto2018DeepNN,
  title={Deep Neural Networks Motivated by Partial Differential Equations},
  author={Lars Ruthotto and Eldad Haber},
  journal={Journal of Mathematical Imaging and Vision},
  doi={10.1007/s10851-019-00903-1},
  year={2018},
  pages={1 - 13}
}
'''

The drivers are periodically maintained to keep up with development of the 
Meganet package. 

## Overview

There is one driver for each dataset, i.e., runCIFAR10.m, runCIFAR100.m, 
and runSTL10.m. The network architecture and training for all cases is set 
up in cnnDriver.m. After the training, the test accuracy can be computed with
cnnResults.m. We also include testCNNDriver.m, which tests the training for 
small-scale instances.

## Note Regarding Efficiency

This code (as its containing Meganet.m package) is not geared to efficiency. 
Even with a good GPU, running all the experiments will take a few hours. Our 
networks use standard functions available in all common machine learning packages
and faster implementations are possible.

## Acknowledgements

This material is in part based upon work supported by the National Science Foundation 
under Grant Numbers 1522599 and 1751636. Any opinions, findings, and conclusions or recommendations 
expressed in this material are those of the author(s) and do not necessarily reflect 
the views of the National Science Foundation. We also thank NVIDIA for the donation
of a GPU that was used to develop and test the codes and all contributors 
to the Meganet package, particularly Eran Treister.




