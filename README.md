# rtmdet-tiny-plantdoc
model:rtmdet-tiny, mmdetection based, plantdox datasets.

All experiments of this model, including comparative experiments and ablation experiments, were conducted based on the mmdetection 3.3.0 framework. 。

 This document provides the running files and results of all models. 

 The model running files are stored in the folder `rtmdet-plantdoc-mmdetection`, and files with the suffix `.ipynb` are the model running files. The correspondence between file names and models is as follows: 

| File name                             | model        | epoch | Additional Notes                                           |
| ------------------------------------- | ------------ | ----- | ---------------------------------------------------------- |
| run-atss.ipynb                        | atss         | 12    |                                                            |
| run-atss-2x.ipynb                     | atss         | 24    |                                                            |
| run-atss-3x.ipynb                     | atss         | 36    |                                                            |
| run-atss-4x.ipynb                     | atss         | 48    |                                                            |
| run-atss-5x.ipynb                     | atss         | 60    |                                                            |
| run-autoassign-4x.ipynb               | autoassign   | 48    |                                                            |
| run-centernet-4x.ipynb                | centernet    | 48    |                                                            |
| run-diffusiondet-4x.ipynb             | diffusiondet | 48    |                                                            |
| run-faster-rcnn-4x.ipynb              | faster-rcnn  | 48    |                                                            |
| run-fcos-4x.ipynb                     | fcos         | 48    |                                                            |
| run-vfnet-4x.ipynb                    | VarifocalNET | 48    |                                                            |
| run-rtmdet-tiny.ipynb                 | rtmdet-tiny  | 300   |                                                            |
| run-rtmdet-tiny-assem.ipynb           | rtmdet-tiny  | 300   | Adopts the AssemFormer mechanism                           |
| run-rtmdet-tiny-axialBlock.ipynb      | rtmdet-tiny  | 300   | Adopts the Axial mechanism                                 |
| run-rtmdet-tiny-bra.ipynb             | rtmdet-tiny  | 300   | Adopts the Bi-Level Routing Attention mechanism            |
| run-rtmdet-tiny-cpca.ipynb            | rtmdet-tiny  | 300   | Adopts the Channel Prior Convolutional Attention mechanism |
| run-rtmdet-tiny-ppa.ipynb             | rtmdet-tiny  | 300   | Adopts the parallelized patch-aware attention mechanism    |
| run-rtmdet-tiny-simam.ipynb           | rtmdet-tiny  | 300   | Adopts the SimAM mechanism                                 |
| run-rtmdet-tiny-eucb.ipynb            | rtmdet-tiny  | 300   | Adopts the EUCB upsampling module                          |
| run-rtmdet-tiny-axialBlock-eucb.ipynb | rtmdet-tiny  | 300   | Adopts both the Axial mechanism and EUCB upsampling module |

2.The running results of all models are located in the `rtmdet-plantdoc-mmdetection/workdirs` folder. Each model's results are stored in a separate folder, with a naming convention similar to that in the table above, which will not be repeated here. Each model provides a model configuration file (a `.py` file) and a log file output during testing. For local runs, the folder name is `work_dirs`, which has been renamed here for uploading to GitHub. If you wish to check the `.pth` files obtained from model training, please enter the following Aliyun Disk link (https://www.alipan.com/s/VfV776E13bJ) to download them on your own. After downloading the `.pth` files, place them into the corresponding folders. 

3.The test set images are saved in the folder `rtmdet-plantdoc-mmdetection/images`. This model was run on a cloud platform, which uses unique paths for file reading. Therefore, the model does not read images from this folder, and the folder path specified in the configuration files (files with the suffix `plantdoc.py`) is not the actual path used. If you wish to run the model locally, please manually modify the folder path in the corresponding `.py` files.