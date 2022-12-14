# Stereo video datasets
+ [Stereo videos scraped from Youtube](https://sites.google.com/view/wsvd/home)
+ [Driving stereo sequences like Kitti](http://www.cvlibs.net/datasets/karlsruhe_sequences/)
+ [Another driving sequence](https://drivingstereo-dataset.github.io/)
+ [Synthetic stereo videos with ground-truth disparity maps](https://richardt.name/publications/dcbgrid/datasets/)
+ [Large scale high-res stereo data from zed camera](www.rovit.ua.es/dataset/uasol/)
+ [Urban stereo dataset](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-02-pedestrian-dataset/)
+ [Visual inertial stereo dataset with fish-eye lens](https://vision.in.tum.de/data/datasets/visual-inertial-dataset)
+ [Synthetic Indoor stereo video dataset](https://github.com/HKBU-HPML/IRS)
+ [Plenoptic and stereo camera video dataset for odometry evaluation](https://www.hs-karlsruhe.de/odometry-data/)
+ [Cityscapes stereo data](https://www.cityscapes-dataset.com/)
+ [Kitti stereo data](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
+ [Underwater stereo images](http://rimlab.ce.unipr.it/Maris.html)
+ [Sintel dataset](http://sintel.is.tue.mpg.de/)
+ [Middlebury stereo data](https://vision.middlebury.edu/stereo/data/)
+ [Underwater stereo images](http://csms.haifa.ac.il/profiles/tTreibitz/datasets/ambient_forwardlooking/index.html)
+ [High speed stereo video dataset (480 fps)](https://stereoblur.shangchenzhou.com)
+ https://arpg.github.io/oivio//
+ [stereo images data](https://dimlrgbd.github.io/#)
+ [Open loris scene dataset](https://shimo.im/docs/HhJj6XHYhdRQ6jjk/read)
+ [3D movie dataset from INRIA](https://www.di.ens.fr/willow/research/stereoseg/)
+ [ETH3D dataset](https://www.eth3d.net/datasets#high-res-multi-view)
+ [Shallow and deep depth of field image pairs]: https://ceciliavision.github.io/vid-auto-focus/
+ [IIITH Stereoscopic 3D data](https://www.iith.ac.in/~lfovia/downloads.html): Stereo data collected for stereo video quality assessment
+ [Another stereoscopic 3D data](http://ivc.univ-nantes.fr/en/databases/NAMA3DS1_COSPAD1/): Stereo data collected for stereo video quality assessment


# Light field video datasets:
There are mainly 4 datasets:
+ [Raytrix 5x5 data](http://clim.inria.fr/Datasets/RaytrixR8Dataset-5x5/index.html): This one contains 5x5 3 video sequences with 5x5 angular resolution data. Has small baseline. But the spatial resolution is quite high
+ [Hybrid LF video dataset](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIG17/lfv/): A video data captured using Lytro. Small spatial resolution. angular resolution is 8x8
+ [X-fields dataset](https://xfields.mpi-inf.mpg.de/): Has about 8 videos. 3 videos have 3x3x3 views, i.e. 3 frames with each 3x3 angualr views. Another set of videos has 5x5x5 views, i.e. 5 frames each with 5x5 angular views.
+ [Camera grid dataset](https://www.interdigital.com/data_sets/light-field-dataset)
+ [Mobile phone stereo examples from DU2Net](https://github.com/augmentedperception/du2net/tree/master/readme_files/right_camera)

# Temporally Consistent Video to video papers

+ ICCV 2017 : [Coherent Online Video Style Transfer](https://arxiv.org/abs/1703.09211)
+ Siggraph Asia 2016 : [Temporally Coherent Completion of Dynamic Video](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/SigAsia_2016_VideoCompletion.pdf)
+ NeurIPS 2018 : [Video-to-Video Synthesis](https://arxiv.org/abs/1808.06601); [Webpage](https://tcwang0509.github.io/vid2vid/)
+ IEEE RAL 2020: [Don't Forget The Past: Recurrent Depth Estimation from Monocular Video](https://arxiv.org/abs/2001.02613)
+ Unpublished : [Robust Consistent Video Depth Estimation](https://arxiv.org/pdf/2012.05901.pdf)
+ CVPR 2019 - [Deep Video Inpainting](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Video_Inpainting_CVPR_2019_paper.pdf): Interesting paper
+ Unpublished - [World-Consistent Video-to-Video Synthesis](https://arxiv.org/pdf/2007.08509.pdf); [Webpage](https://nvlabs.github.io/wc-vid2vid/)
+ ICCV 2019 - [Onion-Peel Networks for Deep Video Completion](https://openaccess.thecvf.com/content_ICCV_2019/papers/Oh_Onion-Peel_Networks_for_Deep_Video_Completion_ICCV_2019_paper.pdf): uses an attention mechanism similar to those in NLP like BERT etc. There is not really anything about recurrence. This is more like Video denoising or Video SR papers.
+ ICCV 2019 - [Exploiting temporal consistency for real-time video depth estimation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Exploiting_Temporal_Consistency_for_Real-Time_Video_Depth_Estimation_ICCV_2019_paper.pdf)
+ CVPR 2019 - [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/pdf/1903.07291.pdf): Not a video paper, but very interesting paper. This generates an image given a semantic map. It has a generator which takes input a noise vector from Gaussian distribution. Then it modulates the features map like batch-norm but the mean and variance vectors are derived from the semantic map.
+ CVPR 2019 - [Single-frame Regularization for Temporally Stable CNNs](https://openaccess.thecvf.com/content_CVPR_2019/papers/Eilertsen_Single-Frame_Regularization_for_Temporally_Stable_CNNs_CVPR_2019_paper.pdf): Somewhat an interesting paper which tries to ensure local temporal consistency for networks trained on single images.


# Shallow depth of field from mobile phones and others
+ [Learning Single Camera Depth Estimation using Dual-Pixels](https://github.com/google-research/google-research/dual_pixels)
+ [DU2-net: Learning Depth Estimation from Dual-Cameras and Dual-Pixels](https://augmentedperception.github.io/du2net/)
+ [DeepLens: Shallow Depth of Field from a Single Image](https://deeplensprj.github.io/deeplens/DeepLens.html) has [Code](https://github.com/scott89/deeplens_eval) and dataset is proposed but not released
+ [Deep Sparse Light Field Refocusing](https://arxiv.org/pdf/2009.02582.pdf): Pretty interesting paper as in it does direct refocusing from sparse light fields


# Light field estimation from sparse measurements
## Angular super-resolution
+ Siggraph Asia 2016 - [Learning-Based View Synthesis for Light Field Cameras](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/PaperData/SIGGRAPHAsia16_ViewSynthesis_LoRes.pdf); [Webpage](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/); [Dataset](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/PaperData/SIGGRAPHAsia16_ViewSynthesis_Trainingset.zip); [Code](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/PaperData/SIGGRAPHAsia16_ViewSynthesis_Code_v2.0.zip)
+ CVPR 2017 - [Light Field Reconstruction Using Deep Convolutional Network on EPI](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wu_Light_Field_Reconstruction_CVPR_2017_paper.pdf): Input 3x3 sparse views. The network takes the EPI images as input and does the upsampling. A supervised framework.
+ ICCV 2017 - [Learning to Synthesize a 4D RGBD Light Field from a Single Image](https://openaccess.thecvf.com/content_ICCV_2017/papers/Srinivasan_Learning_to_Synthesize_ICCV_2017_paper.pdf): Train the network on lots of flower images. Then reconstruct light field from a single image. Disparity based rendering is used. Non-lambertian effects are synthesized using a residual block using the supervision.
+ Siggraph 2018 - [Stereo Magnification: Learning view synthesis using multiplane images](https://arxiv.org/pdf/1805.09817.pdf)
+ CVPRW 2017 - [Compressive Light Field Reconstructions using Deep Learning](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w15/papers/Gupta_Compressive_Light_Field_CVPR_2017_paper.pdf)
+ ECCV 2018 - [Learning to capture light fields through a coded aperture camera](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yasutaka_Inagaki_Learning_to_Capture_ECCV_2018_paper.pdf)
+ ICCV 2017 - [Neural EPI-volume networks for shape from light field](http://openaccess.thecvf.com/content_ICCV_2017/papers/Heber_Neural_EPI-Volume_Networks_ICCV_2017_paper.pdf)
+ ECCV 2018 - [End-to-end view synthesis for light field imaging with pseudo 4DCNN](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yunlong_Wang_End-to-end_View_Synthesis_ECCV_2018_paper.pdf)
+ CVPR 2018 - [Enhancing the spatial resolution of stereo images using a parallax prior](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jeon_Enhancing_the_Spatial_CVPR_2018_paper.pdf)
+ ICCV 2019 - [Extreme View Synthesis](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Extreme_View_Synthesis_ICCV_2019_paper.pdf): It's like the stereo magnification paper
+ Selected Topics in Circuits and Systems - [Light Field Image Compression Using Generative Adversarial Network-Based View Synthesis](https://hpc.pku.edu.cn/docs/pdf/a20191230083.pdf)
+ TIP - [Light Field Super-Resolution using a Low-Rank Prior and Deep Convolutional Neural Networks](https://hal.archives-ouvertes.fr/hal-01984843/document)
+ ACCV 2018 - [Dense light field reconstruction from sparse sampling using residual network](https://arxiv.org/pdf/1806.05506.pdf)
+ Siggraph Asia 20 - [Synthesizing light field from a single image with variable MPI and two network fusion](https://people.engr.tamu.edu/nimak/Data/SIGASIA20_LF_LoRes.pdf)


## Light field from stereo pair
+ CVPR 2015 - [Light Field from Micro-baseline Image Pair](http://liuyebin.com/binolf/binolf.pdf): A traditional, non-learning based method which first predicts the stereo disparity map and then renders the light field
+ Siggraph 2013 - [Joint View Expansion and Filtering for Automultiscopic 3D Displays](http://people.csail.mit.edu/pdidyk/projects/MultiviewConversion/MultiviewConversion.pdf): No code is available; however test images are available [Webpage](http://people.csail.mit.edu/pdidyk/projects/MultiviewConversion/)



## Spatio-angular super-resolution
+ TIP 2018 [Spatial and angular resolution enhancement of light fields using convolutional neural networks](https://arxiv.org/pdf/1707.00815)
+ [Light Field Super-resolution via Attention-Guided Fusion of Hybrid Lenses]

## Temporal super-resolution
+ Siggraph 2017 - [Light field video capture using a learning-based hybrid imaging system](https://dl.acm.org/doi/pdf/10.1145/3072959.3073614)
+ [3DTV at home: eulerian-lagrangian stereo-to-multiview conversion](https://cdfg.mit.edu/assets/files/home3d.pdf)
+ CGF journal 2020 - [Single Sensor Compressive Light Field Video Camera](https://hal.archives-ouvertes.fr/hal-02498719/file/Single_Sensor_Compressive_Light_Field_Video_Camera.pdf)
+ unpublished - [5D Light Field Synthesis from a Monocular Video](https://arxiv.org/pdf/1912.10687)
+ Sig Asia 2020 - [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://xfields.mpi-inf.mpg.de/) : Contains LF video dataset also has [Code](https://github.com/m-bemana/xfields) and [Dataset](https://xfields.mpi-inf.mpg.de/dataset/view_light_time.zip)
+ ICCV 2021 - [SeLFVi: Self-supervised Light Field Video Reconstruction from Stereo Video](https://openaccess.thecvf.com/content/ICCV2021/papers/Shedligeri_SeLFVi_Self-Supervised_Light-Field_Video_Reconstruction_From_Stereo_Video_ICCV_2021_paper.pdf): [Code](https://github.com/asprasan/selfvi), [Webpage](https://asprasan.github.io/pages/webpage-ICCV/index.html) and [Supplementary Material](https://openaccess.thecvf.com/content/ICCV2021/papers/Shedligeri_SeLFVi_Self-Supervised_Light-Field_Video_Reconstruction_From_Stereo_Video_ICCV_2021_paper.pdf)

# Perceptual or other metrics for stereo and light field
+ [Perceptual evaluation of light field image](http://home.ustc.edu.cn/~weichou/papers/18_ICIP_LF.pdf)
+ [Light Field Image Quality Assessment: An Overview](https://ieeexplore.ieee.org/abstract/document/9175517)
+ [Defocus evaluation](https://corp.dxomark.com/wp-content/uploads/2018/02/2018_EI_Image-quality-benchmark-of-computational-bokeh_small.pdf)

# Video defocus
+ Siggraph - [Synthetic defocus and look-ahead autofocus for casual videography](https://arxiv.org/pdf/1905.06326.pdf)
+ Siggraph [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://dl.acm.org/doi/pdf/10.1145/3414685.3417827)

# Some companies working on Light field imaging and displays
+ [FacetVision, Germany](https://www.facetvision.de/)
+ [K|Lens, Germany](https://www.k-lens.de/)
+ [Leia Lumepad, USA](https://www.leiainc.com/)
+ [Dimenco, Netherlands](https://www.dimenco.eu/)
+ [SeeFront, Germany](https://www.seefront.com/)
