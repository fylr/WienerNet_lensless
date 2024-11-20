# Enhancing object recognition for lensless cameras through PSF correction and feature loss

## Authors
Kaiyu Chen, Ying Li, Zhengdai Li, Jiangtao Hu, Youming Guo

## Highlights
- A lightweight and interpretable PSF correction network to address model mismatch issues.
- A feature loss function to emphasize the reconstruction of local regions that are critical for recognition.
- A series of incremental network models that integrate physical models and deep learning, achieving high recognition accuracy and good generalization on complex datasets.

## Abstract
Lensless cameras have significant application prospects in multiple specialized fields due to their unique advantages of small size, low cost, and a naturally optical encryption effect. However, achieving high-accuracy object recognition in complex scenarios remains a challenging task. Firstly, many factors in the actual imaging process can induce the point spread function (PSF) distortion, resulting in a mismatch between the ideal model and the experiment. In this study, these factors are equivalently regarded as a blur kernel. Based on the singular value decomposition (SVD) results of the blur kernel, a lightweight and interpretable PSF correction network is proposed to counteract this blur kernel. Then, inspired by the class activation mapping (CAM) of classification networks, a feature loss function is proposed to indirectly achieve the reconstruction of local regions that play a crucial role in recognition by seeking feature layers alignment. Based on these, a series of incremental network models that integrate physical models and deep learning are designed, which achieve significantly better performance than other advanced methods on the cats\_vs\_dogs and ImageNet\_10 datasets, and also show good generalization in the recognition test of real-world scenes. The proposed method significantly improves the object recognition performance of lensless cameras and has the potential for application in complex scenarios. The codes will be available at [https://github.com/fylr/WienerNet_lensless](https://github.com/fylr/WienerNet_lensless).

## News
wait!!!
