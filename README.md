DATA: data can be downloaded from this resource ...

The project consists of 3 main parts:

1) Segmentation task (Unet)
2) Nodule/non-nodule classification model (our own CNN LunaModel class)
3) Benign/malignant nodule classification (fine-tuning the last convolutional and FF layers LunaModel)

1. Segmentation: this part is realised in datasets_segment and model_segment files (+ unet.py in utils directory). We used the Unet realization from https://github.com/jvanvugt/pytorch-unet and adapted it to our needs. It was exploited in the UnetWrapper class where we add BatchNorm in the beginning and Softmax in the end + Xavier weights initialization. The UnetWrapper class is used in our SegmentationTrainingApp class which helps us to find the nodules through slices of 3D CT scan. It makes our classification task easier in the future. It also has augmentation model. That`s helps us combat overfitting by creating rotated/flipped/noised slices of scans. For segmentation task we use per-pixel CrossEntropy analogue - DiceLoss. Its formula resembles F1-score for usual classification tasks. 

2. Nodule/non-nodule classification: trained on the segmented crops containing dense lumps of tissue (candidates), the model learns to distinguish between nodules (which can be benign or malignant (tumors)) and non-nodules. The LunaModel class consists of tail (BatchNorm), 4 convolutional blocks and head (Softmax) + CrossEntropy as loss function.

3. Bening/malignant nodule classification: the model remains the same as for the second part, but there we used fine-tuning techinques. All weights remain the same, except for the last convolutional and Feed-Forward layers in the end, because our network must find other features from the given images in order to do this task well. Also we need to adjust the weights of the fully-connected layers for proper classification.
