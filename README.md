The project consists of 3 main parts:

1) Segmentation task (Unet)
2) Nodule/non-nodule classification model (our own CNN LunaModel class)
3) Benign/malignant nodule classification (fine-tuning the last convolutional and FF layers LunaModel)

1. Segmentation: this part is realised in datasets_segment and model_segment files (+ unet.py in utils directory). We used the Unet realization from https://github.com/jvanvugt/pytorch-unet and adapted it to our needs. It was exploited in the UnetWrapper class where we add BatchNorm in the beginning and Softmax in the end + Xavier weights initialization. The UnetWrapper class is used in our SegmentationTrainingApp class which helps us to find the nodules through slices of 3D CT scan. It makes our classification task easier in the future.

2. Nodule/non-nodule classification: trained on the segmented crops containing dense lumps of tissue (candidates), the model learns to distinguish between nodules (which can be benign or malignant (tumors)) and non-nodules.
