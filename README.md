<h2>Tensorflow-Image-Segmentation-FIVES-Retinal-Vessel (2025/02/19)</h2>
Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>FIVES</b> Retinal Vessel
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://figshare.com/ndownloader/files/34969398"><b>FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation</b></a>
<br>
<br>
Please see also our experiments:<br>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-HRF-Retinal-Vessel</a> based on
<a href="https://www5.cs.fau.de/research/data/fundus-images/">High-Resolution Fundus (HRF) Image Database</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorlfow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-DRIVE-Retinal-Vessel</a> based on 
<a href="https://drive.grand-challenge.org/">DRIVE: Digital Retinal Images for Vessel Extraction</a>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</a> baased on 
<a href="https://cecas.clemson.edu/~ahoover/stare/">STructured Analysis of the Retina</a>.
<br>
</li>
<li>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
Tensorflow-Image-Segmentation-Retinal-Vessel</a> based on <a href="https://researchdata.kingston.ac.uk/96/">CHASE_DB1 dataset</a>.
</li>
<br>

<br>
<hr>
<b>Actual Image Segmentation for Images of 2048x2048 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/2_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/2_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/2_A.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/3_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/3_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/3_A.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/5_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/5_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/5_A.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this FIVES Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the following figshare web-site:<br><br>
<a href="https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1?file=34969398">
<b>
FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation
</b>
</a>
<br>
Kai Jin, Xingru Huang, Jingxin Zhou, Yunxiang Li, Yan Yan, Yibao Sun, Qianni Zhang, Yaqi Wang, Juan Ye<br>
<br>

<b>FIVES</b> dataset consists of 800 high-resolution multi-disease color fundus photographs with pixel-wise manual annotation. 
The annotation process was standardized through crowdsourcing of a group of medical experts. 
The quality of each image was evaluated, including illumination and color distortion, blur, and low contrast 
distortion, based on which the data splitting was conducted to make sure the balanced distribution of image features.
<br>
<br>
Detailed descriptions can be found in the original paper, and please cite it if utilizing any part of the dataset: <br>

Jin, K., Huang, X., Zhou, J. et al. FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation. <br>
Sci Data 9, 475 (2022). https://doi.org/10.1038/s41597-022-01564-3 <br>
<br>
<h3>
<a id="2">
2 FIVES ImageMask Dataset
</a>
</h3>
 If you would like to train this FIVESSegmentation model by yourself,
 please download the dataset from <a href="https://figshare.com/ndownloader/files/34969398">
 <b>FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation</b></a>
<br>
The folder structure of the dataset is the following,<br>
<pre>
./FIVES A Fundus Image Dataset for AI-based Vessel Segmentation
├─test
│  ├─Ground truth
│  └─Original
└─train
    ├─Ground truth
    └─Original
</pre>
<br>
As shown below, we splitted the original dataset into <b>test</b>, <b>train</b> and <b>valid</b> subsets.
<pre>
./dataset
└─FIVES
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>FIVES Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/FIVES/FIVES_Statistics.png" width="512" height="auto"><br>
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained FIVESTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/FIVES/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/FIVES and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>Color space conversion</b><br>
Used COLOR_BGR2Luv color space converter..
<pre>
[image]
color_converter = "cv2.COLOR_BGR2Luv"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for an image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<br>

<b>Epoch_change_inference output at ending (62,63,64)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>
<br>
In this experiment, the training process was stopped at epoch 64 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/train_console_output_at_epoch_64.png" width="720" height="auto"><br>
<br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/FIVES/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/FIVES/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/FIVES</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for FIVES.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/evaluate_console_output_at_epoch_64.png" width="720" height="auto">
<br><br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/FIVES/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this FIVES/test was not so low, and dice_coef not so high as shown below.
<br>
<pre>
loss,0.1148
dice_coef,0.8585
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/FIVES</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for FIVES.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images (2048x2048 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks (2048x2048 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks (2048x2048 pixels) </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/2_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/2_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/2_A.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/4_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/4_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/4_A.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/17_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/17_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/17_A.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/28_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/28_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/28_A.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/32_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/32_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/32_A.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/images/36_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test/masks/36_A.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/FIVES/mini_test_output/36_A.jpg" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. State-of-the-art retinal vessel segmentation with minimalistic models</b><br>
Adrian Galdran, André Anjos, José Dolz, Hadi Chakor, Hervé Lombaert & Ismail Ben Ayed <br>
<a href="https://www.nature.com/articles/s41598-022-09675-y">
https://www.nature.com/articles/s41598-022-09675-y
</a>
<br>
<br>
<b>2. FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation</b><br>
Kai Jin, Xingru Huang, Jingxing Zhou, Yunxiang Li, Yan Yan, Yibao Sun, Qianni Zhang, Yaqi Wang & Juan Ye <br>
<a href="https://www.nature.com/articles/s41597-022-01564-3">https://www.nature.com/articles/s41597-022-01564-3</a>
<br>
<br>
