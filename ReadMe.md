### Introduction
Something about Pandemic being over but increased awareness still when it coms to disease spread prevention. Automatically detecting whether people wear face masks is an efficent way to blablabla. 


### Related Work
Since the conception of CNNs by..., these deeplearning architectures using learnable convolutional filters as layers, have established themselves as a suitable tool for image classification (refine sentence please and add citations). From earlier architectures like lenet, vgg16 to inception and resnet, models have become more complex, less efficient and deeper. In the last few years, smaller and more efficent models such as mobile net and efficientnet ...(finish sentence pls). Face mask detection is often a real-time and therefore process-time critical task. A particularly well suited model family - which is also used as a backbone in this work, is Efficientnet-v2. It combines great classification accuracy, a relatively small memory footprint and time-efficency. Pretrained CNNs, often on large datasets such as ImageNet, can be adapted to different downstream tasks via so called transfer-learning. Here the weights are initialized as the best performing ones on the pretraining task and are then either frozen or further finetuned on a task-specific dataset. If the weights are frozen, only a suitable classification head needs to be trained, which speeds up training but might impact accuracy. A more recent development, Vision Transformers (ViTs), which use the attention mechanism intrduced by vashwani et al (ref) and a image-tiling or patch approach. Benchmarks show that ViTs achieve even higher performance on image classification tasks. (ref) However, this increased performance comes with the downside of much larger and complicated models, that are both difficult to train - often requiring enormous amounts of pretraining data - as well as slow output generation at inference time (attention mechanism scales quadratically). 

NOW MORE ON PAPERS THAT DO FACE MASK DETECTION WITH CNNS and OTHER ALGOS. Maybe something on feature extraction with fcns.

### Dataset and Preprocessing
#### Set 1
#### Set 2
(#### Set 3)

#### Augmentation

### Model Architectures
#### Baseline: EfficientNetV2 
#### Custom Approach: Contrastive Feature Disambiguation and XGB
As previously mentioned, transfer-learning with frozen weight comes with the upside of easier and quicker training but often loses some accuracy, as the pretrained features are not perfect representations of some classes, depending on the alignemnt of the pretraining task and the downstream task. We propose a custom solution that combines the frozen weights of the backbone fully convolutional network (FCN) part of efficientnet-v2 and a strong feature-based gradient boosting method, XGBoost. To arrive at feature representations that are meaningful and disambiguated, we first use supervised contrastive pretraining - as described by ... et al in 20XX on the FCN. We then use Global Average pooling and a linear projection head to reduce the feature dimensionality. The disambiguated features are then used as predictors for our binary classification task with XGBoost.

### Evaluation
#### Training 
#### Classification Performance
#### Grad-CAM: Qualitative Assessment
#### Feature Importance
#### Ablation Study
#### Conclusion
