https://arxiv.org/pdf/2103.00020.pdf

Abstract
=======================================================
SOTA computer vision systems at that time are trained to predict a fixed set of predetermined object categories. This restricted form of super-vision limits their generality and usability because additional labeled data is needed to specify any other visual concept.

If we can learn directly from raw text about images it will be more productive and promising much broad source of supervision.

They demonstrate simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn learn SOTA image representations from scratch on a dataset of 400 million(image, text) pairs collected from the internet. After pre-training natural language is used to reference learned visual concept(or describe new one) enabling **zero-shot transfer(^1)** model to down-stream task.

The model is often competitive with a fully supervised baseline without the need for any dataset specific training.

**(^1)Zero-shot transfer(zero shot learning):** problem setup in deep learning where, at test time, a learner observes samples from classes which were not observed during training, and needs to predict the class that they belong to. 어떤 힌트도 주지않아도 다른 태스크를 잘 수행하는것을 제로샷이라 함. 다른예로는 few shot이 있는데 그건 몇개만 힌트를주고 잘 하는것을 말함.

Upstream - pre training
downstream = fine tuning
EX)기존에는 강아지 판별하는 모델한테 고양이를 판별하라고 하면 upstream의 과정들을 다 뜯어 고쳐야함, 하지만 CLIP을 사용 함 으로써 downstream만 뜯어고쳐도 고양이 판별을 가능하게 한것

ResNet: CNN 변종
data의 layer 를 쌓을때마다 accuracy 가 올라간다는 장점이 있음
그래서 데이터가 비교적 적은 분야에서 주로 쓰임(예: 의료분야)
핵심포인트
========
NLP를 연구하면서 단어들의 연관성이나 상관관계에서 강점을 보인다는것을 발견하고 그것을 computer vision에도 적용을 시키려한 첫 시도이다. 이 논문을 기점으로 Computer vision분야가 또 한번 새로운 도약을 했다고 생각해도 무방. 언어의 연관성, 상관관계등이 사진에도 적용이 되었고 그로인해서 인터넷에 있는 엄청난 양의 날것의 (raw text)데이터를(글과 사진) 가져와서 Training을 시켰을때 Zero-shot transfer가 가능했다는것.
Dimension에 관하여 우리는 고차원에 있는 데이터(이미지, 텍스트)를 잘 표현할 수 있는 가상의 저차원 공간이 있다고 가정을 한다. 그랬을때 예를들어 푸들의 이미지가 있다면 그 이미지를 잘 나타내는 저차원의 공간이 있다는 것. 어떻게 사용될 수 있냐면, 만약 푸들이라는 이미지와 비슷한 텍스트의 차원이 푸들의 가상 저차원공간에 가깝게 나타 날 것이라는것이다. 그말은, 자연스럽게 이미지와 텍스트의 상관관계를 기계가 파악 할 수 있다는것. 이것이 핵심이라고 생각 하면 된다!
Figure 1에서 row는 텍스트, column은 이미지 인코더가 사용되는데, 두가지의 데이터(이미지, 텍스트)를 벡터값으로 변환하여서 cosine similarity(벡터값의 유사도를 계산하는 방법) 를 사용하여 둘의 유사도를 구한 뒤 softmax로 normalize를 해서 가장 확률적으로 가까운 단어를 1로 만들고 나머지는 0으로 만들어버린것.
Introduction and motivating work
=======================================================
Pre-training methods which learn directly from raw text has been revolutionized on NLP. Task-agnostic objectives like autoregressive, and masked language modeling expanded to many order of magnitude in compute, model capacity, and data, steadily improving capabilities. Because of text-to-text developed as standardized input output interface, that made task-agnostic architecture able to do zero-shot transfer to downstream dataset which made possible to remove the need of specialized output heads or dataset specific customization. Flagship system like GPT3 are competitive with bespoke models while require little or no dataset specific training data

These results suggest that aggregate supervision which has access to modern pre-training methods in web-scale collection of text is surpasses high-quality crowd-labeled NLP datasets. But in computer vision, it is still standard practice to pre-trained model on crowd-labeled datasets like ImageNet.

20 years ago, method of improving ***content-based image retrieval*** by training a model to predict the nouns and adjectives in text documents paired with images had been explored.

**Content-Based Image Retrieval (CBIR)** is *"a way of retrieving images from a database"*. In CBIR, a user specifies a query image and gets the images in the database similar to the query image. To find the most similar images, CBIR compares the content of the input image to the database images.

At 2007, Quanttoni demonstrated, ***it was possible to learn more data efficient image representations*** via ****manifold learning*** in weight space of ****classifiers trained to predict words in captions associated with images***.
	and so on about history of improving text to image classification.

These represent the current pragmatic middle ground between "learning from limited amount of supervised 'Gold-Labels'." and learning from practically unlimited amount of raw text.
Natural language is able to express, and therefor supervise, a much wider set of visual concept through generality(자연어는 훨씬 더 넓은 범위의 시각적인 컨셉을 대부분 표현할수 있다.)

***A crucial differences*** between weakly supervised models and recent exploration of leaning image representations directly from natural language ***is scale***

They studied the behavior of image classifiers trained with natural language supervision at large scale. And enabled by large amount of publicly available data of this form on internet, and they created new dataset of 400 million pairs and demonstrated simplified version of ConVIRT trained from scratch, which called ***CLIP***
for Contrastive Language-Image Pretraining which is efficient method of learning from natural language supervision. They studied scalability of CLIP by training eight models two orders of magnitude of compute(2차수 규모의 컴퓨팅을 걸쳐 교육한 8개 모델), and observe transfer performance is able to predict function of compute smoothly.

They found that CLIP learns to perform a wide set of tasks during pre-training including OCR(Optical character recognition), geo-localization, action-recognition, and many others.

Approach
=======================================================
*2.1 Natural Language Supervision
	Core of approach is idea of leaning perception from supervision contained in natural language. 
	Learning from natural language has several potential strengths over other training methods.
		It is much easier to scale natural language supervision compared to standard crowd-source labeling for image classification since it does not require annotations to be in classic "machine learning compatible format" such as canonical 1-to-N majority vote "gold label". 
		Instead method which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet.(자연어를 활용하는 메소드의 장점은 수동적으로 인터넷상의 방대한 양의 텍스트로부터 배울 수 있기때문이다.)
		Learning from natural language also has important advantage over most unsupervised or self-supervised learning approaches.. It doesn't "just" learn a representation but also connects that representation to language which enables flexible zero-shot transfer.(그냥 뜻만 이해하는게 아니라 그 뜻을 언어랑 연결시킨다.)
	2.2 Creating a Sufficient Large Dataset
		3 existing major dataset for computer vision had so small amount of data compare to what they think is enough. So they tried to make new data set called WIT for Web Image Text. The reason they tried to make new dataset was because there is large quantities of data of large language form available publicly on internet. The amount of WIT was 400 million pairs including image and text both.
	2.3 Selecting efficient pre-training method
		They found training efficiency was key to successfully scaling natural language supervision. So they selected final pre-training method based on metric.
		They started with same bag-of-words encoding baseline, they swapped the predictive objective for a contrastive objective in Figure2 and observed a further 4x efficiency improvement in the rate of zero-shot-transfer to ImageNet.
		Given batch of N(image, text) pairs, CLIP is trained to predict which of the N x N possible (image, text) parings across the batch actually occurred. For that CLIP learned multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embedding of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of (N^2) - (N) incorrect pairings.
		Due to large size dataset overfitting isn't major concern.
		They trained CLIP without initialize weights for ImageNet and text encoder.
		They also didn't use non-linear projection between representations and contrastive embedding space.
		Instead they used only linear projection to map from each encoder's representation to multi-modal embedding space.
		They removed text transformation function because many of the pairs in CLIP's pre-training dataset are only a single sentence.
		They also simplified image-transformation function.
	2.4 Choosing and Scaling a Model


