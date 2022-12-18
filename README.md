## PROJECT GO EMOTIONS


***Introduction***

In this project, we will improve given a baseline model on the “goemotions” dataset. The goal is to predict the emotions given from 58k Reddit comments. There are 28 emotions possible. We are in a multilabel case where several emotions can be predicted from a comment. 
Link of the dataset : https://www.tensorflow.org/datasets/catalog/goemotions 
It has been created by three contributors (Dana Alon, a-googler, Conchylicultor are their usernames on Github)  (https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/goemotions.py ) and comments have been chosen carefully and curated by this three contributors. They don’t precise how they collect the data.
The goal is to predict one or several emotions involved by a Reddit comment. We have a first Sequential model that we will detail below. We would like to improve it (test accuracy, f1-score), and for this, we have several ways to do so, that we will discover through this report.

***Cleaning***

We remove any columns that are not related to text or emotions labels. ((['id','author','subreddit', 'link_id', 'parent_id', 'created_utc','rater_id','example_very_unclear'])

***First model***

We will use a sequential model with four different layers, which is simple to begin : 
•	An embedding layer : Embedding layer enables us to convert each word into a fixed length vector of defined size. The resultant vector is a dense one with having real values instead of just 0's and 1's. The fixed length of word vectors helps us to represent words in a better way along with reduced dimensions.

•	A global average pooling 1D: It applies average pooling on the spatial dimensions until each spatial dimension is one, and leaves other dimensions unchanged. It replaces the Flatten layers in CNN. It generates one feature map for each corresponding category of the classification task in the last Conv layer.

•	Two dense layers : Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer. We use ReLu as first activation, and then Softmax, which correspond to a multi-label classification.
 
Metrics	on First model baseline on 15 epochs :

* F1-score on average macro	0.084
* Test accuracy	0.359
* Test loss	0.134

The F1-score is really low. We will try to improve it in the next parts, as well as the test accuracy. We will not try to complex too much the model, because it leads to overfitting. Only one or two layers will be added.
Visualization
We observe that the emotion “neutral” isn’t present in the train dataset but is in the test dataset too much represented. We will so remove this category. We observe also that the dataset is unbalanced between the representation of different emotions. We could create some categories of emotions to balance it and simplify the model.

After removing the neutral emotion from the dataset, we can train the baseline model and see if there is a difference on the accuracy or the f1-score.

Metrics	on First model baseline on 15 epochs :

* F1-score on average macro	0.061
* Test accuracy	0.221
* Test loss	0.123

It is strange to observe that the accuracy and the f1-score decreased. However, there are a bit less errors in the test dataset. (1%) We could expect to see the f1-score increasing, thanks to the removal of an unknown label by the model.
Improve the model and create categories of emotions

A large enough deep model should be able to perfect fit a subset of our data. We could try to train the model on a subset of data and reach 100% training accuracy. We may increase the number of neurons and layers until we are able to do so. This indicates that our architecture has enough capacity for the data. Then we could use this architecture on our full data. Before doing so, we can use an LSTM layer instead of only dense layers. We are dealing with sequences and LSTM are more appropriate for this kind of data.
 

We will train this model on the 27 emotions, and on the 6 categories emotions (['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']), that contain all the 27 emotions, and compare the result.
 
Metrics	First model improved with LSTM on 15 epochs (27 emotions) :

* F1-score on average macro	0.117
* Test accuracy	0.252
* Test loss	0.119

It seems it is prone to overfitting if we add more epochs. Early stopping could be a good thing to add. We can see graphically that the best case seems to be around the 7th epoch. Let’s see the results on the 6 categories : ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'].

Metrics	First model improved with LSTM on 15 epochs (6 categories) :

* F1-score on average macro	0.38
* Test accuracy	0.516
* Test loss	0.291

The results on the 6 categories are relevant. It is easier to predict 6 labels instead of 27. The accuracy and the F1—score are higher, however the loss increased also a bit.
In comparison with the baseline model, the results are good. We could go further now by use this results in order to predict each emotion in each category.

***Try normalization***

To improve the results, we also tied to normalize the text. The results were totally out of the range and underfitted. We should increase the learning rate to get a comparable result (because the loss has barely changed). In this way, the loss could decrease until what we have already reach before.
This brings us to hyperparameters.
Hyperparameters
We set hyperparameters to default for this project. A way to improve the model or the F1-score would be to optimize them. We didn’t cover this topic here, but it is a solution that we have to keep in mind.

***Conclusion***

LSTM is more adapted to sequences. Using this type of layers is key to improve a sequence treatment as “goemotions” analysis. 

Cleaning well our data and know make some visualization before analysis is important to have good results and understand the dataset.

Optimizing hyperparameters can change a lot the results if we study and test them rigorously.

Reduce the dimensionality of the labels can improve the accuracy and the F1-score.

Text normalization needs to be tried again.
	
	





