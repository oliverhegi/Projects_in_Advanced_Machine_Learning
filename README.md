# Projects in Advanced Machine Learning

## What’s included:
*	[`World_Happiness_Prediction.ipynb`](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/World_Happiness_Prediction.ipynb): Predicting happiness with the UN World Happiness Data using Deep Learning and classic ML techniques
*	[`Covid_X-Rays.ipynb`](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/COVID_X-Rays.ipynb): Identifying Covid-19 infections in X-Ray images using Transfer Learning and Fire Modules
*	[`Sentiment_Analysis_with_SST.ipynb`](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/SST_Text_Classification.ipynb): Conducting sentiment analysis with LSTM and embeddings layers

These projects were all completed as part of a course on Advanced Machine Learning. Each project focuses on a different areas of Machine Learning (ML). The first project [World_Happiness_Prediction.ipynb](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/World_Happiness_Prediction.ipynb) uses tabular data and compares classic ML techniques with Deep Learning. The second project, [Covid_X-Rays.ipynb](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/COVID_X-Rays.ipynb), is focused on computer vision applications and uses sophisticated Convolutional Neural Networks (CNNs) and Transfer Learning. The third project is about Natural Language Processing (NLP). [Sentiment_Analysis_with_SST.ipynb](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/SST_Text_Classification.ipynb) uses LSTMs and pre-trained embeddings layers for Sentiment Analysis.

## Descriptions:

### [Project 1 – World Happiness Prediction](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/World_Happiness_Prediction.ipynb):

![PCA1](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Images/PCA1.png)

This project uses data from the [U.N. World Happiness Report]( https://worldhappiness.report/ed/2022/) to predict the self-reported happiness of a country. The data is first analyzed and preprocessed using a Principle Component Analysis (PCA). Several regression tree models, such as Random Forest and Gradient Boosting Machines (GBM) are fitted and used to make predictions. Next a neural network is trained, and the results are analyzed using SHAP values.

![PCA2](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Images/PCA2.png)

The chart above shows predictor SHAP values for all countries, ordered by the output value (happiest to the left and least happy to the right). This chart shows that there is diversity in feature importance and not every output can be predicted with the same features.

Key takeaways:
1.	The richness of the data is limited and does not benefit from highly complex models such as Neural Networks and Gradient Boosting Machines. As such, simpler models, such as Random Forest, outperformed more complex models. Simpler tree-based ensembling models were able to benefit from more depth in each tree whereas gradient boosting trees would overfit the data very quickly.
2.	The analysis of the SHAP values has shown that the relative feature-importance varies dramatically between observations. The happiness of some countries can be predicted with just GDP per capita and social support, other countries require a combination of predictors. This, along with a relatively low overall F1 Score / accuracy suggests that (self-reported) happiness is quite challenging to predict, even with a pannel of predictors.

### [Project 2 – COVID X-Rays](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/COVID_X-Rays.ipynb):

![InceptionV3](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Images/InceptionV3.png)

This project uses Deep Learning models to classify X-ray images from healthy patients, patients with COVID-19, and patients with pneumonia. Several different Convolutional Neural Networks (CNNs) are used, with varying degrees of complexity. [Fire modules]( https://paperswithcode.com/method/fire-module) are applied to squeeze convolutional layers and subsequently expand them. 

```
# Create function to define fire modules
def fire(x, squeeze, expand):
  y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, padding='same', activation='relu')(x) 
  y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, padding='same', activation='relu')(y)
  y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, padding='same', activation='relu')(y)
  return tf.keras.layers.concatenate([y1, y3])

# this is to make it behave similarly to other Keras layers
def fire_module(squeeze, expand):
  return lambda x: fire(x, squeeze, expand)
```

Transfer Learning with a [Keras/ImageNet](https://keras.io/api/applications/inceptionv3/) model called InceptionV3 is used to import a pretrained image classification model.

Data augmentation is used when training the models to avoid overfitting. The augmentations include rescaling, rotations, shifts, and horizontal flips.

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
```

![Data_Augmentation](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Images/Data_Augmentation.png)

A CNN with batch normalization performed the best out of all models and achieved a testing accuracy of 94.42%. At the time of submission, this model was ranked number 8 in terms of accuracy out of around 300 submissions to the class [AIModelShare Leaderboard]( https://www.modelshare.ai/detail/model:3338).

Key takeaways:
1.	Even with very rich data, such as image files, a more complex model does not always lead to better results. 
2.	Techniques to avoid overfitting, such as data augmentation, skipped connections (ResNet), regularization, and batch normalization, are usually good practice and can lead to higher out-of-sample accuracy, even if they add additional computation time. 

### [Project 3 – SST Text Classification](https://github.com/oliverhegi/Projects_in_Advanced_Machine_Learning/blob/main/Projects/SST_Text_Classification.ipynb):

This project uses movie reviews from [Stanford Sentiment Treebank (SST)]( https://github.com/oliverhegi/Sentiment_Analysis_with_SST_QMSS_Project3/blob/main/SST_Text_Classification.ipynb) to train sentiment analysis models. Rather than a bag-of-words approach, this project uses embeddings layers and sequential data. The models use LSTMs, 1-Dimensional convolutions (1D CNNs), and Transfer Learning.

```
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=40))

model.add(Conv1D(100, 5, activation='relu')) 
model.add(MaxPooling1D(5))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
```

The best performing model uses an embeddings layer from [Stanford Global Vectors for Word Representation (GloVe)]( https://nlp.stanford.edu/projects/glove/). These embeddings are frozen and a 1D convolution layer is trained on top. The validation accuracy ends up at 79% after 5 epochs.

Key takeaways:
1.	LSTM architectures are very computationally intensive. A 1D convolutional Neural Network with skipped connections (ResNet) and other tricks can produce similar results at a fraction of the training time.
2.	Embeddings matrices can be used to learn semantic relationships. Open-source embeddings matrices such as GloVe are a great starting place for any sentiment analysis model.


## Credits
-	[AIModelShare]( https://www.modelshare.ai/)
