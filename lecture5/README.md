# Basic keras notebooks

In these notebooks you will learn about some basic notionts of keras and neural network training. However, the first implementation covers the evaluation of a machine learning model based on the Naive Bayse approach. Then in the next notebooks the approach is extended to keras with an increasing level of complications. To follow the lecture, you should try the notebooks in this order.

* 1 **NaiveBayes.ipynb** : simple ML approach in python;
* 2 **MNIST_on_a_simple_Keras_example.ipynb** : digit recognition;
* 3 **Tensor Algebra.ipynb** : quick intro to tensorial algebra to learn about layer operations;
* 4 **BostonHousingPriceExample.ipynb** : simple regression neural network;
* 5 **MNIST_Train_evaluation.ipynb** : Learn to manage the training dataset;
* 6 **MNIST_convolutional_model.ipynb** : reimplementation of example 2 & 5 with a convolutional layer.

## Conda environment definition

To use the notebooks available here, please install the following packages with conda.

```
conda create -n keras
conda activate keras
conda install tensorflow-gpu=2.4.1 cudatoolkit jupyter matplotlib scikit-image numpy
conda install -c conda-forge nibabel
```

## References

These examples were inspired by Franchois Chollet "Deep Learning with Python" book and tutorials. [Here](https://github.com/fchollet/deep-learning-with-python-notebooks) you may find the entire tutorial of his book and course. 
