## Machine Learning and Deep Learning Tutorial
_Machine Learning and Deep learning Tutorial_ (shortly _ML/DL Tutorial_) has been initiated to teach undergraduate students in [SEOULTECH](https://en.seoultech.ac.kr/) about basic concepts of machine learning and deep learning with hands-on practices using [scikit-learn](https://scikit-learn.org/) and  [PyTorch](https://pytorch.org/). Even though there are so many good lectures and tutorials on machine learning and deep learning, I would like to share my viewpoint and summary with humble slides and examples because I also have learned many things from open and public lectures, tutorials, and articles. I hope that my slides and examples are also helpful to others.

* To clone this repository (codes and slides): `git clone https://github.com/mint-lab/dl_tutorial.git`
* To fork this repository to your Github: [Click here](https://github.com/mint-lab/dl_tutorial/fork)
* To download codes and slides as a ZIP file: [Click here](https://github.com/mint-lab/dl_tutorial/archive/master.zip)

This tutorial is the last part of my lecture. Its prior knowledge on Python and mathematics is also available in _[Programming Meets Mathematics](https://github.com/mint-lab/prog_meets_math)_.



### ML/DL Lecture Slides
* [Machine Learning with scikit-learn](https://github.com/mint-lab/dl_tutorial/blob/master/slides/ml_tutorial.pdf)
* [Deep Learning with PyTorch](https://github.com/mint-lab/dl_tutorial/blob/master/slides/dl_tutorial.pdf)



### ML Example Codes
:memo: Source codes are enumerated in the order of [its lecture slides](https://github.com/mint-lab/dl_tutorial/blob/master/slides/ml_tutorial.pdf).
* **scikit-learn**
  * [Three Steps of scikit-learn: Instantiation, `fit` and `predict`](https://github.com/mint-lab/dl_tutorial/blob/master/examples/iris_classification.py)
* **Classification**
  * [SVM Classifiers](https://github.com/mint-lab/dl_tutorial/blob/master/examples/iris_classification_svm.py)
  * [Decision Tree Classifiers](https://github.com/mint-lab/dl_tutorial/blob/master/examples/iris_classification_tree.py)
  * [Naive Bayes Classifiers](https://github.com/mint-lab/dl_tutorial/blob/master/examples/iris_classification_bayes.py)
  * [More Classifiers](https://github.com/mint-lab/dl_tutorial/blob/master/examples/iris_classification_more.py)
  * Lab) Breast Cancer Classification [[slides]](https://github.com/mint-lab/dl_tutorial/blob/master/slides/ml01_lab.pdf) [[skeleton code]](https://github.com/mint-lab/dl_tutorial/blob/master/examples/wdbc_classification_skeleton.py)
* **Regression**
  * [Linear Regression](https://github.com/mint-lab/dl_tutorial/blob/master/examples/line_fitting_sklearn.py)
* **Clustering**
  * [K-means Clustering](https://github.com/mint-lab/dl_tutorial/blob/master/examples/iris_clustering_kmeans.py)
* **Data Separation**
  * Lab) Breast Cancer Classification with Cross-validation [[slides]](https://github.com/mint-lab/dl_tutorial/blob/master/slides/ml02_lab.pdf) [[skeleton code]](https://github.com/mint-lab/dl_tutorial/blob/master/examples/wdbc_classification_cv.py)




### DL Example Codes
:memo: Source codes are enumerated in the order of [its lecture slides](https://github.com/mint-lab/dl_tutorial/blob/master/slides/dl_tutorial.pdf).

* **PyTorch**
  * [Creating a Tensor](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch01_create_tensor.py)
  * [Reshaping a Tensor](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch02_reshape_tensor.py)
  * [Line Fitting from Two Points](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch03_line_fitting.py)
  * [CPU vs. GPU-acceleration](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch04_cpu_vs_gpu.py)
  * [Automatic Differentiation](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch05_autograd.py)
  * [Automatic Differentiation - More Analysis](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch05_autograd_analysis.py)
  * [Gradient Descent by Hands](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch06_grad_descent.py)
  * [Gradient Descent by `torch.optim`](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch06_grad_descent_optimizer.py)
  * Lab) Object Detection using YOLO [[slides]](https://github.com/mint-lab/dl_tutorial/blob/master/slides/dl01_lab.pdf) [[skeleton code (py)]](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch_yolo.py) [[skeleton code (ipynb)]](https://github.com/mint-lab/dl_tutorial/blob/master/examples/pytorch_yolo.ipynb)
* **Neural Network**
  * [Visualizing Activation Functions](https://github.com/mint-lab/dl_tutorial/blob/master/examples/dnn_basic_activation_func.py)
  * [Observing Vanishing Gradient - Sigmoid](https://github.com/mint-lab/dl_tutorial/blob/master/examples/dnn_basic_vanishing_grad.py)
  * [Observing Vanishing Gradient - ReLU](https://github.com/mint-lab/dl_tutorial/blob/master/examples/dnn_basic_vanishing_grad_relu.py)
  * The Iris Flower Dataset [[UCI ML Repository]](https://archive.ics.uci.edu/ml/datasets/iris)
    * [Iris Flower Classification - No Class](https://github.com/mint-lab/dl_tutorial/blob/master/examples/dnn_iris2_no_class.py)
    * [Iris Flower Classification - My Style](https://github.com/mint-lab/dl_tutorial/blob/master/examples/dnn_iris2.py)
* **Convolutional Neural Network**
  * The MNIST Dataset [[homepage]](http://yann.lecun.com/exdb/mnist/)
    * [Loading the MNIST Dataset](https://github.com/mint-lab/dl_tutorial/blob/master/examples/cnn_mnist_dataset.py)
    * [Digit Classificaiton with the MNIST Dataset](https://github.com/mint-lab/dl_tutorial/blob/master/examples/cnn_mnist.py)
    * [Loading My Network and Testing It](https://github.com/mint-lab/dl_tutorial/blob/master/examples/cnn_mnist_load.py)
    * [Different Styles for NN Classes](https://github.com/mint-lab/dl_tutorial/blob/master/examples/cnn_mnist_class_style.py)
  * [Visualizing Learning Rate Schedulers](https://github.com/mint-lab/dl_tutorial/blob/master/examples/cnn_basic_lr_scheduler.py)
* **Recurrent Neural Network**
  * The Name2Lang Dataset [[homepage]](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial)
    * [Name2Lang Classification with a Character-level RNN](https://github.com/mint-lab/dl_tutorial/blob/master/examples/rnn_name2lang.py)

> **Note)** All examples contain their basic NN architectures and hyperparameters. One of main objectives in practices will be their performance improvement by changing the architectures and selecting hyperparameters.



### License
* [Beerware](http://en.wikipedia.org/wiki/Beerware)



### Authors
* [Sunglok Choi](http://mint-lab.github.io/sunglok)
