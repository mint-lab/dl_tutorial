## Deep Learning Tutorial with PyTorch
_Deep learning tutorial with PyTorch_ has been initiated to teach undergraduate students in [SeoulTech](https://en.seoultech.ac.kr/) about basic concepts of deep learning and hands-on practices with [PyTorch](https://pytorch.org/). Even though there are so many good lectures and tutorials on deep learning, I would like to share my humble slides and examples because I also have learned many things from open and public lectures, tutorials, and articles. I hope that my slides and examples are also helpful to others.

* Download [my tutorial slides](https://github.com/mint-lab/dl_tutorial/releases/download/misc/dl_tutorial.pdf)
* Download [my practice examples in a ZIP file](https://github.com/mint-lab/dl_tutorial/archive/master.zip)

This tutorial is the last part of my four sequels of tutorials. Many examples and practices are connected each other. Please refer its neighborhood tutorials also.
1. Python Review: From Beginners to Intermediate
2. Programming meets Mathematics: Building Mathematical Intuition with SciPy
3. Machine Learning Tutorial with Scikit-learn
4. [Deep Learning Tutorial with PyTorch](https://github.com/mint-lab/dl_tutorial)


### Practice Examples
Source codes are enumerated in the order of [my tutorial slides](https://github.com/mint-lab/dl_tutorial/releases/download/misc/dl_tutorial.pdf).

* **PyTorch**
  * [Creating a Tensor](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch01_create_tensor.py)
  * [Reshaping a Tensor](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch02_reshape_tensor.py)
  * [Line Fitting from Two Points](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch03_line_fitting.py)
  * [CPU vs. GPU-acceleration](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch04_cpu_vs_gpu.py)
  * [Automatic Differentiation](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch05_autograd.py)
  * [Automatic Differentiation - More Analysis](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch05_autograd_analysis.py)
  * [Gradient Descent by Hands](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch06_grad_descent.py)
  * [Gradient Descent by `torch.optim`](https://github.com/mint-lab/dl_tutorial/blob/master/pytorch06_grad_descent_optimizer.py)
* **Neural Network**
  * [Visualizing Activation Functions](https://github.com/mint-lab/dl_tutorial/blob/master/dnn_basic01_activation_func.py)
  * [Observing Vanishing Gradient - Sigmoid](https://github.com/mint-lab/dl_tutorial/blob/master/dnn_basic02_vanishing_grad.py)
  * [Observing Vanishing Gradient - ReLU](https://github.com/mint-lab/dl_tutorial/blob/master/dnn_basic02_vanishing_grad_relu.py)
  * The Iris Flower Dataset [[UCI ML Repository]](https://archive.ics.uci.edu/ml/datasets/iris)
    * [Iris Flower Classification - No Class](https://github.com/mint-lab/dl_tutorial/blob/master/dnn_iris2_no_class.py)
    * [Iris Flower Classification - My Style](https://github.com/mint-lab/dl_tutorial/blob/master/dnn_iris2.py)
* **Convolutional Neural Network**
  * The MNIST Dataset [[homepage]](http://yann.lecun.com/exdb/mnist/)
    * [Loading the MNIST Dataset](https://github.com/mint-lab/dl_tutorial/blob/master/cnn_mnist_dataset.py)
    * [Digit Classificaiton with the MNIST Dataset](https://github.com/mint-lab/dl_tutorial/blob/master/cnn_mnist.py)
    * [Loading My Network and Testing It](https://github.com/mint-lab/dl_tutorial/blob/master/cnn_mnist_load.py)
    * [Different Styles for NN Classes](https://github.com/mint-lab/dl_tutorial/blob/master/cnn_mnist_class_style.py)
  * [Visualizing Learning Rate Schedulers](https://github.com/mint-lab/dl_tutorial/blob/master/cnn_basic01_lr_scheduler.py)
* **Recurrent Neural Network**
  * The Name2Lang Dataset [[homepage]](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial)
    * [Name2Lang Classification with a Character-level RNN](https://github.com/mint-lab/dl_tutorial/blob/master/rnn_name2lang.py)

> **NOTE:** All examples contain their basic NN architectures and hyperparameters. One of main objectives in practices will be their performance improvement by changing the architectures and selecting hyperparameters.


### Author
* [Sunglok Choi](http://mint-lab.github.io/) (sunglok AT seoultech DOT ac DOT kr)
