---
layout: post
title: Numpy's einsum for convolutions
published: false
date: '2019-03-15'
mathjax: true
---

[//]: # image: /img/hello_world.jpeg

In the last few days I've been doing the second assignment of Stanford's [CS231n](http://cs231n.stanford.edu/) course on convolutional neural networks. In it, the forward (and backward) passes of a convolutional layer needed to be implemented in a "naive" way, as they provide you with a faster implementation already. 

Part of the assignment is to also test the speedup gained by usign their function instead of yours, which lends itself to competition between the people that are completing the assignment at the same time as you. This encouraged me to try and do the implementation as fast as I could.

At first I had to fully understand how convolutions (in a machine learning sense) worked and what they meant. In this Paul-Louis Pröve's [blog post](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) helped me a lot with it's great visualizations, but they were not enough to get me to understand how to code it. It was finally the aforementioned course's [notes](http://cs231n.github.io/convolutional-networks/) which made it clear for me through a great interactive animation of the process.

Convolution, in practice, takes a \\( w_{pjkl} \\) tensor (as in multidimensional array) and slides it across another tensor 




