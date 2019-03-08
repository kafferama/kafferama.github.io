---
layout: post
title: Numpy's einsum for convolutions
published: false
date: '2019-03-15'
---

[//]: # image: /img/hello_world.jpeg

In the last few days I've been doing the second assignment of Stanford's [CS231n](http://cs231n.stanford.edu/) course on convolutional neural networks. In it, the forward (and backward) passes of a convolutional layer needed to be implemented in a "naive" way, as they provide you with a faster implementation already. 

Part of the assignment is also to test the speedup gained by usign their function instead of yours, which lends itself to competition between the people that are completing the assignment at the same time as you. This made me try to do the implementation as fast as I could.


