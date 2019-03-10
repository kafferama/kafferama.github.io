---
layout: post
title: Numpy's einsum for convolutions
published: false
date: '2019-03-15'
mathjax: true
---

[//]: # image: /img/hello_world.jpeg
## Problem definition
In the last few days I've been doing the second assignment of Stanford's [CS231n](http://cs231n.stanford.edu/) course on convolutional neural networks. In it, the forward (and backward) passes of a convolutional layer needed to be implemented in a "naive" way, as they provide you with a faster implementation already.

> If you're trying to this by yourself try and do it before reading this post, I give my answer to the problem at the end.

Part of the assignment is to also test the speedup gained by usign their function instead of yours, which lends itself to competition between the people that are completing the assignment at the same time as you. This encouraged me to try and do the implementation as fast as I could.

## Convolutions

At first, I had to fully understand how convolutions (in a machine learning sense) worked and what they meant. In this Paul-Louis Pröve's [blog post](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) helped me a lot with it's great visualizations, but they were not enough to get me to understand how to code it. It was finally the aforementioned course's [notes](http://cs231n.github.io/convolutional-networks/) which made it clear for me through a great interactive animation of the process.

Convolution, in practice, takes a 4-D \\( w \\) tensor (as in multidimensional array) and slides it across another 4-D tensor \\( x \\) to produce a set of filtered images. \\( w \\) is of shape \\( (F, C, HH, WW) \\), in which \\(F\\) is the number of filters to be applied to each image, \\(C\\) is the number of channels each image has, \\(HH\\) and \\(WW\\) are the height and width of each filter. \\(x\\) is of shape \\((N, C, H, W)\\), in which N is the number of images in the minibatch, \\(C\\) is the number of channels and \\(H\\) and \\(W\\) are the height and width of each image. Knowing this is important because it will help us later in the operations we will carry out.

## Einsum

For my implementation to be fast, I wanted to do most of the work on `numpy`, meaning I had to avoid Python's loops as much as I could. Because of this, the idea of using a function I had seen before and had never dared to use came to my mind. The function was of course `np.einsum`, a function with the ability to operate on multiple dimensions of a multidimensional array with one function call, transferring all the hard work to  `numpy` and saving (potentially) a lot of time.

Even though I had encountered the function before, the [documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) for it leaves a lot to be desired, so I never had a clue on how to use it (but understood what it did). Because of this, I had to look into other sources and I gladly found [this explaination](http://ajcr.net/Basic-guide-to-einsum/) for it by Alex Riley, together with [this great blog post](http://jessicastringham.net/2018/01/01/einsum.html) by Jessica Stringham.

From Alex's post we can see that the function takes a string that states what we want to do and applies it to the input tensor we give it(` np.einsum('einsum string',tensors)`). An `einsum string` for  element-wise multiplication for two tensors A and B looks like this: `ij,ij->ij`. From this we can decompose the string into three parts:

1. The first part names the axes of our A tensor.
2. The second part names the axes of our B tensor.
3. The last part says which axes we want to keep.

Note that the naming of axes is divided by a comma (or multiple commas for more than two tensors), the naming part and the resulting tensor part are divided by a `->` (which can be omitted, but for me its easier if you keep track of everything, see Alex's post for more on that) and the naming has a certain logic to it. This logic is dictated by three simple rules:

1. Repeating a letter in the first part (before the arrow) means to multiply over that axis.
2. Omiting a letter in the final part (after the arrow) means to sum over that axis.
3. The letters after the arrow can have any order, so transposing is possible over any pair of axes.







