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

Jessica's post goes over this same explanaition (and for the same application!) probably a lot better, but sadly she did not go too deep into how to name the axes for the task at hand. Which brings us to our way of solving it.

## Our solution

For this solution, I began thinking of the problem for just one image with one filter before scaling it to the whole minibatch (which is quite straight-forward). For one \\( Wp \\) by \\( Hp \\) image I wanted to produce a \\( Wp \\) by \\( Hp \\) filtered image, where Hp and Wp are given by:

	Hp = 1 + (H + 2 * pad - HH) / stride
    Wp = 1 + (W + 2 * pad - WW) / stride

For this I would have to element-wise multiply the filter weights with the part of the image I'm looking at,  do this for every channel and sum everything up. This would give me the first entry of the filtered image. The only two things left are to look at different parts of the image taking into account the stride parameter and upscaling it to all the filters and all the images in the minibatch. Luckily for us this is as easy as it could be thanks to einsum.

To formalize everything I just said, let us summarize the operations we just recognized: `dot` product between the weight and the part of the image we're looking at, together with another `dot` product over the channels of the image. We will name the axes of the \\(x\\) minibatch input of shape \\((N, C, H, W)\\) with the letters `ijkl`, this selection is arbitrary and just uses common indexes, which leaves us with \\( x_{ijkl} \\). As we're operating it with \\(w\\) of shape \\( (F, C, HH, WW) \\) we will name it next, but taking into account the operations. 

First of all, we look at multiplications. We want the last two axes (which correspond to weights and pixels) to multiply with each other so we name the last to axes of \\(w\\) accordingly (\\(w_{,,kl}\\)). Every filter has got a weight matrix for every channel so we want to multiply for each channel, giving us \\(w_{,jkl}\\) in accordance with \\(x\\). Lastly, we don't want the dimension of the different filters to be affected, so we want a separate index for it, leaving us with \\(w_{pjkl}\\). (Again the p was chosen randomly).

Second of all, we look at summation. For the letters after the arrow we want summation to occur on the last three axes of our arrays. Which leaves us with `ip`, an array of shape \\((N, F)\\). Note how this makes the operation happen independently for all the filters in the layer and all the images in the minibatch.

### To recap

We had an array of shape \\((N, C, H, W)\\) and another one with shape \\( (F, C, HH, WW) \\), we wanted an array of shape \\((N, F, 1, 1)\\) (we are making one pixel of the output at a time). We named everything accordingly leaving us with `ijkl,pjkl->ip`, i.e.:

| \\((N, C, H, W)\\) | \\( (F, C, HH, WW) \\) | \\((N, F, 1, 1)\\) |
|--------------------|------------------------|--------------------|
| `ijkl,`            | `pjkl->`               | `ip`               |





