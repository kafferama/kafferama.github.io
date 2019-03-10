---
layout: post
title: Numpy's einsum for convolutions
date: '2019-03-09'
mathjax: true
---

[//]: # image: /img/hello_world.jpeg

In the past few days, I've been doing the second assignment of Stanford's [CS231n](http://cs231n.stanford.edu/) course on convolutional neural networks. In it, the forward (and backward) passes of a convolutional layer needed to be implemented in a "naive" way, as they provide you with a faster implementation already.

> If you're trying to this by yourself try and do it before reading this post, I give my answer to the problem at the end.

Part of the assignment is to also test the speedup gained by using their function instead of yours, which lends itself to competition between the people that are completing the assignment at the same time as you. This encouraged me to try and do the implementation as fast as I could.

## Convolutions

At first, I had to fully understand how convolutions (in a machine learning sense) worked and what they meant. In this Paul-Louis Pröve's [blog post](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) helped me a lot with its great visualizations, but they were not enough to get me to understand how to code it. It was finally the aforementioned course's [notes](http://cs231n.github.io/convolutional-networks/) which made it clear for me through a great interactive animation of the process.

Convolution, in practice, takes a 4-D \\( w \\) tensor (as in a multidimensional array) and slides it across another 4-D tensor \\( x \\) to produce a set of filtered images. \\( w \\) is of shape \\\( (F, C, HH, WW) \\\), in which \\(F\\) is the number of filters to be applied to each image, \\(C\\) is the number of channels each image has, \\(HH\\) and \\(WW\\) are the height and width of each filter. \\(x\\) is of shape \\((N, C, H, W)\\), in which N is the number of images in the minibatch, \\(C\\) is the number of channels and \\(H\\) and \\(W\\) are the height and width of each image. Knowing this is important because it will help us later in the operations we will carry out.

## Einsum

For my implementation to be fast, I wanted to do most of the work on `numpy`, meaning I had to avoid Python's loops as much as I could. Because of this, the idea of using a function I had seen before and had never dared to use came to my mind. The function was of course `np.einsum`, a function with the ability to operate on multiple dimensions of a multidimensional array with one function call, transferring all the hard work to  `numpy` and saving (potentially) a lot of time.

Even though I had encountered the function before, the [documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) for it leaves a lot to be desired, so I never had a clue on how to use it (but understood what it did). Because of this, I had to look into other sources and I gladly found [this explaination](http://ajcr.net/Basic-guide-to-einsum/) for it by Alex Riley, together with [this great blog post](http://jessicastringham.net/2018/01/01/einsum.html) by Jessica Stringham.

From Alex's post, we can see that the function takes a string that states what we want to do and applies it to the input tensor we give it(` np.einsum('einsum string', tensors)`). An `einsum string` for element-wise multiplication for two tensors A and B looks like this: `ij,ij->ij`. From this we can decompose the string into three parts:

1. The first part names the axes of our A tensor.
2. The second part names the axes of our B tensor.
3. The last part says which axes we want to keep.

Note that the naming of axes is divided by a comma (or multiple commas for more than two tensors), the naming part and the resulting tensor part are divided by a `->` (which can be omitted, but for me its easier if you keep track of everything, see Alex's post for more on that) and the naming has a certain logic to it. This logic is dictated by three simple rules:

1. Repeating a letter in the first part (before the arrow) means to multiply over that axis.
2. Omitting a letter in the final part (after the arrow) means to sum over that axis.
3. The letters after the arrow can have any order, so transposing is possible over any pair of axes.

Jessica's post goes over this same explanation (and for the same application!) probably a lot better, but sadly she did not go too deep into how to name the axes for the task at hand. Which brings us to our way of solving it.

## Our solution

For this solution, I began thinking of the problem for just one image with one filter before scaling it to the whole minibatch (which is quite straight-forward). For one \\( Wp \\) by \\( Hp \\) image I wanted to produce a \\( Wp \\) by \\( Hp \\) filtered image, where Hp and Wp are given by:

    Hp = 1 + (H + 2 * pad - HH) / stride
    Wp = 1 + (W + 2 * pad - WW) / stride

For this, I would have to element-wise multiply the filter weights with the part of the image I'm looking at,  do this for every channel and sum everything up. This would give me the first entry of the filtered image. The only two things left are to look at different parts of the image taking into account the stride parameter and upscaling it to all the filters and all the images in the minibatch. Luckily for us, this is as easy as it could be thanks to einsum.

To formalize everything I just said, let us summarize the operations we just recognized: `dot` product between the weight and the part of the image we're looking at, together with another `dot` product over the channels of the image. We will name the axes of the \\(x\\) minibatch input of shape \\((N, C, H, W)\\) with the letters `ijkl`, this selection is arbitrary and just uses common indexes, which leaves us with \\( x_{ijkl} \\). As we're operating it with \\(w\\) of shape \\( (F, C, HH, WW) \\) we will name it next, but taking into account the operations. 

First of all, we look at multiplications. We want the last two axes (which correspond to weights and pixels) to multiply with each other so we name the last to axes of \\(w\\) accordingly (\\(w_{,,kl}\\)). Every filter has got a weight matrix for every channel so we want to multiply for each channel, giving us \\(w_{,jkl}\\) in accordance with \\(x\\). Lastly, we don't want the dimension of the different filters to be affected, so we want a separate index for it, leaving us with \\(w_{pjkl}\\). (Again the p was chosen randomly).

Second of all, we look at summation. For the letters after the arrow, we want summation to occur on the last three axes of our arrays. Which leaves us with `ip`, an array of shape \\((N, F)\\). Note how this makes the operation happen independently for all the filters in the layer and all the images in the minibatch.

### To recap

We had an array of shape \\((N, C, H, W)\\) and another one with shape \\( (F, C, HH, WW) \\), we wanted an array of shape \\((N, F, 1, 1)\\) (we are making one pixel of the output at a time). We named everything accordingly leaving us with `ijkl,pjkl->ip`, i.e.:

| \\((N, C, H, W)\\) | \\( (F, C, HH, WW) \\) | \\((N, F, 1, 1)\\) |
|--------------------|------------------------|--------------------|
| `ijkl,`            | `pjkl->`               | `ip`               |

## Code

Putting it all together we will get to:

{% highlight python %} 

for a in range(Wp):
    for b in range(Hp):
        out[...,a,b] = np.einsum('ijkl,pjkl->ip',x[..,b*stride:b*stride+HH,b*stride:b*stride+WW], w)

out+= b[np.newaxis, ..., np.newaxis, np.newaxis] 
    
{% endhighlight %}

Where the two for loops are for sliding on both our output and input images with a step of size `stride`. Notice that at the end we reshape our bias term \\(b \\) of shape \\( (F,) \\) in order to be able to add it to every channel in the filter axis.


## Performance

At the beginning of this post, I mentioned how the assignment gave tools to compare your own solution with theirs. Most of the people that were doing the assignment with me saw a boost of \\( ~400x \\) with respect to their code, my solution above had a boost of performance of around \\( ~3x \\), which made it more than a hundred times faster than most of the "naive" implementations my partners came up with. 

It is worth mentioning that the fast solution the course gives you makes use of `np.as_strided`, a function that is also used and very well explained by Jessica's post. I did not want to get into that for this assignment.

## It gets better

To make things even better, `np.einsum` comes with an optimizer attached to it. The optimizer tries to find the optimal sequence of operations in order to perform the sums and multiplications in the best order possible. The way it works is quite well explained in the [documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html). Adding the `optimize = 'greedy'` option made my code only two times slower than the code they came up at Stanford's code, which is an improvement of 30% just by adding an option.

## Note

As a final note I would like to mention that even though the function is called `einsum` and it refers to Einstein's notation, the Einstein notation found in tensor calculus (not multidimensional arrays, but things with actual physical meaning) when an index is repeated it means that summation occurs over that axis and in the result you don't have that dimension anymore, when being strict the index has to be repeated in a certain way. Overall, `einsum` provides you with even more control than Einstein summation convention would, but coming from a physics background it is confusing when they behave differently having the same name. In a similar way calling a multidimensional array a tensor is.

## More resources

Apart from the ones already in the post, you can check:

- [Great post](https://rockt.github.io/2018/04/30/einsum) by Tim Rocktäschel that also includes `einsum` in Tensorflow and PyTorch.
- [Another take](https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/) on it by Olexa Bilaniuk which may be easier to get from a CS perspective. It also has a lot of great examples of different operations.
- [Jessica's post](http://jessicastringham.net/2017/12/31/stride-tricks.html) on using strides if you want to know more about that. 

