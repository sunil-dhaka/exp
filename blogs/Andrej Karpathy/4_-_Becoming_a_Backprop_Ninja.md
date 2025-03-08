---
layout: default
title: 4 - Becoming a Backprop Ninja
parent: Andrej Karpathy
has_children: false
nav_order: 4
---

# Implementing Backward Pass Manually on the Level of Tensors

Hello everyone! Today, we are continuing our implementation of "make more". So far, we've come up to the point of implementing multilayer perceptrons and our neural net looks like this. We've been implementing this over the last few lectures.

I'm sure everyone is very excited to delve into recurring neural networks and all of their variants. The diagrams look cool and it's very exciting and interesting. We're going to get better results, but unfortunately, I think we have to remain here for one more lecture. 

The reason for that is we've already trained this multilayer perceptron and we are getting pretty good loss. I think we have a pretty decent understanding of the architecture and how it works. But the line of code here that I take issue with is `loss.backward()`. We are taking a PyTorch autograd and using it to calculate all of our gradients along the way. 

I would like to remove the use of `loss.backward()` and I would like us to write our backward pass manually on the level of tensors. I think that this is a very useful exercise for the following reasons.

I actually have an entire blog post on this topic, but I'd like to call back propagation a "leaky abstraction". What I mean by that is back propagation doesn't just make your neural networks work magically. It's not the case that you can just stack up arbitrary Lego blocks of differentiable functions, cross your fingers, and back propagate and everything is great. Things don't just work automatically. It is a leaky abstraction in the sense that you can shoot yourself in the foot if you do not understand its internals. It will not work magically or optimally and you will need to understand how it works under the hood if you're hoping to debug it and if you are hoping to address it in your neural network.

This blog post here from a while ago goes into some of those examples. For example, we've already covered the flat tails of these functions and how you do not want to saturate them too much because your gradients will die. The case of dead neurons which we've already covered as well. The case of exploding or vanishing gradients in the case of recurrent neural networks which we are about to cover.

Also, you will often come across some examples in the wild. This is a snippet that I found in a random code base on the internet where they actually have a very subtle but pretty major bug in their implementation. The bug points at the fact that the author of this code does not actually understand back propagation. What they're trying to do here is they're trying to clip the loss at a certain maximum value. But actually, what they're trying to do is they're trying to limit the gradients to have a maximum value instead of trying to clip the loss at a maximum value. Indirectly, they're basically causing some of the outliers to be actually ignored because when you clip a loss of an outlier, you are setting its gradient to zero.

So, have a look through this and read through it. There are basically a bunch of subtle issues that you're going to avoid if you actually know what you're doing. That's why I don't think it's the case that because PyTorch or other frameworks offer autograd, it is okay for us to ignore how it works.

Now, we've actually already covered autograd and we wrote micrograd. But micrograd was an autograd engine only on the level of individual scalars. The atoms were single individual numbers and I don't think it's enough. I'd like us to basically think about back propagation on the level of tensors as well. 

In summary, I...
# Debugging Neural Networks and the Evolution of Deep Learning

I believe it's a good exercise to manually write your backward pass. It's very valuable as it helps you become better at debugging neural networks and ensures that you understand what you're doing. It makes everything fully explicit, so you're not going to be nervous about what is hidden away from you. In general, we're going to emerge stronger from this exercise. 

A bit of a fun historical note here is that today, writing your backward pass by hand is not recommended and no one does it except for the purpose of exercise. But about 10 years ago in deep learning, this was fairly standard and in fact pervasive. At the time, everyone used to write their own backward pass by hand manually, including myself. It's just what you would do. So we used to write backward pass by hand and now everyone just calls `loss.backward()`. 

We've lost something in this transition and I want to give you a few examples of this. There's a 2006 paper from Jeff Hinton and Russel Salakhutdinov in Science that was influential at the time. This paper was about training some architectures called Restricted Boltzmann Machines, which is basically an autoencoder trained here. 

Around 2010, I had a library for training Restricted Boltzmann Machines. At the time, it was written in Matlab as Python was not used for deep learning pervasively. It was all Matlab, a scientific computing package that everyone would use. Matlab, although barely a programming language, had a very convenient tensor class and was a computing environment where you could run everything on a CPU. It also had very nice plots to go with it and a built-in debugger. 

The code in this package in 2010 that I wrote for fitting Restricted Boltzmann Machines is recognizable. I was creating the data in the XY batches, initializing the neural net with weights and biases just like we're used to. Then, there's the training loop where we actually do the forward pass. At this time, they didn't even necessarily use backpropagation to train neural networks. This, in particular, implements Contrastive Divergence which estimates a gradient. Then, we take that gradient and use it for a parameter update along the lines that we're used to. 

Here's one more example from a paper of mine from 2014 called Fragmented Embeddings. In this paper, I was aligning images and text. It's kind of like a clip if you're familiar with it, but instead of working on the level of entire images and entire sentences, it was working on the level of individual objects and little pieces of sentences. I was embedding them and then calculating a clip-like loss. 

I dug up the code from 2014 of how I implemented this and it was already in Numpy and Python. I was implementing the cost function and it was standard to implement not just the cost but also the backward pass manually. I was calculating the image embeddings, sentence embeddings, the loss function, and then once I had the loss function, I did the backward pass right there. I backpropagated through the loss function and through the neural net and appended regularization. Everything was done by hand manually and you would just write out the backward pass. Then, you would use a gradient checker to make sure that your numerical estimate of the gradient was correct.
# Becoming a Back Propagation Ninja

In this lecture, we will be discussing how to manually implement back propagation in a neural network. This was the standard for a long time, but today, it is more common to use an auto grad engine. However, understanding how to manually implement back propagation can be useful in gaining a deeper understanding of how neural networks work on an intuitive level. 

In our previous lecture, we implemented a two-layer multiplayer perceptron with a batch normalization layer in a Jupyter Notebook. In this lecture, we will keep everything the same, but we will get rid of the loss and backward functions and instead, we will write the backward pass manually.

## Starter Code

The first few cells in our starter code are identical to what we are used to. We are doing some imports, loading the dataset, and processing the dataset. None of this has changed.

We are also introducing a utility function that we will use later to compare the gradients. In particular, we will have the gradients that we estimate manually ourselves and we will have gradients that PyTorch calculates. We will be checking for correctness, assuming of course that PyTorch is correct.

## Initialization

Next, we have the initialization that we are quite used to. We have our embedding table for the characters, the first layer, second layer, and the batch normalization in between. Here is where we create all the parameters.

You will note that I changed the initialization a little bit to be small numbers. Normally, you would set the biases to be all zero. Here, I am setting them to be small random numbers. I am doing this because if your variables are initialized to exactly zero, sometimes what can happen is that it can mask an incorrect implementation of a gradient. By making it small numbers, I am trying to unmask those potential errors in these calculations.

You will also notice that I am using a bias in the first layer, despite having batch normalization right afterwards. This would typically not be what you do because we talked about the fact that you do not need the bias. However, I am doing this here just for fun because we are going to have a gradient with respect to it and we can check that we are still calculating it correctly, even though this bias is superfluous.

## Forward Pass

Next, I am calculating a single batch and then doing a forward pass. You will notice that the forward pass is significantly expanded from what we are used to. The reason that the forward pass is longer is for two reasons. 

Firstly, here we just had an `F.cross_entropy`, but here I am bringing back an explicit implementation of the loss function. Secondly, I have broken up the implementation into manageable chunks. We have a lot more intermediate tensors along the way in the forward pass. This is because we are about to go backwards and calculate the gradients in this back propagation from the bottom to the top. 

So, we are going to go upwards and just like we have, for example, the `log_probs` tensor in a forward pass, in the backward pass we are going to have a `d_log_probs`.
# Back Propagation in PyTorch

In this tutorial, we are going to store the derivative of the loss with respect to the `log_probs` tensor. We will be prepending `D` to every one of these tensors and calculating it along the way of this back propagation.

As an example, we have `a`, `b`, and `raw` here. We're going to be calculating `D_a`, `D_b`, and `D_raw`. Here, I'm telling PyTorch that we want to retain the gradient of all these intermediate values. This is because, in exercise one, we're going to calculate the backward pass. We're going to calculate all these `D` values (D variables) and use the `CNP` function I've introduced above to check our correctness with respect to what PyTorch is telling us.

This is going to be exercise one, where we sort of back propagate through this entire graph. Now, just to give you a very quick preview of what's going to happen in exercise two and below, here we have fully broken up the loss and back propagated through it manually in all the little atomic pieces that make it up. But here, we're going to collapse the loss into a single cross-entropy call. Instead, we're going to analytically derive, using math and paper and pencil, the gradient of the loss with respect to the logits. Instead of back propagating through all of its little chunks one at a time, we're just going to analytically derive what that gradient is and we're going to implement that, which is much more efficient, as we'll see in a bit.

Then, we're going to do the exact same thing for batch normalization. Instead of breaking up batch normalization into all the tiny components, we're going to use pen, paper, and mathematics, and calculus to derive the gradient through the batch normalization layer. So, we're going to calculate the backward pass through the batch normalization layer in a much more efficient expression instead of backward propagating through all of its little pieces independently. This is going to be exercise three.

In exercise four, we're going to put it all together. This is the full code of training this two-layer MLP. We're going to basically insert our manual back prop and we're going to take out `loss.backward()`. You will basically see that you can get all the same results using fully your own code. The only thing we're using from PyTorch is the `torch.tensor` to make the calculations efficient. But otherwise, you will understand fully what it means to forward and backward a neural net and train it, which I think will be awesome.

So, let's get to it. I read all the cells of this notebook all the way up to here and I'm going to erase this and I'm going to start implementing the backward pass, starting with `D_log_probs`. We want to understand what should go here to calculate the gradient of the loss with respect to all the elements of the `log_probs` tensor.

Now, I'm going to give away the answer here, but I wanted to put a quick note here that I think would be most pedagogically useful for you. I recommend you go into the description of this video and find the link to this Jupyter notebook. You can find it both on GitHub and on Google Colab. You don't have to install anything, you'll just go to a website on Google Colab and you can try to implement these derivatives or gradients yourself. Then, if you are not able to, come to my video and see me do it. So, work in tandem, try it first yourself, and then see me give away the answer. I think that'll be most valuable to you, and that's how I recommend you go through this lecture.

We are starting here with `D_log_probs`. `D_log_probs` will hold the derivative of the loss with respect to all the elements of `log_probs`. The shape of this is 32 by 27. So, it's not going to...
# Understanding the Derivative of Loss with Respect to Log Probabilities

You might be surprised to learn that `d_log_probs` should also be an array of size 32 by 27. This is because we want the derivative of loss with respect to all of its elements. Therefore, the sizes of these elements are always going to be equal.

Now, how does `log_probs` influence the loss? The loss is calculated as the negative of `log_probs` indexed with a range of `n` and `yb`, and then the mean of that. Just as a reminder, `yb` is essentially an array of all the correct indices.

What we're doing here is taking the `log_probs` array of size 32 by 27. Then, we are going through every single row and in each row, we are selecting the index 8, then 14, 15, and so on. We're going down the rows using the iterator range of `n` and then we are always selecting the index of the column specified by the tensor `yb`. 

In the zeroth row, we are taking the eighth column, in the first row, we're taking the 14th column, and so on. `log_probs` at this point selects all those log probabilities of the correct next character in a sequence. The shape of this is, of course, 32 because our batch size is 32. These elements get selected and then their mean, and the negative of that, becomes the loss.

To understand the numerical form of the derivative, let's work with a simpler example. Once we've selected these examples, we're taking the mean and then the negative. The loss can be written as the negative of `a + b + c` divided by three. This is how we achieve the mean of three numbers `a`, `b`, and `c`, although we actually have 32 numbers here.

So, what is the derivative of the loss with respect to `a`? If we simplify this expression mathematically, it's just negative one over three. You can see that if we don't just have `a`, `b`, and `c` but we have 32 numbers, then the derivative of the loss with respect to every one of those numbers is going to be one over `n` more generally, because `n` is the size of the batch, 32 in this case. So, the derivative of the loss with respect to `log_probs` is negative 1 over `n` in all these places.

What about the other elements inside `log_probs`? `log_probs` is a large array with a shape of 32 by 27, but only 32 of them participate in the loss calculation. What's the derivative of all the other elements that do not get selected here? Their gradient is intuitively zero because they did not participate in the loss. Most of these numbers inside this tensor do not feed into the loss, so if we were to change these numbers, then the loss doesn't change. This is another way of saying that the derivative of the loss with respect to them is zero, they don't impact it.

Here's a way to implement this derivative: we start out with `torch.zeros_like(log_probs)`. This is going to create an array of zeros exactly in the shape of `log_probs`. Then, we need to set the derivative of negative 1 over `n` inside exactly these locations. We can do this by indexing `log_probs` in the identical way.
# Understanding PyTorch Backpropagation

In this tutorial, we will be discussing the backpropagation process in PyTorch. We will be using the `d_log_props` function as our derivative candidate. Let's start by uncommenting the first line and checking if this is correct.

```python
# Uncomment the first line
# CMP ran and let's go back to CMP
```

What the code is doing is calculating if the value calculated by us, which is `DT`, is exactly equal to `T.grad` as calculated by PyTorch. This is done by making sure that all the elements are exactly equal and then converting this to a single Boolean value. We don't want the Boolean tensor, we just want a Boolean value.

Next, we are making sure that if they're not exactly equal, maybe they are approximately equal because of some floating point issues, but they're very, very close. Here we are using `torch.allclose`, which has a little bit of a wiggle available because sometimes you can get very, very close but if you use a slightly different calculation because of floating point arithmetic, you can get a slightly different result. This is checking if you get an approximately close result.

We are also checking the maximum value that has the highest difference and what is the difference in the absolute value difference between those two. We are printing whether we have an exact equality, an approximate equality, and what is the largest difference.

From the results, we see that we actually have exact equality and so therefore of course we also have an approximate equality and the maximum difference is exactly zero. Basically, our `d_log_props` is exactly equal to what PyTorch calculated to be `logprops.grad` in its backpropagation. So far, we're working pretty well.

Let's now continue our backpropagation. We have that `log_props` depends on `probs` through a log. All the elements of `probs` are being element-wise applied log to. If we want `d_props` then remember your micrograph training. We have a log node that takes in `probs` and creates `log_probs` and the `probs` will be the local derivative of that individual operation log times the derivative loss with respect to its output which in this case is `d_log_props`.

The local derivative of this operation is taking log element-wise and we can see from the alpha function that `d` by `dx` of `log` of `x` is simply one over `x`. Therefore, in this case, `x` is `probs` so we have `d` by `dx` is one over `x` which is one over `probs`. This is the local derivative and then we want to chain it. This is the chain rule times `d_log_props`.

```python
# Uncomment this and run the cell in place
# The derivative of props as we calculated here is exactly correct
```

Notice here how this works. `probs` is going to be inverted and then element-wise multiplied here. If your `probs` is very close to one, that means your network is currently predicting the character correctly, then this will become one over one and `d_log_probs` just gets passed through.

But if your probabilities are incorrectly assigned, so if the correct character here is getting a very low probability then 1.0 dividing by it will boost this and then multiply by the `log_props`. Basically, what this line is doing intuitively is it's taking the examples that have a very low probability.
# Understanding PyTorch Backward Pass

Currently, we are assigned to boosting their gradient. You can look at it that way. Next up is `CountSumImp`. We want the reverse of this. Now, let me just pause here and introduce what's happening here in general because I know it's a little bit confusing. 

We have the logits that come out of the neural net. What I'm doing is I'm finding the maximum in each row and I'm subtracting it for the purposes of numerical stability. We talked about how if you do not do this, you run into numerical issues if some of the logits take on too large values because we end up exponentiating them. So, this is done just for safety numerically. 

Then, here's the exponentiation of all the logits to create our counts. We want to take the sum of these counts and normalize so that all of the probabilities sum to one. Now, here instead of using `1 / CountSum`, I use `**-1` (raised to the power of negative one). Mathematically, they are identical. I just found that there's something wrong with the PyTorch implementation of the backward pass of division. It gives a real result but that doesn't happen for `**-1`. That's why I'm using this formula instead. 

Basically, all that's happening here is we got the logits, we're going to exponentiate all of them, and we want to normalize the counts to create our probabilities. It's just that it's happening across multiple lines. 

So now, we want to first take the derivative. We want to backpropagate into `CountSumImp` and then into counts as well. We actually have to be careful here because we have to scrutinize and be careful with the shapes. `Counts.shape` and `CountSumImp.shape` are different. In particular, `Counts` is `32x27` but `CountSumImp` is `32x1`. 

In this multiplication here, we also have an implicit broadcasting that PyTorch will do because it needs to take this column tensor of 32 numbers and replicate it horizontally 27 times to align these two tensors so it can do an element-wise multiply. 

So really what this looks like is the following using a toy example again. What we really have here is `Probs = Counts * CountSumImp`. So, `C = A * B`. But `A` is `3x3` and `B` is just `3x1` (a column tensor). PyTorch internally replicated the elements of `B` across all the columns in this multiplication. 

Now, we're trying to backpropagate through this operation to `CountSumImp`. When we're calculating this derivative, it's important to realize that these two look like a single operation but actually, it's two operations applied sequentially. The first operation that PyTorch did is it took this column tensor and replicated it across all the columns basically 27 times. That's the first operation, it's a replication. Then the second operation is the multiplication. 

So let's first backpropagate through the multiplication. If these two arrays are of the same size and we just have `A` and `B` both of them `3x3`, then how do we backpropagate through a multiplication? If we just have scalars and not tensors then if you have `C = A * B` then what is the derivative of `C` with respect to `B`? Well, it's just `A`. And so that's the local derivative. 

So here in our case, undoing the multiplication and backpropagating.
# Backpropagation through the Replication

Through just the multiplication itself, which is element-wise, we get the local derivative. In this case, it's simply 'counts' because 'counts' is the 'a'. So, this is the local derivative and then times the chain rule D props. This here is the derivative or the gradient but with respect to replicated 'B'. 

However, we don't have a replicated 'B', we just have a single 'B' column. So, how do we now backpropagate through the replication? Intuitively, this 'B1' is the same variable and it's just reused multiple times. You can look at it as being equivalent to a case we've encountered in micrograd. 

In micrograd, we had an example where a single node has its output feeding into two branches of the graph until the last function. We discussed how the correct thing to do in the backward pass is to sum all the gradients that arrive at any one node. So, across these different branches, the gradients would sum. If a node is used multiple times, the gradients for all of its uses sum during backpropagation. 

Here, 'B1' is used multiple times in all these columns and therefore the right thing to do here is to sum horizontally across all the rows. I'm going to sum in Dimension one but we want to retain this Dimension so that the counts sum end and its gradient are going to be exactly the same shape. We want to make sure that we keep them as true so we don't lose this dimension and this will make the count sum M be exactly shape 32 by 1. 

Running this, we see that we get an exact match. The derivative is exactly correct. 

Now, let's also backpropagate into 'counts' which is the other variable here to create 'probs'. From 'probs' to 'count sum INF' we just did that, let's go into 'counts' as well. The 'counts' are 'a' so DC by d 'a' is just 'B' so therefore it's 'count sum INF' and then times chain rule the 'probs'. 

Now 'count sum INF' is 32 by 1 and 'probs' is 32 by 27. Those will broadcast fine and will give us 'dcounts'. There's no additional summation required here. There will be a broadcasting that happens in this multiply here because 'count sum INF' needs to be replicated again to correctly multiply 'probs' but that's going to give the correct result. 

As far as the single operation is concerned, we backpropagate from 'probs' to 'counts' but we can't actually check the derivative 'counts'. I have it much later on and the reason for that is because 'count sum INF' depends on 'counts' and so there's a second Branch here that we have to finish because 'count sum INF' backpropagates into 'count sum' and 'count sum' will backpropagate into 'counts'. 

So, 'counts' is a node that is being used twice. It's used right here in 'probs' and it goes through this other Branch through 'count sum INF'. So even though we've calculated the first contribution of it, we still have to calculate the second contribution of it later. 

Continuing with this Branch, we have the derivative for 'count sum INF' now we want the derivative of 'count sum'. 'D count sum' equals what is the local derivative of this operation. This is basically an element-wise one over 'count sum'. So 'count sum' raised to the power of negative one is the same as one over 'count sum'. If we go to Wolfram Alpha, we see that 'x' to the negative one 'D' by 'D' by 'D' by 'DX' of it is basically Negative 'X' to the negative 2.
# Understanding the Concept of Derivatives in Machine Learning

"Over squared" is the same as negative X to the negative two. So, the sum here will be local, and the derivative is going to be negative. The sum counts to the negative two, which is the local derivative times the chain rule. Let's uncomment this and check that I am correct. 

We have perfect equality, and there's no sketchiness going on here with any shapes because these are of the same shape. Next up, we want to back-propagate through this line. We have that count sum, which is the count sum along the rows. 

Keep in mind that counts are 32 by 27, and count sum is 32 by 1. In this back-propagation, we need to take this column of derivatives and transform it into a two-dimensional array of derivatives. 

We're taking in some kind of an input, like a three by three matrix A, and we are summing up the rows into a column tensor B1, B2, B3. That is basically this. Now we have the derivatives of the loss with respect to B, all the elements of B, and now we want the derivative loss with respect to all these little A's. 

How do the B's depend on the A's is basically what we're after. What is the local derivative of this operation? We can see here that B1 only depends on these elements here. The derivative of B1 with respect to all of these elements down here is zero. But for these elements here like A11, A12, etc., the local derivative is one. 

So, the derivative of B1 with respect to A11, for example, is one. So it's one, one, and one. When we have the derivative of loss with respect to B1, the local derivative of B1 with respect to these inputs is zeros here, but it's one on these guys. 

In the chain rule, we have the local derivative times the derivative of B1. Because the local derivative is one on these three elements, the local derivative multiplying the derivative of B1 will just be the derivative of B1. 

You can look at it as a router. Basically, an addition is a router of gradient. Whatever gradient comes from above, it just gets routed equally to all the elements that participate in that addition. So in this case, the derivative of B1 will just flow equally to the derivative of A11, A12, and A13. 

If we have a derivative of all the elements of B in this column tensor, which is D count sum that we've calculated just now, we basically see that what that amounts to is all of these are now flowing to all these elements of A, and they're doing that horizontally. 

We want to take the D count sum of size 30 by 1 and replicate it 27 times horizontally to create a 32 by 27 array. There are many ways to implement this operation. You could, of course, just replicate the tensor, but I think maybe one clean one is that the counts are simply torch dot ones, like a two-dimensional array of ones in the shape of counts, so 32 by 27 times D count sum. 

This way, we're letting the broadcasting here basically implement the replication. But then we have to also be careful because D counts were already calculated earlier here, and that was just the first branch. We're now finishing the second branch, so we need to make sure that these gradients add, so plus equals. 

Let's comment out the comparison and make sure, crossing fingers, that we have the correct result. PyTorch agrees with us on this gradient as well.
# Understanding the Derivatives of Logits and Logit Maxes

Hopefully, we're getting the hang of this now. We're dealing with element-wise operations of norm logits. Now, we want to calculate the derivative of norm logits. Because it's an element-wise operation, everything is very simple. 

What is the local derivative of e to the x? It's famously just e to the x. So, this is the local derivative. We've already calculated it and it's inside the counts, so we may as well potentially just reuse counts. That is the local derivative. As funny as it looks, constant decount is the derivative of the norm logits. 

Now, let's erase this and verify it. So, that's the norm logits. We are here on this line now, the norm logits. We have that and we're trying to calculate the logits and deloget maxes. 

Back propagating through this line, we have to be careful here because the shapes are not the same and so there's an implicit broadcasting happening here. Norm logits has this shape 32 by 27, logits does as well, but logit maxes is only 32 by one. So, there's a broadcasting here in the minus. 

Here, I try to sort of write out a two example again. We basically have that this is our C equals A minus B. We see that because of the shape, these are three by three but this one is just a column. For example, every element of C, we have to look at how it came to be. Every element of C is just the corresponding element of A minus the associated B. 

So, it's very clear now that the derivatives of every one of these C's with respect to their inputs are one for the corresponding A and it's a negative one for the corresponding B. Therefore, the derivatives on the C will flow equally to the corresponding A's and then also to the corresponding B's. But then in addition to that, the B's are broadcast so we'll have to do the additional sum just like we did before. Of course, the derivatives for B's will undergo a minus because the local derivative here is negative one. So, DC three two by DB3 is negative one. 

Let's just implement that. Basically, delugits will be exactly copying the derivative on norm logits. So, delugits equals the norm logits and I'll do a dot clone for safety, so we're just making a copy. Then we have that the loaded maxes will be the negative of the non-legits because of the negative sign. 

We have to be careful because logit maxes is a column. Just like we saw before, because we keep replicating the same elements across all the columns, then in the backward pass because we keep reusing this, these are all just separate branches of use of that one variable. Therefore, we have to do a sum along one with keepdim equals true so that we don't destroy this dimension. Then the logit maxes will be the same shape. 

We have to be careful because this deloaches is not the final deloaches. That's because not only do we get gradient signal into logits through here but the logit maxes is a function of logits and that's a second branch into logits. So, this is not yet our final derivative for logits. We will come back later for the second branch. For now, the logit maxes is the final derivative. 

Let's just run this and logit maxes hit by torch agrees with us. So, that was the derivative into through this line. Now, before we move on, I want to pause here briefly and I want to look at these logit maxes and especially their...
# Gradients and Numerical Stability in Softmax Implementation

In a previous lecture, we discussed the importance of numerical stability in the implementation of softmax. The primary reason for this is to ensure that the softmax doesn't overflow. 

We use a max function to guarantee that the highest number in each row of logits is zero, making the softmax safe. This is because if you add or subtract any value equally to all the elements in a row of the logits tensor, the value of the probabilities will remain unchanged. 

This has repercussions. If changing the logits max does not change the probabilities and therefore does not change the loss, then the gradient on the logits max should be zero. In other words, saying those two things is the same. 

We hope that this is very small numbers, so we hope this is zero. However, due to floating point inconsistencies, this doesn't come out exactly zero. Only in some of the rows it does, but we get extremely small values like 1e-9 or 10. This tells us that the values of logits max are not impacting the loss as they shouldn't. 

It feels kind of weird to backpropagate through this branch. If you have any implementation of `f.cross_entropy` in PyTorch and you block together all these elements and you're not doing the backpropagation piece by piece, then you would probably assume that the derivative through here is exactly zero. 

This branch is only done for numerical stability. But it's interesting to see that even if you break up everything into the full atoms and you still do the computation with respect to numerical stability, the correct thing happens and you still get very small gradients here. This reflects the fact that the values of these do not matter with respect to the final loss.

Let's now continue backpropagation through this line. We've just calculated the logits max and now we want to backpropagate into logits through this second branch. 

In PyTorch, the max function returns both the values and the indices at which those values account for the maximum value. In the forward pass, we only used values because that's all we needed. But in the backward pass, it's extremely useful to know where those maximum values occurred. 

We have the indices at which they occurred and this will help us do the backpropagation. The derivative flowing through here should be one times the local derivatives is 1 for the appropriate entry that was plucked out, and then times the global derivative of the logits max. 

What we're doing here is taking the `deloachet_max` and scattering it to the correct positions in these logits from where the maximum values came.
# Implementing a One-Hot Vector in PyTorch

I've come up with a single line of code that implements a one-hot vector in PyTorch. Let's start by creating a tensor of zeros and then populating the correct elements. We use the indices here and set them to be one. However, you can also use the `one_hot` function in PyTorch.

```python
F.one_hot(torch.max(input, dim=1).indices, num_classes=27)
```

In the code above, I'm taking the indices of the maximum values over the first dimension and telling PyTorch that the dimension of every one of these tensors should be 27. This will create an array where the maximum values from each row are represented as one, and all the other elements are zero. This is what we call a one-hot vector.

Next, I'm multiplying this one-hot vector by the logit maxima. Keep in mind that this is a column of 32 by 1. When I'm doing this multiplication, the logit maxima will broadcast and that column will get replicated. An element-wise multiplication will ensure that each of these just gets routed to whichever one of these bits is turned on.

This is another way to implement this kind of operation, and both methods can be used. I'm using `+=` because we already calculated the logits here and this is not the second branch.

Let's look at the logits and make sure that this is correct. As expected, we have exactly the correct answer.

# Matrix Multiplication and Bias Offset in a Linear Layer

Next up, we want to continue with logits here that is an outcome of a matrix multiplication and a bias offset in this linear layer. I've printed out the shapes of all these intermediate tensors. We see that logits is of course 32 by 27 as we've just seen. Then the `H` here is 32 by 64. These are 64-dimensional hidden states and then this `W` matrix projects those 64-dimensional vectors into 27 dimensions. There's also a 27-dimensional offset, which is a one-dimensional vector.

We should note that this plus here actually broadcasts because `H` multiplied by `W2` will give us a 32 by 27. Then this plus `B2` is a 27-dimensional vector. In the rules of broadcasting, what's going to happen with this bias vector is that this one-dimensional vector of 27 will get aligned with a padded dimension of one on the left and it will basically become a row vector. Then it will get replicated vertically 32 times to make it 32 by 27 and then there's an element-wise multiplication.

# Backpropagation from Logits to Hidden States

The question is, how do we backpropagate from logits to the hidden states, the weight matrix `W2`, and the bias `B2`? You might think that we need to go to some matrix calculus and then we have to look up the derivative for a matrix multiplication. But actually, you don't have to do any of that. You can go back to first principles and derive this yourself on a piece of paper.

Specifically, what I like to do and what I find works well for me is to find a specific small example that you then fully write out. Then, in the process of analyzing how that individual small example works, you will understand the broader pattern and you'll be able to generalize and write out the full general formula for how these derivatives flow in an expression like this. Let's try that out.

*Pardon the low-budget production here.*
# Understanding Matrix Multiplication and Derivatives

What I've done here is written out a mathematical concept on a piece of paper. What we are interested in is a scenario where we multiply 'B' plus 'C' to create 'D'. We have the derivative of the loss with respect to 'D' and we'd like to know what the derivative of the loss is with respect to 'A', 'B', and 'C'.

These are small two-dimensional examples of matrix multiplication. We have a 2x2 matrix multiplied by another 2x2 matrix, plus a vector of just two elements, 'C1' and 'C2', which gives us another 2x2 matrix. 

Notice here that I have a bias vector called 'C'. The bias vector is 'C1' and 'C2' but as I described, that bias vector will become a row vector in the broadcasting and will replicate vertically. That's what's happening here as well. 'C1' and 'C2' are replicated vertically and we see how we have two rows of 'C1' and 'C2' as a result.

When I say "write it out", I mean breaking up this matrix multiplication into the actual process that's going on under the hood. As a result of matrix multiplication and how it works, 'D11' is the result of a dot product between the first row of 'A' and the first column of 'B'. So, 'A11 B11' plus 'A12 B21' plus 'C1', and so on for all the other elements of 'D'. 

Once you actually write it out, it becomes obvious. This is just a bunch of multipliers and additions. We know from micrograd how to differentiate multiplies and adds. So, this is not scary anymore. It's not just matrix multiplication, it's just a bit tedious. But this is completely tractable. We have 'DL' by 'D' for all of these and we want 'DL' by all these little other variables. So, how do we achieve that and how do we actually get the gradients?

Let's, for example, derive the derivative of the loss with respect to 'A11'. We see here that 'A11' occurs twice in our simple expression and influences 'D11' and 'D12'. So, what is 'DL' by 'DA11'? Well, it's 'DL' by 'D11' times the local derivative of 'D11' which in this case is just 'B11' because that's what's multiplying 'A11' here. 

Likewise, the local derivative of 'D12' with respect to 'A11' is just 'B12' and so 'B12' will, in the chain rule, therefore multiply 'DL' by 'D12'. And then because 'A11' is used both to produce 'D11' and 'D12', we need to add up the contributions of both of those chains that are running in parallel. That's why we get a plus, just adding up those two contributions. That gives us 'DL' by 'DA11'. 

We can do the exact same analysis for all the other elements of 'A' and when you simply write it out, it's just super simple. Taking gradients on expressions like this, you find that this matrix 'DL' by 'DA' that we're after, if we just arrange all of them in the same shape as 'A' takes, so 'A' is just a 2x2 matrix, so 'DL' by 'DA' here will be also just the same shape matrix with the derivatives. 

We see that actually we can express what we've written out here as a matrix multiplication. It just so happens that all of these formulas that we've derived here by taking gradients can actually be expressed as a matrix multiplication. In particular, we see that it is the matrix multiplication of these two array matrices. 

So, it is the 'DL' by 'D' and then matrix multiplying 'B' but 'B' transpose actually. So you see that 'B21' and 'B12' have changed place, whereas before we had of course 'B11', 'B12', 'B21', 'B22'. So you see that this other.
# Understanding Matrix Transposition and Derivatives

Matrix B is transposed, and through simple reasoning, we can break up the expression in the case of a very simple example. We find that DL by d a is simply equal to DL by DD Matrix. 

We also want the derivative with respect to B and C. For B, I'm not actually doing the full derivation because it's not deep, it's just annoying and exhausting. You can actually do this analysis yourself. You'll also find that if you take these expressions and differentiate with respect to B instead of A, you will find that DL by DB is also a matrix multiplication. In this case, you have to take the Matrix A and transpose it and Matrix multiply that with DL by DD. That's what gives you DL by DB. 

For the offsets C1 and C2, if you again just differentiate with respect to C1, you will find an expression like this, and C2 an expression like this. Basically, you'll find the DL by DC is simply because they're just offsetting these Expressions. You just have to take the DL by DD Matrix of the derivatives of D and you just have to sum across the columns and that gives you the derivatives for C.

In short, the backward Paths of a matrix multiply is a matrix multiply. Instead of just like we had D equals A times B plus C in the scalar case, we arrive at something very similar but now with a matrix multiplication instead of a scalar multiplication. So the derivative of D with respect to A is DL by DD Matrix multiplied by B transpose and here it's A transpose multiplied by DL by DD but in both cases, it's a matrix multiplication with the derivative and the other term in the multiplication. For C, it is a sum.

Now, I'll tell you a secret. I can never remember the formulas that we just arrived for back propagation information multiplication and I can back propagate through these Expressions just fine. The reason this works is because the dimensions have to work out. 

Let me give you an example. Say I want to create DH, then what should the H be? Number one, I have to know that the shape of DH must be the same as the shape of H, and the shape of H is 32 by 64. The other piece of information I know is that DH must be some kind of matrix multiplication of the logits with W2. Delogits is 32 by 27 and W2 is a 64 by 27. There is only a single way to make the shape work out in this case and it is indeed the correct result. In particular, here H needs to be 32 by 64. The only way to achieve that is to take delogits and Matrix multiply it with W2 but I have to transpose it to make the dimensions work out. So W2 transpose and it's the only way to make these to Matrix multiply those two pieces to make the shapes work out and that turns out to be the correct formula. 

Similarly, now if I want DW2, well I know that it must be a matrix multiplication of D logits and H and maybe there's a few transposes in there as well. I don't know which way it is so I have to come to W2 and I see that its shape is 64 by 27 and that has to come from some multiplication of these two. To get a 64 by 27, I need to take...
# Backpropagation in Neural Networks

Firstly, we need to transpose the matrix and then perform matrix multiplication. This will result in a 64 by 32 matrix. Next, we need to multiply this with a 32 by 27 matrix, which will give us a 64 by 27 matrix. It's crucial to ensure that this is multiplied with the logistic function in such a way that the dimensions work out. The only way to achieve this is to use matrix multiplication.

If we look at the code, we can see that this is exactly what's happening. We have a transpose function for 'H' multiplied with 'deloaches', which is 'W2'. Then 'db2' is simply the vertical sum. In the same way, there's only one way to make the shapes work out. We don't have to remember that it's a vertical sum along the zero axis because that's the only way that this makes sense. The shape of 'B2' is 27, so in order to get 'deloaches' here, which is 30 by 27, we know that it's just a sum over 'deloaches' in some direction. That direction must be zero because we need to eliminate this dimension.

Next, we have the backward pass for the linear layer. We can uncomment these three lines of code and check that we got all the three derivatives correct. When we run the code, we see that 'h', 'wh', and 'B2' are all exactly correct. So, we have successfully backpropagated.

Next, we already have the derivative for 'h' and we need to backpropagate through 'tanh' into 'h preact'. We want to derive 'DH preact'. We have to backpropagate through a 'tanh' function. We've already done this in micrograd and we remember that 'tanh' has a very simple backward formula.

Unfortunately, if we just put in 'D' by 'DX' of 'tanh' of 'X' into Wolfram Alpha, it tells us that it's a hyperbolic secant function squared of 'X', which is not exactly helpful. But luckily, Google image search gives us the simpler formula. If we have that 'a' is equal to 'tanh' of 'Z', then 'd a' by 'DZ' by propagating through 'tanh' is just one minus 'a' squared. Note that '1' minus 'a' squared, 'a' here is the output of the 'tanh', not the input to the 'tanh' 'Z'. So, the 'D A' by 'DZ' is here formulated in terms of the output of that 'tanh'.

In our case, that is '1' minus the output of 'tanh' squared, which here is 'H'. So it's 'h' squared and that is the local derivative and then times the chain rule 'DH'. That is going to be our candidate implementation.

Next, we have 'DH preact' and we want to backpropagate into the gain, the 'B', and 'raw' and the 'B' and bias. These are the parameters of the batch norm that take the 'B' and 'raw' that is exact unit Gaussian and then scale it and shift it. These are the parameters of the batch norm.

Now, we do have a multiplication, but it's worth noting that this multiplication is very different from the matrix multiplication. Matrix multiplications are dot products between rows and columns of these matrices. This is an element-wise multiplication, so things are quite a bit simpler. However, we do have to be careful with some of the broadcasting happening in this operation.
# Understanding the Batch Norm Layer

In this article, we will delve into the intricacies of the Batch Norm (BN) layer, focusing on the lines of code that make it work. We will also discuss the importance of ensuring that all the shapes work out fine and that the broadcasting is correctly back-propagated.

Let's start with the BN gain and BN bias. The shapes of BN gain and BN bias are 1 by 64, while the shapes of preact and raw are 32 by 64. We need to be careful with these shapes and ensure that they work out correctly. 

The local derivative in the case of a times b equals c is just b, the other one. So, the local derivative is just b and raw, and then times chain rule, so DH preact. This is the candidate gradient. 

However, we need to be careful because BN gain is of size 1 by 64, but this would be 32 by 64. The correct thing to do in this case is to sum because it's being replicated. Therefore, all the gradients in each of the rows that are now flowing backwards need to sum up to that same tensor DB and gain. We have to sum across all the examples, which is the direction in which this gets replicated.

We also need to be careful because BN gain is of shape 1 by 64. So, in fact, I need to keep them as true, otherwise, I would just get 64. I don't actually remember why the BN gain and the BN bias I made them be 1 by 64, but the biases B1 and B2 I just made them be one-dimensional vectors. They're not two-dimensional tensors. I can't recall exactly why I left the gain and the bias as two-dimensional but it doesn't really matter as long as you are consistent and you're keeping it the same.

Next up, we have BN raw. DB and raw will be BN gain multiplying DH preact, that's our chain rule. We have to be careful with the dimensions of this. DH preact is 32 by 64. BN gain is 1 by 64. So, it will just get replicated and to create this multiplication which is the correct thing because in a forward pass it also gets replicated in just the same way. So, in fact, we don't need the brackets here, we're done, and the shapes are already correct.

Finally, for the bias, it's very similar to the bias we saw in the linear layer. We see that the gradients from each preact will simply flow into the biases and add up because these are just offsets. Basically, we want this to be DH preact but it needs to sum along the right dimension. Similar to the gain, we need to sum across the zeroth dimension, the examples, because of the way that the bias gets replicated vertically. We also want to have keep them as true. This will take this and sum it up and give us a 1 by 64.

This is the candidate implementation. It makes all the shapes work. To check that we are getting the correct result for all the three tensors, we can uncomment these three lines. Indeed, we see that all of that got back-propagated correctly. So now we get to the batch norm layer, we see how here BN gain and BN bias are the key elements.
# Back Propagation and Batch Normalization

In this article, we will discuss the process of back propagation and batch normalization. We will break down the batch form into manageable pieces so we can back propagate through each line individually. 

Let's start with the parameters. The back propagation ends with `B` and `raw` now being the output of the standardization. Here, we are breaking up the batch form into manageable pieces so we can back propagate through each line individually. 

The variable `BN_mean_I` is the sum. `BN_diff` is `x` minus `mu`. `BN_div_2` is `x` minus `mu` squared, which is inside the variance. `BN_VAR` is the variance, represented as Sigma Square. This is `BN_bar` and it's basically the sum of squares. This is the `x` minus `mu` squared and then the sum. 

You'll notice one departure here. It is normalized as 1 over `m`, which is the number of examples. Here, I'm normalizing as one over `n` minus 1 instead of `N`. This is deliberate and I'll come back to that in a bit when we are at this line. It is something called the Bessel's correction. 

`BN_var_inv` then becomes basically `BN_var` plus Epsilon. Epsilon is one negative five and then it's one over the square root. This is the same as raising to the power of negative 0.5 because 0.5 is the square root and then negative makes it one over the square root. 

`BN_Bar_M` is a one over this denominator here and then we can see that `BN_raw`, which is the `X_hat` here, is equal to the `BN_diff` (the numerator) multiplied by the `BN_bar_in`. 

The line here that creates `pre-h`, `pre-act` was the last piece we've already back propagated through. So now what we want to do is we are here and we have `BN_raw` and we have to first back propagate into `BN_diff` and `BN_Bar_M`. 

Now we're here and we have `DB_BN_raw` and we need to back propagate through this line. I've written out the shapes here and indeed `BN_VAR_M` is a shape 1 by 64. So, there is a broadcasting happening here that we have to be careful with but it is just an element-wise simple multiplication. 

By now we should be pretty comfortable with that. To get `DB_BN_diff` we know that this is just `BN_varm` multiplied with `DB_BN_raw`. Conversely, to get `DB_BN_VAR_inv`, we need to take `BN_diff` multiplied with `DB_BN_raw`. 

We need to make sure that broadcasting is obeyed. So in particular, `BN_VAR_M` multiplying with `DB_BN_raw` will be okay and give us 32 by 64 as we expect. But `DB_BN_VAR_inv` would be taking a 32 by 64, multiplying it by 32 by 64. So this is a 32 by 64. But of course `DB_BN_VAR_inv` is only 1 by 64. So the second line here needs a sum across the examples and because there's this dimension here we need to make sure that `keepdims` is true. 

We'll actually notice that `DB_BN_diff` is going to be incorrect. This is actually expected because we're not done with `BN_diff`. So in particular, `BN_raw` is a function of `BN_diff` but actually `BN_VAR` is a function of `BN_VAR` which is a function of `BN_div_2` which is a function of `BN_diff`. So it comes here so `BN_diff` branches out into two branches and we've only done one branch of it. 

In conclusion, the process of back propagation and batch normalization involves careful manipulation of variables and functions. It requires a deep understanding of the relationships between these variables and the mathematical operations that are performed on them.
# Back Propagation and Batch Normalization

We have to continue our back propagation and eventually come back to B and diff. Then we'll be able to do a plus equals and get the actual card gradient. For now, it is good to verify that CMP also works. It doesn't just lie to us and tell us that everything is always correct. It can, in fact, detect when your gradient is not correct. So, that's good to see as well.

Now we have the derivative here and we're trying to back propagate through this line. Because we're raising to a power of negative 0.5, I brought up the power rule. We see that basically we have that the BM bar will now be, we bring down the exponent so negative 0.5 times X, which is this, and now raised to the power of negative 0.5 minus 1 which is negative 1.5.

We would have to also apply a small chain rule here in our head because we need to take further the derivative of B and VAR with respect to this expression here inside the bracket. But because this is an elementalized operation and everything is fairly simple, that's just one and so there's nothing to do there. So this is the local derivative and then times the global derivative to create the chain rule. This is just times the BM bar have. So this is our candidate. Let me bring this down and we see that we have the correct result.

Before we propagate through the next line, I want to briefly talk about the note here where I'm using the Bessel's correction dividing by n minus 1 instead of dividing by n when I normalize here the sum of squares. You'll notice that this is a departure from the paper which uses one over n instead, not one over n minus one. Their m is our n.

It turns out that there are two ways of estimating variance of an array. One is the biased estimate which is one over n and the other one is the unbiased estimate which is one over n minus one. Confusingly in the paper, this is not very clearly described and also it's a detail that kind of matters. They are using the biased version training time but later when they are talking about the inference, they are mentioning that when they do the inference, they are using the unbiased estimate which is the n minus one version for inference and to calibrate the running mean and the running variance.

They actually introduce a train-test mismatch where in training they use the biased version and in the test time they use the unbiased version. I find this extremely confusing. You can read more about the Bessel's correction and why dividing by n minus one gives you a better estimate of the variance in a case where you have population size or samples for the population that are very small. That is indeed the case for us because we are dealing with many patches and these mini matches are a small sample of a larger population which is the entire training set.

It just turns out that if you just estimate it using one over n, that actually almost always underestimates the variance and it is a biased estimator. It is advised that you use the unbiased version and divide by n minus one. You can go through this article here that I liked that actually describes the full reasoning and I'll link it in the video description.

When you calculate the torch variance, you'll notice that they take the unbiased flag whether or not you want to divide by n or n minus one. Confusingly, they do not mention what the default is for unbiased but I believe unbiased by default is true. I'm not sure why the docs here don't cite that.
# The Bachelor: A Deep Dive into Standard Deviation and Batch Normalization

In the Bachelor, we are presented with a documentation that is somewhat confusing and incorrect. It states that the standard deviation is calculated via the biased estimator. However, this is not entirely accurate. Several people have pointed out this discrepancy in numerous issues since then. 

The rabbit hole goes deeper. If you follow the paper exactly, you'll notice that they use the biased version for training. However, when estimating the running standard deviation, they use the unbiased version. This creates a train-test mismatch. 

To cut a long story short, I'm not a fan of train-test discrepancies. I consider the fact that we use the biased version during training time and the unbiased version during test time to be a bug. I don't believe there's a good reason for this discrepancy. The paper doesn't delve into the reasoning behind this decision. 

That's why I prefer to use the Bessel's correction in my own work. Unfortunately, Bessel does not take a keyword argument that tells you whether or not you want to use the unbiased version or the biased version in both train and test. Therefore, anyone using batch normalization, in my view, has a bit of a bug in their code. 

This turns out to be much less of a problem if your batch mini-batch sizes are a bit larger. However, I find this unpardonable. Perhaps someone can explain why this is okay, but for now, I prefer to use the unbiased version consistently, both during training and at test time. That's why I'm using one over n minus one here. 

Let's now actually back-propagate through this line. The first thing I always like to do is scrutinize the shapes. In particular, looking at the shapes of what's involved, I see that `b` and `VAR` shape is 1 by 64. It's a row vector and `BND` if two dot shape is 32 by 64. 

Clearly, we're doing a sum over the zeroth axis to squash the first dimension of the shapes here using a sum. This hints to me that there will be some kind of a replication or broadcasting in the backward pass. You may notice a pattern here. Anytime you have a sum in the forward pass, it turns into a replication or broadcasting in the backward pass along the same dimension. Conversely, when we have a replication or broadcasting in the forward pass, it indicates a variable reuse. In the backward pass, that turns into a sum over the exact same dimension. Hopefully, you're noticing this duality. Those two are kind of like the opposite of each other in the forward and backward pass. 

Once we understand the shapes, the next thing I like to do is look at a toy example in my head. This helps me understand roughly how the variable dependencies go in the mathematical formula. 

Here, we have a two-dimensional array of the end of two which we are scaling by a constant and then we are summing vertically over the columns. If we have a two by two matrix `a` and then we sum over the columns and scale, we would get a row vector `B1 B2`. `B1` depends on `a` in this way, whereas it's just the sum scaled of `a` and `B2` in this way where it's the second column sum and scale. 

Looking at this, what we want to do now is we have the derivatives on `B1` and `B2` and we want to back-propagate them into `A's`. It's clear that just differentiating in your head, the local derivative here is one over n minus one times one for each one of these `A's`.
# Understanding the Derivative Flow in PyTorch

In essence, the derivative of B1 has to flow through the columns of a matrix, scaled by one over n minus one. This is a rough approximation of what's happening here. Intuitively, the derivative flow tells us that DB and diff2 will be the local derivative of this operation. There are many ways to accomplish this, but I prefer to use a method like this: 

```python
torch.dot(ones_like(bndf2))
```

This creates a large two-dimensional array of ones, which I then scale by 1.0 divided by n minus 1. This results in an array of one over n minus one, which serves as the local derivative. 

For the chain rule, I simply multiply the two arrays. Notice that one array is 32 by 64 and the other is just 1 by 64. I'm letting the broadcasting do the replication because internally in PyTorch, `dbnbar` (which is a 1 by 64 row vector) will get copied vertically until the two arrays are of the same shape. Then, there will be an element-wise multiplication. The broadcasting is essentially doing the replication, and I will end up with the derivatives of DB and diff2. 

This is the candidate solution. Let's test it:

```python
# Uncomment this line to check the solution
# print(dbnbar)
```

Indeed, we see that this is the correct formula. 

Next, let's differentiate `bndf` in this equation. Here, we have that `bndf` is element-wise squared to create `bndf2`. This is a relatively simple derivative because it's a simple element-wise operation, similar to the scalar case. The derivative of `bndf` should be 2x, where x is `bndf`. That's the local derivative, and then we multiply it by the chain rule. The shapes of these arrays are the same, so we multiply them together. 

This is the backward pass for this variable. We have to be careful because we already calculated `dbm_depth`. This is just the end of the other branch coming back to `bndf`, because `bndf` was already back-propagated to way over here from `bndf_raw`. We now completed the second branch, so we have to do `+=`. 

We had an incorrect derivative for `bndf` before, and I'm hoping that once we append this last missing piece, we have the exact correctness. Let's run `bndf` to check. `bndf` now actually shows the exact correct derivative, which is comforting. 

Let's now back-propagate through this line here. The first thing we do, of course, is check the shapes. The shape of `bndf` is 32 by 64. `hpbn` is the same shape, but `bndf_mean_i` is a row vector 1 by 64. So this minus here will actually do broadcasting, and we have to be careful with that. Because of the duality, broadcasting in the forward pass means a variable reuse and therefore there will be a sum in the backward pass. 

Let's write out the backward pass here now. To back-propagate into the `hpbn`, because these are the same shape, the local derivative for each one of the elements here is just one for the corresponding element in here. This means that the gradient just simply copies, it's just a variable assignment. So, I'm just going to clone this tensor just for safety to create an exact copy of `dbndf`. 

Then, to back-propagate into `bndf_mean_i`, we have to consider that this is a row vector and this is a matrix. So, the broadcasting that happened here in the forward pass means that there's a sum happening in the backward pass. The local derivative of each one of these elements with respect to the corresponding element in `bndf_mean_i` is just one. So, we have to sum up all the elements in `dbndf` along dimension zero to get `dbndf_mean_i`. 

This is the backward pass for this line. Let's bring it down here and check it:

```python
# Uncomment this line to check the solution
# print(dbndf_mean_i)
```

Indeed, we see that this is the correct formula. 

In conclusion, understanding the derivative flow in PyTorch is crucial for implementing and debugging your own custom autograd Functions. By methodically checking the shapes of your tensors and carefully considering the effects of broadcasting, you can ensure that your backward passes are correctly computing the gradients.
# Backpropagation in PyTorch

In this tutorial, we will be discussing the process of backpropagation in PyTorch. We will start by examining the local derivative, which is essentially negative `torch.1` of the shape of `B` and `diff`. This is then multiplied by the backpropagation for the replicated `B` and `mean I`. 

```python
# Candidate backward pass
# Commenting out the previous lines
# Enter
```

However, this is supposed to be incorrect. The reason being, we are backpropagating from `a b` and `diff` into `hpbn`, but we're not done yet. `B` and `mean I` depends on `hpbn` and there will be a second portion of that derivative coming from this second branch. 

So, let's now backpropagate from `B` and `mean I` into `hpbn`. Here, we have to be careful because there's a sum along the zeroth dimension, which will turn into broadcasting in the backward pass. 

```python
# The hpbn gradient will be scaled by 1 over n
# The gradient here on dbn mean I is going to be scaled by 1 over n
# It's going to flow across all the columns and deposit itself into the hpvn
```

We want this gradient scaled by `1/n`. To achieve this, we scale down the gradient and then replicate it across all the rows. We can do this by using `torch.lunslike` of `hpbn` and let the broadcasting do the work of replication. 

```python
# Broadcasting and scaling
```

This completes the backpropagation of the batchnorm layer. Now, let's backpropagate through the linear layer one. 

```python
# Inspect the shapes
# dmcat should be a matrix multiplication of dhbn with W1 and one transpose thrown in there
# To make MCAT be 32 by 30, I need to take dhpn (32 by 64) and multiply it by w1 transpose
```

Backpropagating through linear layers is fairly easy, just by matching the shapes. 

In conclusion, backpropagation is a fundamental concept in neural networks and understanding it is crucial for implementing and debugging neural networks. PyTorch provides a powerful and flexible platform for implementing these concepts.
# Backpropagation in Neural Networks

To get the only one I need to end up with 30 by 64. So, to get that, I need to take the MCAT transpose and multiply that by it. Finally, to get DB1, this is an addition, and we saw that basically, I need to just sum the elements in DHPBN along some dimension. 

To make the dimensions work out, I need to sum along the zeroth axis here to eliminate this dimension, and we do not keep dims. So, we want to just get a single one-dimensional lecture of 64. These are the claimed derivatives. Let me put that here and let me uncomment three lines and cross our fingers. Everything is great. 

Okay, so we now continue. Almost there, we have the derivative of MCAT, and we want to backpropagate into M. So, I again copied this line over here. This is the forward pass, and then this is the shapes. So, remember that the shape here was 32 by 30 and the original shape of M was 32 by 3 by 10. 

This layer in the forward pass, as you recall, did the concatenation of these three 10-dimensional character vectors. So now, we just want to undo that. This is actually a relatively straightforward operation because the backward pass of the view is just a representation of the array. It's just a logical form of how you interpret the array. 

So, let's just reinterpret it to be what it was before. So, in other words, the end is not 32 by 30. It is basically DMCAT, but if you view it as the original shape, just M dot shape, you can pass in tuples into view. We just re-represent that view and then we uncomment this line here and hopefully, the derivative of M is correct. 

In this case, we just have to re-represent the shape of those derivatives into the original view. So now, we are at the final line and the only thing that's left to backpropagate through is this indexing operation here, MSC at xB. 

So, as I did before, I copy-pasted this line here and let's look at the shapes of everything that's involved and remind ourselves how this worked. M.shape was 32 by 3 by 10. It says 32 examples and then we have three characters each one of them has a 10-dimensional embedding. 

This was achieved by taking the lookup table C which has 27 possible characters, each of them 10 dimensional, and we looked up at the rows that were specified inside this tensor xB. So, xB is 32 by 3 and it's basically giving us for each example the identity or the index of which character is part of that example. 

Here, I'm showing the first five rows of three of this tensor xB. We can see that for example, the first example in this batch is that the first character and the fourth character comes into the neural net. Then, we want to predict the next character in a sequence after the characters 1, 1, 4. 

Basically, what's happening here is there are integers inside xB and each one of these integers is specifying which row of C we want to pluck out. Then, we arrange those rows that we've plucked out into a 32 by 3 by 10 tensor and we just package them into the sensor. 

Now, what's happening is that we have D amp. So, for every one of these plucked out rows, we have their gradients now. But they're arranged inside this 32 by 3 by 10 tensor. So, all we have to do now is...
# Back Propagation Through Indexing

We just need to route this gradient backwards through this assignment. We need to find which row of C that every one of these 10-dimensional embeddings come from, and then we need to deposit them into DC. We just need to undo the indexing. Of course, if any of these rows of C was used multiple times, which almost certainly is the case like the row one and one was used multiple times, then we have to remember that the gradients that arrive there have to add. For each occurrence, we have to have an addition.

Let's now write this out. I don't actually know if there's a much better way to do this than a for loop, unfortunately, in Python. Maybe someone can come up with a vectorized efficient operation, but for now, let's just use for loops. 

Let me create a `torch.zeros` like C to initialize a 27 by 10 tensor of all zeros. Then, for `k` in range `xb.shape[0]`, and for `j` in range `xb.shape[1]`, this is going to iterate over all the elements of `xb`, all these integers. 

Let's get the index at this position. The index is basically `xb[k][j]`, so an example of that is 11 or 14 and so on. In the forward pass, we took the row of C at index and we deposited it into `m[k][j]`. That's what happened, that's where they are packaged. 

Now we need to go backwards and we just need to route `dm[k][j]`. We now have these derivatives for each position and it's 10-dimensional. We just need to go into the correct row of C, so `dc[ix]` is this but `+=` because there could be multiple occurrences. Like the same row could have been used many many times and so all of those derivatives will just go backwards through the indexing and they will add. 

Let's uncomment this and cross our fingers. Hey, so that's it, we've back propagated through this entire beast. So there we go, it totally makes sense.

## Exercise Two

Now we come to exercise two. It basically turns out that in this first exercise we were doing way too much work. We were back propagating way too much and it was all good practice and so on, but it's not what you would do in practice. 

The reason for that is, for example, here I separated out this loss calculation over multiple lines and I broke it up all to its smallest atomic pieces and we back propagated through all of those individually. But it turns out that if you just look at the mathematical expression for the loss, then actually you can do the differentiation on pen and paper and a lot of terms cancel and simplify. 

The mathematical expression you end up with can be significantly shorter and easier to implement than back propagating through all the little pieces of everything you've done. Before we had this complicated forward paths going from logits to the loss, but in PyTorch everything can just be glued together into a single call at that cross entropy. You just pass in logits and the labels and you get the exact same loss as I verify here. 

So our previous loss and the fast loss coming from the chunk of operations as a single mathematical expression is the same, but it's much much faster in a forward pass.
# Faster Backward Pass in Neural Networks

The backward pass in neural networks can be significantly faster. The reason for this is that if you look at the mathematical form of the backward pass and differentiate it, you will end up with a very small and short expression. This is what we aim to achieve. We want to, in a single operation or very quickly, go directly to the derivatives of the logits. 

We need to implement the logits as a function of logits and yb's. This will be significantly shorter than the previous method where we had to go all the way to the derivatives of the logits. All of this work can be skipped in a much simpler mathematical expression that you can implement. 

You can give it a shot yourself. Look at the mathematical expression of the loss and differentiate it with respect to the logits. 

## A Hint to Get Started

Here's a hint to get you started. We have logits, then there's a softmax function that takes the logits and gives you probabilities. We are using the identity of the correct next character to pluck out a row of probabilities, take the negative log of it to get our negative log probability, and then we average all the log probabilities to get our loss. 

For a single individual example, we have that loss is equal to negative log probability, where P here is thought of as a vector of all the probabilities at the Y position where Y is the label. P here is the softmax function, so the ith component of P of this probability vector is just the softmax function, raising all the logits to the power of E and normalizing so everything sums to 1. 

If you write out P of Y, you can just write out the softmax and then we're interested in the derivative of the loss with respect to the ith logit. It's a derivative of this expression where we have L indexed with the specific label Y and on the bottom, we have a sum over J of e to the L J and the negative log of all that. 

Try to derive the expression for the loss by the derivative of the ith logit and then we're going to implement it. 

## The Result

Here's the result. I applied the rules of calculus from the first or second year of a bachelor's degree. The expression simplifies quite a bit. You have to separate out the analysis in the case where the ith index that you're interested in inside logits is either equal to the label or it's not equal to the label. Then the expression simplifies and cancels in a slightly different way. 

What we end up with is something very simple. We either end up with P at index i, where P is this vector of probabilities after a softmax, or P at index i minus 1, where we just simply subtract one. In any case, we just need to calculate the softmax P and then in the correct dimension, we need to subtract one. That's the gradient in the form that it takes analytically. 

Let's implement this. We have to keep in mind that this is only done for a single example but here we are working with batches of examples. We have to be careful of that. The loss for a batch is the average loss over all the examples. In other words, it's the sum of the loss for each individual example and then divided by the number of examples.
# Understanding Logits and Softmax in PyTorch

In this article, we will delve into the concept of logits and softmax in PyTorch. We will also discuss how to backpropagate through these functions and the importance of being careful with it.

## Understanding Logits

Logits are going to be of the softmax function. PyTorch has a softmax function that you can call. We want to apply the softmax on the logits and we want to go in the dimension that is one. Essentially, we want to do the softmax along the rows of these logits.

At the correct positions, we need to subtract 1. So, for logits at iterating over all the rows and indexing into the columns provided by the correct labels inside YB, we need to subtract one.

## Understanding the Average Loss

The average loss is the loss and in the average, there's a division by `n` of all the losses added up. We need to also propagate through that division. The gradient has to be scaled down by `n` as well because of the mean. This should be the result.

Upon verification, we may not get an exact match. However, the maximum difference from logits from PyTorch and our logits is on the order of 5e negative 9. It's a tiny number. Because of floating point precision, we don't get the exact bitwise result, but we basically get the correct answer approximately.

## Visualizing Logits

Let's pause here briefly before we move on to the next exercise. I'd like us to get an intuitive sense of what the logits are because it has a beautiful and very simple explanation.

Here, I'm taking the logits and visualizing them. We have a batch of 32 examples of 27 characters. The logits are the probabilities that the properties Matrix in the forward pass. The black squares are the positions of the correct indices where we subtracted one.

These are the derivatives on the logits. Let's look at just the first row. I'm plotting the probabilities of these logits and then taking just the first row. This is the probability row and then the logits of the first row, and multiplying by `n` just for us so that we don't have the scaling by `n` in here and everything is more interpretable.

We see that it's exactly equal to the probability of course, but then the position of the correct index has a minus equals one so minus one on that position. If you take the logits at zero and sum it, it actually sums to zero.

## Understanding the Gradients

You should think of these gradients at each cell as a force. We are going to be pulling down on the probabilities of the incorrect characters and we're going to be pulling up on the probability at the correct index. That's what's happening in each row.

The amount of push and pull is exactly equalized because the sum is zero. The amount to which we pull down in the probabilities and the amount that we push up on the probability of the correct character is equal.

Think of the neural network now as a massive pulley system. We're up here on top of the logits and we're pulling down the probabilities of incorrect characters and pulling up the property of the correct ones. In this complicated pulley system, because everything is mathematically determined, think of it as this tension translating to this complicated pulling mechanism and then eventually we get a tug on the weights and the biases.
# Understanding Neural Networks: Training and Backward Pass

In training a neural network, we can visualize the process as a tug of war. We tug in the direction we prefer for each of these elements, and the parameters slowly give in to the tug. This is what training a neural network looks like on a high level.

The forces of push and pull in these gradients are very intuitive. We're pushing and pulling on the correct answer and the incorrect answers. The amount of force that we're applying is proportional to the probabilities that came out in the forward pass.

For example, if our probabilities came out exactly correct, so they would have had zero everywhere except for one at the correct position, then the logits would be all a row of zeros for that example. There would be no push and pull. So, the amount to which your prediction is incorrect is exactly the amount by which you're going to get a pull or a push in that dimension.

If you have, for example, a very confidently mispredicted element, that element is going to be pulled down very heavily, and the correct answer is going to be pulled up to the same amount. The other characters are not going to be influenced too much.

The amounts to which you mispredict is then proportional to the strength of the pull. This is happening independently in all the dimensions of this tensor. It's very intuitive and easy to think through. This is basically the magic of the cross-entropy loss and what it's doing dynamically in the backward pass of the neural net.

Now, we get to exercise number three, which is a fun exercise, depending on your definition of fun. We are going to do for batch normalization exactly what we did for cross-entropy loss in exercise number two. That is, we are going to consider it as a glued single mathematical expression and backpropagate through it in a very efficient manner. We are going to derive a much simpler formula for the backward path of batch normalization.

Previously, we've broken up batch normalization into all of the little intermediate pieces and all the atomic operations inside it, and then we backpropagated through it one by one. Now, we just have a single forward pass of a batch form, and it's all glued together. We see that we get the exact same result as before.

For the backward pass, we'd like to also implement a single formula for backpropagating through this entire operation, that is the batch normalization. In the forward pass, previously, we took the hidden states of the pre-batch normalization and created the hidden states just before the activation. In the batch normalization paper, the hidden states of the pre-batch normalization is denoted as X and the hidden states just before the activation is denoted as Y.

In the backward pass, what we'd like to do now is, we have the derivative of the hidden states just before the activation, and we'd like to produce the derivative of the hidden states of the pre-batch normalization. We'd like to do that in a very efficient manner. That's the name of the game: calculate the derivative of the hidden states of the pre-batch normalization given the derivative of the hidden states just before the activation. For the purposes of this exercise, we're going to ignore gamma and beta and their derivatives because they take on a very simple form in a very similar way to what we did above.

To help you a little bit, I started off the implementation here on pen and paper and took two sheets of paper to derive the mathematical formulas for the backward pass. Basically, to set up the problem.
# Understanding Backpropagation in Batch Normalization

Firstly, let's write out the MU, Sigma Square, variance x i hat, and Y I, exactly as in the paper, except for the bezel correction. Then, in a backward pass, we have the derivative of the loss with respect to all the elements of Y. Remember that Y is a vector, so there are multiple numbers here. So, we have all the derivatives with respect to all the Y's. 

There's also a demo and a beta, which is kind of like the compute graph. The gamma and the beta, there's the X hat, and then the MU and the sigma squared, and the X. So we have DL by DYI and we want DL by d x i for all the I's in these vectors. 

This is the compute graph and you have to be careful because these are vectors, so there are many nodes here inside x, x hat, and Y. But mu and sigma, sorry Sigma Square, are just individual scalars, single numbers. So you have to be careful with that. You have to imagine there are multiple nodes here or you're going to get your math wrong. 

As an example, I would suggest that you go in the following order: one, two, three, four, in terms of the back propagation. So back propagating to X hat, then into Sigma Square, then into mu, and then into X. Just like in a topological sort in micrograd, we would go from right to left. You're doing the exact same thing, except you're doing it with symbols and on a piece of paper. 

For number one, I'm not giving away too much. If you want DL of d x i hat, then we just take DL by DYI and multiply it by gamma because of this expression here where any individual Yi is just gamma times x i hat plus beta. This doesn't help you too much there, but this gives you basically the derivatives for all the X hats. 

Now, try to go through this computational graph and derive what is DL by D Sigma Square, and then what is DL by B mu, and then what is D L by DX. 

To get DL by D Sigma Square, we have to remember that there are many X hats here, and remember that Sigma square is just a single individual number. So when we look at the expression for the L by D Sigma Square, we have to consider all the possible paths. Sigma square has a large fan out, there are lots of arrows coming out from Sigma square into all the X hats. And then there's a back propagating signal from each X hat into Sigma square. That's why we actually need to sum over all those I's from I equal to 1 to m, of the DL by d x i hat (which is the global gradient) times the x i Hat by D Sigma Square (which is the local gradient) of this operation here. 

Mathematically, you get a certain expression for DL by D Sigma square and we're going to be using this expression when we back propagate into mu and then eventually into X. 

Now, let's continue our back propagation into mu. What is D L by D mu? Again, be careful that mu influences X hat and X hat is actually lots of values. For example, if our mini batch size is 32 as it is in our example, then this is 32 numbers and 32 arrows going back to mu. And then mu going to Sigma square is just a single Arrow because Sigma square is a scalar. So in total, there are 33 arrows emanating from mu and all of them have gradients coming into mu.
# Understanding the Summation of Gradients

When we look at the expression for DL by D mu, we are summing up over all the gradients of DL by d x i hat times the x i Hat by D mu. This is represented by the arrow in the diagram, and there are 32 arrows here. Additionally, there is one arrow from the L by the sigma Square Times the sigma squared by D mu.

To work out this expression, we need to simplify it. The first term gives us an expression. However, something interesting happens with the second term. When we look at the sigma squared by D mu and simplify it, we find that if we assume that in a special case where mu is actually the average of X I's, as it is in this case, then the gradient vanishes and becomes exactly zero. This makes the entire second term cancel out.

If you just have a mathematical expression like this and you look at D Sigma Square by D mu, you would get some mathematical formula for how mu impacts Sigma Square. But if it is the special case that Nu is actually equal to the average, as it is in the case of normalization, that gradient will actually vanish and become zero. So the whole term cancels and we just get a fairly straightforward expression here for DL by D mu.

Now, we get to the most complex part which is deriving DL by dxi, which is ultimately what we're after. Inside X, there are 32 numbers, there are 32 Little X I's. Each x i has an arrow going to Mu, an arrow going to Sigma Square, and an arrow going to X hat. Each x i hat is just a function of x i and all the other scalars, so x i hat only depends on x i and none of the other X's. Therefore, there are actually 32 arrows in this single Arrow, but those 32 arrows are going exactly parallel, they don't interfere and they're just going parallel between x and x hat.

In back propagation, we now need to apply the chain rule and we need to add up those three contributions. We have DL by D mu, DL by D Sigma Square, and DL by d x i hat. But we need three other terms. To derive them, we just look at these Expressions here and differentiate with respect to x i. 

Finally, we plug everything together. All of these terms are multiplied with all of these terms and added up according to this formula. This part can get a little bit complex, but it's all part of understanding the summation of gradients.
# Implementing the Backward Pass for Batch Normalization

What ends up happening is that you get a large expression. The thing to be very careful with here, of course, is that we are working with a DL by dxi for a specific 'I' here. But when we are plugging in some of these terms, like say, this term here 'DL by D signal squared', you see how the 'L by D Sigma squared I' ends up with an expression. 

I'm iterating over little 'I's here, but I can't use 'I' as the variable when I plug in here because this is a different 'I' from this 'I'. This 'I' here is just a placeholder, like a local variable for a for-loop. So here, when I plug that in, you notice that I rename the 'I' to a 'J' because I need to make sure that this 'J' is not this 'I'. This 'J' is like a little local iterator over 32 terms, and so you have to be careful with that when you're plugging in the expressions from here to here. You may have to rename 'I's into 'J's, and you have to be very careful about what is actually an 'I' with respect to the 'L by t x i'. So some of these are 'J's, some of these are 'I's.

Then we simplify this expression. The big thing to notice here is that a bunch of terms just kind of come out to the front, and you can refactor them. There's a 'sigma squared plus Epsilon' raised to the power of negative three over two. This 'Sigma squared plus Epsilon' can be actually separated out into three terms, each of them are 'Sigma squared plus Epsilon' to the negative one over two. So the three of them multiplied is equal to this, and then those three terms can go different places because of the multiplication. 

One of them actually comes out to the front and will end up here outside. One of them joins up with this term, and one of them joins up with this other term. Then when you simplify the expression, you'll notice that some of these terms that are coming out are just the 'x i hats'. So you can simplify just by rewriting that.

What we end up with at the end is a fairly simple mathematical expression over here that I cannot simplify further. But basically, you'll notice that it only uses the stuff we have, and it derives the thing we need. So we have the 'L by d y' for all the 'I's, and those are used plenty of times here. In addition, what we're using is these 'x i hats' and 'XJ hats', and they just come from the forward pass. 

Otherwise, this is a simple expression, and it gives us 'DL by d x i' for all the 'I's, and that's ultimately what we're interested in. So that's the end of the BatchNorm backward pass analytically. Let's now implement this final result.

I implemented the expression into a single line of code here, and you can see that the max diff is tiny, so this is the correct implementation of this formula. Now, I'll just tell you that getting this formula here from this mathematical expression was not trivial. There's a lot going on packed into this one formula. 

This is a whole exercise by itself because you have to consider the fact that this formula here is just for a single neuron and a batch of 32 examples. But what I'm doing here is that we actually have 64 neurons. So this expression has to, in parallel, evaluate the BatchNorm backward pass for all of those 64 neurons independently. This has to happen basically in every single column of the inputs here. 

In addition to that, you see how there are a bunch of sums here, and we need to make sure that when I do those sums, they broadcast correctly onto everything else that's here.
# Understanding the Mathematical Formula in Neural Networks

Getting this expression is highly non-trivial, and I invite you to look through it and step through it. It's a whole exercise to make sure that this checks out. But once all the shapes are green and once you convince yourself that it's correct, you can also verify that PyTorch gets the exact same answer as well. This gives you a lot of peace of mind that this mathematical formula is correctly implemented here, broadcasted correctly, and replicated in parallel for all of the 64 neurons inside this batch normalization layer.

## Exercise Number Four

Finally, exercise number four asks you to put it all together. Here we have a redefinition of the entire problem. We reinitialize the neural network from scratch and everything. Instead of calling `loss.backward()`, we want to have the manual backpropagation here as we derived it above. Go up, copy-paste all the chunks of code that we've already derived, put them here, and derive your own gradients. Then optimize this neural network using your own gradients all the way to the calibration of the batch normalization and the evaluation of the loss.

I was able to achieve quite a good loss, basically the same loss you would achieve before. That shouldn't be surprising because all we've done is we've really gotten to `loss.backward()` and we've pulled out all the code and inserted it here. But those gradients are identical and everything is identical and the results are identical. It's just that we have full visibility on exactly what goes on under the hood of `loss.backward()` in this specific case.

## The Full Backward Pass

This is all of our code. This is the full backward pass using the simplified backward pass for the cross-entropy loss and the max generalization. So backpropagating through cross-entropy, the second layer, the `tanh` nonlinearity, the batch normalization, through the first layer, and through the embedding. This is only maybe 20 lines of code or something like that and that's what gives us gradients.

Now we can potentially erase `loss.backward()`. The way I have the code set up is you should be able to run this entire cell once you fill this in and this will run for only 100 iterations and then break. It breaks because it gives you an opportunity to check your gradients against PyTorch. Our gradients are not exactly equal, they are approximately equal and the differences are tiny, around 1e-9. I don't exactly know where they're coming from, to be honest.

## Confidence in Gradients

Once we have some confidence that the gradients are basically correct, we can take out the gradient tracking. We can disable this breaking statement and then we can disable `loss.backward()`. We don't need it anymore, it feels amazing to say that. When we are doing the update, we're not going to use `p.grad`, this is the old way of PyTorch. We don't have that anymore because we're not doing backward. We are going to use this update where we see that I'm iterating over the gradients to be in the same order as the parameters and I'm zipping them up the gradients and the parameters into `p` and `grad`. Then here I'm going to step with just the `grad` that we derived manually.

The last piece is that none of this now requires gradients from PyTorch. One thing you can do here is you can do `with torch.no_grad():` and offset this whole code block. Really what you're saying is you're telling PyTorch that "Hey, I'm not using your gradients anymore, I'm using my own."
# Implementing Backward Pass in Neural Networks

In this tutorial, we will not call backward on any of this, which allows PyTorch to be more efficient. Let's run this and see how it performs. As you can see, the loss's backward is commented out, and we're optimizing. We'll let this run and hopefully, we'll get a good result.

After allowing the neural net to finish optimization, I calibrated the batch normalization parameters because I did not keep track of the running mean and variance in the training loop. Running the loss, we obtained a pretty good loss, very similar to what we've achieved before.

Next, I sampled from the model and we see some of the gibberish that we're sort of used to. Basically, the model worked and samples pretty decent results compared to what we were used to. The big deal is that we did not use loss.backward, we did not use PyTorch's autograd, and we estimated our gradients ourselves by hand.

Looking at the backward pass of this neural net, you might think to yourself, "Actually, that's not too complicated." Each one of these layers is like three lines of code or something like that and most of it is fairly straightforward, potentially with the notable exception of the batch normalization backward pass. Otherwise, it's pretty good.

That's everything I wanted to cover for this lecture. What I liked about it is that it gave us a very nice diversity of layers to backpropagate through. It gives a pretty nice and comprehensive sense of how these backward passes are implemented and how they work. You'd be able to derive them yourself but of course, in practice, you probably don't want to and you want to use PyTorch's autograd. 

Hopefully, you have some intuition about how gradients flow backwards through the neural net, starting at the loss and how they flow through all the variables and all the intermediate results. If you understood a good chunk of it and if you have a sense of that, then you can count yourself as one of these buff individuals on the left instead of those on the right.

In the next lecture, we're actually going to go to recurrent neural networks, LSTMs, and all the other variants of RNNs. We're going to start to complexify the architecture and start to achieve better log likelihoods. I'm really looking forward to it.