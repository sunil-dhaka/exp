---
layout: default
title: 3 - Activations Gradients BatchNorm
parent: Andrej Karpathy
has_children: false
nav_order: 3
---

# Implementing MakeMore: Understanding Activations and Gradients in Neural Networks

Hello everyone! Today, we are continuing our implementation of MakeMore. In the last lecture, we implemented the multi-layer perceptron (MLP) for character-level language modeling, following the lines of Benjiotyle's 2003 paper. We took a few characters from the past and used an MLP to predict the next character in a sequence.

Now, we'd like to move on to more complex and larger neural networks like Recurrent Neural Networks (RNNs) and their variations like the Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), and so on. 

However, before we do that, we have to stick around the level of MLP for a bit longer. I'd like us to have a very good intuitive understanding of the activations in the neural net during training, especially the gradients that are flowing backwards and how they behave. 

Understanding the activations and the gradients is crucial to comprehend the history of the development of these architectures. While RNNs are very expressive and can, in principle, implement all the algorithms, they are not easily optimizable with the first-order ingredient-based techniques that we use all the time. 

A lot of the variants of RNNs have tried to improve this situation. So, let's get started.

The starting code for this lecture is largely the code from lecture four, but I've cleaned it up a bit. We are importing all the torch and matplotlib utilities, reading in the words just like before. We have a total of 32,000 example words and a vocabulary of all the lowercase letters and the special dot token.

We are reading in the dataset, processing it, and creating three splits: the train, dev, and the test split. 

In the MLP, I removed a bunch of magic numbers and instead, we have the dimensionality of the embedding space of the characters and the number of hidden units in the hidden layer. I've pulled them outside here so that we don't have to go and change all these magic numbers all the time.

We have the same neural net with 11,000 parameters that we optimize now over 200,000 steps with a batch size of 32. I refactored the code here a bit, but there are no functional changes. I just created a few extra variables, added a few more comments, and removed all the magic numbers.

When we optimize, we saw that our loss looked something like this: the train and val loss were about 2.16 and so on. 

I refactored the code a bit for the evaluation of arbitrary splits. You pass in a string of which split you'd like to evaluate. Depending on train, val, or test, I index in and get the correct split. This is the forward pass of the network and evaluation of the loss and printing it.

One thing that you'll notice here is I'm using a decorator `torch.nograd` which you can also look up and read the documentation of. Basically, what this decorator does on top of a function is that whatever happens in this function is assumed by torch to never require any gradients, so it will not do any of the bookkeeping that it does to keep track.
# Anticipating Gradients for an Efficient Backward Pass

In the world of PyTorch, it's almost as if all the tensors that get created have a `requires_grad` of `False`. This makes everything much more efficient because you're telling PyTorch that you will not call `backward` on any of this computation. Therefore, it doesn't need to maintain the graph under the hood. This is what `torch.no_grad()` does. You can also use a context manager with `torch.no_grad()`. 

## Sampling from a Model

Just as before, we have the sampling from a model. This involves passing a neural net, getting the distribution, sampling from it, adjusting the context window, and repeating until we get the special end token. We see that we are starting to get much nicer looking words sampled from the model. However, they're still not amazing and they're still not fully named. But it's much better than when we had them with the byground model. 

## Scrutinizing the Initialization

The first thing I would like to scrutinize is the initialization. I can tell that our network is very improperly configured at initialization. There are multiple things wrong with it, but let's just start with the first one. 

On the zeroth iteration, the very first iteration, we are recording a loss of 27. This rapidly comes down to roughly one or two. I can tell that the initialization is all messed up because this is way too high. 

In the training of neural nets, it is almost always the case that you will have a rough idea for what loss to expect at initialization. That just depends on the loss function and the problem set up. In this case, I do not expect 27. I expect a much lower number and we can calculate it together. 

At initialization, there are 27 characters that could come next for any one training example. We have no reason to believe any character is much more likely than others. So, we'd expect that the probability distribution that comes out initially is a uniform distribution, assigning about equal probability to all the 27 characters. 

The probability for any character would be roughly 1 over 27. That is the probability we should record. Then the loss is the negative log probability. So, let's wrap this in a tensor and then take the log of it. The negative log probability is the loss we would expect, which is 3.29, much lower than 27. 

What's happening right now is that at initialization, the neural net is creating probability distributions that are all messed up. Some characters are very confident and some characters are very not confident. The network is very confidently wrong, and that's what makes it record very high loss. 

Here's a smaller four-dimensional example of the issue. Let's say we only have four characters and then we have logits that come out of the neural net and they are very close to zero. When we take the softmax of all zeros, we get probabilities that are a diffuse distribution. It sums to one and is exactly uniform. 

In this case, if the label is, say, 2, it doesn't actually matter if the label is 2 or 3 or 1 or 0 because it's a uniform distribution. We're recording the exact same loss in this case, 1.38. This is the loss we would expect for a four-dimensional example. As we start to manipulate these logits, we're going to be changing the loss.
# Understanding Neural Network Initialization

In the process of initializing a neural network, we might encounter a situation where the logits, or the vector of raw predictions that a classification model generates, could be a very high number, like five for instance. In such a case, we'll record a very low loss because we're assigning the correct probability at the initialization by chance to the correct label. 

However, it's much more likely that some other dimension will have a high logit. What will happen then is we start to record much higher loss. The logits might take on extreme values and we record really high loss. 

For example, if we have `torch.randen` of four, these are normally distributed numbers. We can also print the logits, the probabilities that come out of it, and the loss. Because these logits are near zero for the most part, the loss that comes out is okay. However, because these are more extreme values, it's very unlikely that you're going to be guessing the correct bucket and then you're confidently wrong and recording very high loss. 

If your logits are coming out even more extreme, you might get extremely insane losses, like infinity even at initialization. This is not good and we want the logits to be roughly zero when the network is initialized. In fact, the logits don't have to be just zero, they just have to be equal. For example, if all the logits are one, then because of the normalization inside the softmax, this will actually come out okay. 

By symmetry, we don't want it to be any arbitrary positive or negative number, we just want it to be all zeros and record the loss that we expect at initialization. 

Let's now quickly see where things go wrong in our example. Here we have the initialization. Let me reinitialize the neural net and here let me break after the very first iteration so we only see the initial loss which is 27. That's way too high. 

Now we can inspect the variables involved and we see that the logits here, if we just print some of these, if we just print the first row, we see that the logits take on quite extreme values. That's what's creating the fake confidence and incorrect answers and makes the loss get very high. These logits should be much closer to zero. 

Let's think through how we can achieve logits coming out of this neural net to be closer to zero. You see here that logits are calculated as the hidden states multiplied by W2 plus B2. 

First of all, currently we're initializing B2 as random values of the right size, but because we want roughly zero, we don't actually want to be adding a bias of random numbers. In fact, I'm going to add a times zero here to make sure that B2 is just basically zero at the initialization. 

Second, this is H multiplied by W2. If we want logits to be very small, then we would be multiplying W2 and making that smaller. For example, if we scale down W2 by 0.1 all the elements, then if I do again just the very first iteration, you see that we are getting much closer to what we expect. 

You're probably wondering, can we just set this to zero? Then we get of course exactly what we're looking for at initialization. The reason I don't usually do this is because I'm very nervous and I'll show you in a second why you don't want to.
# Understanding Neural Network Initialization

In this article, we will discuss the importance of setting the weights of a neural network. It's crucial to note that these weights, also known as 'W's, should not be set exactly to zero. Instead, they should be small numbers. For the output layer in this specific case, setting the weights to zero might be fine, but things can go wrong very quickly if you do that. So, let's just go with 0.01.

With this weight, our loss is close enough but has some entropy. It's not exactly zero; it's got some low entropy. This low entropy is used for symmetry breaking, as we'll see later. The logits are now coming out much closer to zero, and everything is well and good.

Let's now run the optimization with this new initialization and see what losses we record. After letting it run, we see that we started off good and then we came down a bit. The plot of the loss now doesn't have this hockey stick appearance. 

During the first few iterations of the loss, what's happening is that the optimization is just squashing down the logits and then rearranging them. We took away this easy part of the loss function where the weights were just being shrunk down. Therefore, we don't get these easy gains in the beginning, and we're just getting some of the hard gains of training the actual neural network. Hence, there's no hockey stick appearance.

Good things are happening in that both number one, the loss at initialization is what we expect, and the loss doesn't look like a hockey stick. This is true for any neuron that you might train and something to look out for. Secondly, the loss that came out is actually quite a bit improved. We get a slightly improved result because we're spending more cycles, more time optimizing the neural network, instead of just spending the first several thousand iterations probably just squashing down the weights because they are way too high in the beginning in the initialization.

Even though everything is looking good on the level of the loss and we get something that we expect, there's still a deeper problem lurking inside this neural network and its initialization. The logits are now okay, but the problem now is with the values of H, the activations of the hidden states.

If we visualize this tensor H, it's kind of hard to see, but the problem here, roughly speaking, is that many of the elements are one or negative one. Recall that the `torch.nh` function is a squashing function. It takes arbitrary numbers and squashes them into a range of negative one and one, and it does so smoothly.

To get a better idea of the distribution of the values inside this tensor, let's look at the histogram of H. We can see that H is 32 examples and 200 activations in each example. We can view it as negative one to stretch it out into one large vector. We can then call `tolist` to convert this into one large Python list of floats. Then we can pass this into `plt.hist` for histogram and we say we want 50 bins.
# Understanding Neural Networks: A Deep Dive into Activation Functions

In this article, we will be discussing the role of activation functions in neural networks, specifically focusing on the hyperbolic tangent (tanh) function. We will also explore the concept of backpropagation and how it impacts the performance of a neural network.

Let's start by examining a histogram of a neural network's activations. We notice that most of the values predominantly take on the value of negative one and one. This indicates that the tanh function is very active. 

To understand why this is the case, we can look at the pre-activations. The distribution of the pre-activations is very broad, taking numbers between negative 15 and 15. This is why in the tanh function, everything is being squashed and capped to be in the range of negative one and one. A lot of numbers here take on very extreme values.

If you are new to neural networks, you might not see this as an issue. However, if you're well-versed in the dark arts of backpropagation, you might start to sweat when you see the distribution of tanh activations. 

During backpropagation, we start at the loss and flow through the network backwards. In particular, we're going to backpropagate through the tanh function. This layer is made up of 200 neurons for each one of these examples and it implements an element-wise tanh function. 

Let's look at what happens in the tanh function in the backward pass. We can refer back to our previous code from the very first lecture. We saw that the input here was `x` and then we calculate `t` which is the tanh of `x`. `t` is between negative 1 and 1. It's the output of the tanh function. 

In the backward pass, we backpropagate through a tanh function by taking the gradient and then multiplying it with the local gradient which took the form of `1 - t^2`. 

So, what happens if the outputs of your tanh function are very close to negative one or one? If you plug in `t = 1` here, you're going to get a zero multiplying the gradient. No matter what the gradient is, we are killing the gradient and we're effectively stopping the backpropagation through this tanh unit. Similarly, when `t = -1`, this will again become zero and the gradient just stops. 

Intuitively, this makes sense because if the output of a tanh neuron is very close to one, then we are in the tail of this tanh function. Changing the input is not going to impact the output of the tanh function too much because it's in a flat region of the tanh function. Therefore, there's no impact on the loss. 

Indeed, the weights and the biases along with this tanh neuron do not impact the loss because the output of the tanh unit is in the flat region of the tanh function. We can change them however we want and the loss is not impacted. That's another way to justify that indeed the gradient would be basically zero. It vanishes. 

Indeed, when `t = 0`, we get `1 * gradient`. So when the tanh takes on exactly a value of zero, then the gradient is just passed through. The more you are in the flat tails of the tanh function, the more the gradient is squashed. 

In fact, the gradient flowing through the tanh function can only ever decrease. The amount that it decreases is proportional to a square, depending on how far you are in the flat tails of the tanh function. And that's what's happening in our neural network.
# Understanding Neural Networks: Gradients and Activation Functions

The primary concern in neural networks is that if all the outputs, denoted as H, are in the flat regions of negative one and one, then the gradients flowing through the network will get destroyed at this layer. However, there is a redeeming quality here, and we can actually get a sense of the problem as follows.

I've written some code to illustrate this. Essentially, we want to take a look at H, take the absolute value, and see how often it is in a flat region, say greater than 0.99. The result is a Boolean tensor. In this tensor, you get a white if the condition is true and a black if it is false. 

In this example, we have 32 examples and 200 hidden neurons. We see that a lot of this tensor is white, indicating that all these tanh neurons were very active and are in a flat tail. In all these cases, the backward gradient would get destroyed.

We would be in a lot of trouble if, for any one of these 200 neurons, the entire column is white. In that case, we have what's called a "dead neuron". This could be a tanh neuron where the initialization of the weights and the biases could be such that no single example ever activates this tanh in the active part of the tanh. If all the examples land in the tail, then this neuron will never learn. It is a dead neuron.

Upon scrutinizing this and looking for columns of completely white, we see that this is not the case. I don't see a single neuron that is all white. Therefore, it is the case that for every one of these tanh neurons, we do have some examples that activate them in the active part of the tanh. Some gradients will flow through, and this neuron will learn. The neuron will change, move, and do something.

However, you can sometimes find yourself in cases where you have dead neurons. This manifests when, no matter what inputs you plug in from your dataset, this tanh neuron always fires completely one or completely negative one. Then it will just not learn because all the gradients will be zeroed out.

This is true not just for tanh but for a lot of other non-linearities that people use in neural networks. We certainly use tanh a lot, but sigmoid will have the exact same issue because it is a squashing neuron. The same will be true for sigmoid.

The same will also apply to a ReLU (Rectified Linear Unit). ReLU has a completely flat region below zero. If you have a ReLU neuron, it is a pass-through if it is positive. If the pre-activation is negative, it will just shut it off. Since the region here is completely flat, during back propagation, this would be exactly zeroing out the gradient. All of the gradient would be set exactly to zero, instead of just a very small number depending on how positive or negative T is.

You can get, for example, a dead ReLU neuron. A dead ReLU neuron would basically look like a neuron with a ReLU nonlinearity that never activates. So, for any examples that you plug in the dataset, it never turns on. It's always in this flat region. Then this ReLU neuron is a dead neuron. Its weights and bias will never learn. They will never get a gradient because the neuron never activated. This can sometimes happen, and it's something to be aware of when working with neural networks.
# Understanding Neural Networks: Initialization, Optimization, and Activation

Neural networks are complex systems that require careful initialization and optimization. The weights and biases of a network can sometimes result in some neurons being permanently inactive, a phenomenon often referred to as 'dead neurons'. This can occur by chance during initialization or during optimization if the learning rate is too high.

For instance, if a neuron receives too much of a gradient, it can get knocked off the data manifold. From then on, no example ever activates this neuron, rendering it permanently inactive. This can be likened to permanent brain damage in the mind of a network.

If your learning rate is very high and you have a neural network with regular neurons, you might train the network and get some loss. However, if you go through the entire training set and forward your examples, you might find neurons that never activate. These are the 'dead neurons' in your network. They will never turn on and usually, during training, these neurons are changing and moving. But due to a high gradient somewhere by chance, they get knocked off and nothing ever activates them. From then on, they are just dead. This is a form of permanent brain damage that can happen to some of these neurons.

Nonlinearities like leaky ReLU will not suffer from this issue as much because it doesn't have flat tails, so you almost always get gradients. The Exponential Linear Unit (ELU) is also fairly frequently used, but it might suffer from this issue because it has flat parts. This is something to be aware of and something to be concerned about.

In some cases, we have too many activations (H) that take on extreme values. This is not optimal and not something you want, especially during initialization. The H pre-activation that's flowing to 10H is too extreme, too large. It's creating a distribution that is too saturated on both sides of the 10H. This means that there's less training for these neurons because they update less frequently.

So, how do we fix this? The H pre-activation is MCAT, which comes from C. These are uniform Gaussian but then it's multiplied by W1 plus B1. The H pre-activation is too far off from zero and that's causing the issue. We want this pre-activation to be closer to zero, very similar to what we have with logits.

It's okay to set the biases to a very small number. We can either multiply it by 0.001 to get a little bit of entropy. I sometimes like to do that just so that there's a little bit of variation and diversity in the original initialization of these 10H neurons. I find in practice that this can help optimization a little bit.

The weights can also be squashed. Let's multiply everything by 0.1. After rerunning the first batch and looking at the histogram, we can see that the pre-activations are now between -1.5 and 1.5. This means we expect much less white, indicating that there are no neurons that are saturated above 0.99.
# Understanding Neural Network Initialization

In either direction, this is actually a pretty decent place to be. It's very much about changing W1 here, so maybe we can go to point two. Okay, so maybe something like this is a nice distribution. So maybe this is what our initialization should be. 

Let's now erase these and start with initialization. Let's run the full optimization without the break and see what we get. Okay, so the optimization finished and I re-ran the loss. This is the result that we get. Just as a reminder, I put down all the losses that we saw previously in this lecture. 

We see that we actually do get an improvement here. Just as a reminder, we started off with a validation loss of 2.17. When we started by fixing the softmax being confidently wrong, we came down to 2.13. By fixing the 10h layer being way too saturated, we came down to 2.10. 

The reason this is happening, of course, is because our initialization is better and so we're spending more time doing productive training instead of not very productive training because our gradients are set to zero. We have to learn very simple things like the overconfidence of the softmax in the beginning and we're spending cycles just squashing down the weight matrix. 

This is illustrating initialization and its impacts on performance. Just by being aware of the internals of these neural nets and their activations, their gradients, now we're working with a very small network. This is just one layer multiplayer perceptron. Because the network is so shallow, the optimization problem is actually quite easy and very forgiving. 

Even though our initialization was terrible, the network still learned eventually. It just got a bit worse result. This is not the case in general though. Once we actually start working with much deeper networks that have say 50 layers, things can get much more complicated and these problems stack up. 

You can actually get into a place where the network is basically not training at all if your initialization is bad enough. The deeper your network is and the more complex it is, the less forgiving it is to some of these errors. This is something that we definitely need to be aware of and something to scrutinize, something to plot and something to be careful with. 

That's great that that worked for us, but what we have here now is all these metric numbers like point two. Where do I come up with this and how am I supposed to set these if I have a large neural net with lots and lots of layers? 

Obviously, no one does this by hand. There are actually some relatively principled ways of setting these scales that I would like to introduce to you now. 

Let me paste some code here that I prepared just to motivate the discussion of this. What I'm doing here is we have some random input here x that is drawn from a Gaussian and there are 1000 examples that are 10 dimensional. Then we have a weight and layer here that is also initialized using Gaussian just like we did here. 

These neurons in the hidden layer look at 10 inputs and there are 200 neurons in this hidden layer. Then we have here just like here, in this case, the multiplication X multiplied by W to get the pre-activations of these neurons. 

Basically, the analysis here looks at, suppose these are uniform Gaussian and these weights are uniform Gaussian. If I do x times W and we forget for now.
# Understanding Neural Networks: Bias, Non-Linearity, and Gaussian Distribution

In the beginning, the input in a neural network is a normal Gaussian distribution with a mean of zero and a standard deviation of one. The standard deviation is simply a measure of the spread of the distribution. However, once we multiply the input and look at the histogram of the output (Y), we see that the mean remains the same, about zero, because this is a symmetric operation. But, the standard deviation has expanded to three. The input standard deviation was one, but now it has grown to three. What you're seeing in the histogram is that this Gaussian is expanding.

We're expanding this Gaussian from the input, and we don't want that. We want most of the neural networks to have relatively similar activations, so a unit Gaussian roughly throughout the neural network. The question then is, how do we scale these weights to preserve the distribution to remain a Gaussian?

Intuitively, if we multiply these elements of the weights by a large number, let's say by five, then this Gaussian grows and grows in standard deviation. So now we're at 15. Basically, these numbers in the output Y take on more and more extreme values. But if we scale it down, let's say 0.2, then conversely this Gaussian is getting smaller and smaller, and it's shrinking. You can see that the standard deviation is 0.6. So the question is, what do I multiply by here to exactly preserve the standard deviation to be one?

It turns out that the correct answer, mathematically when you work out through the variance of this multiplication, is that you are supposed to divide by the square root of the fan-in. The fan-in is basically the number of input elements, in this case, 10. So we are supposed to divide by the square root of 10. When you divide by the square root of 10, we see that the output Gaussian has exactly a standard deviation of one.

Unsurprisingly, a number of papers have looked into how to best initialize neural networks. In the case of multi-layer perceptions, we can have fairly deep networks that have these non-linearities in between, and we want to make sure that the activations are well-behaved. They don't expand to infinity or shrink all the way to zero. The question is, how do we initialize the weights so that these activations take on reasonable values throughout the network?

One paper that has studied this in quite a bit of detail, often referenced, is "Delving Deep into Rectifiers" by Kaiming He et al. In this case, they actually study convolutional neural networks and they studied especially the ReLU non-linearity and the PReLU non-linearity instead of a tanh non-linearity, but the analysis is very similar.

Basically, what happens here is for them, the relation that they care about quite a bit here is a squashing function where all the negative numbers are simply clamped to zero. So the positive numbers are passed through but everything negative is just set to zero. Because you are basically throwing away half of the distribution, they find in their analysis of the forward activations in the neural net that you have to compensate for that with a gain.

In conclusion, they find that when they initialize their weights, they have to do it with a zero mean Gaussian whose standard deviation is adjusted by the square root of the fan-in. This ensures that the activations in the neural network are well-behaved and do not expand to infinity or shrink to zero.
# Understanding Neural Network Initialization

The standard deviation is the square root of 2 over the fan-in. Here, we are initializing a concussion with the square root of the fan-in. This NL here is the Fanon, so what we have is the square root of one over the fan-in because we have the division here. Now, they have to add this factor of 2 because of the ReLU (Rectified Linear Unit) which basically discards half of the distribution and clamps it at zero. That's where you get an initial factor.

In addition to that, this paper also studies not just the behavior of the activations in the forward pass of the neural net but also studies the back propagation. We have to make sure that the gradients also are well-behaved because ultimately they end up updating our parameters. What they find here, through a lot of analysis, is that if you properly initialize the forward pass, the backward pass is also approximately initialized up to a constant factor that has to do with the size of the number of hidden neurons in an early and late layer. Empirically, they find that this is not a choice that matters too much.

This timing initialization is also implemented in PyTorch. If you go to the torch.nn.net documentation, you'll find timing normal. In my opinion, this is probably the most common way of initializing neural networks now. It takes a few keyword arguments here. Number one, it wants to know the mode. Would you like to normalize the activations or would you like to normalize the gradients to always be Gaussian with zero mean and a unit or one standard deviation? Because they find in the paper that this doesn't matter too much, most people just leave it as the default which is Fan-in. 

Secondly, you need to pass in the nonlinearity that you are using because depending on the nonlinearity, we need to calculate a slightly different gain. If your nonlinearity is just linear, so there's no nonlinearity, then the gain here will be one and we have the exact same formula that we've got here. But if the nonlinearity is something else, we're going to get a slightly different gain.

For example, in the case of ReLU, this gain is the square root of 2. The reason it's a square root is because the two is inside of the square root. In the case of linear or identity, we just get a gain of one. In the case of tanh, which is what we're using here, the advised gain is 5 over 3. 

Intuitively, why do we need a gain on top of the initialization? It's because tanh, just like ReLU, is a contractive transformation. What that means is you're taking the output distribution from this matrix multiplication and then you are squashing it in some way. Now, ReLU squashes it by taking everything below zero and clamping it to zero. Tanh also squashes it because it's a contractual operation. It will take the tails and squeeze them in. In order to fight the squeezing in, we need to boost the weights a little bit so that we renormalize everything back to standard unit standard deviation. That's why there's a little bit of a gain that comes out.

I'm skipping through this section a little bit quickly and I'm doing that intentionally. The reason for that is because about seven years ago when this paper was written, you had to actually be methodical and step by step in understanding neural network initialization.
# Modern Innovations in Neural Network Initialization

In the early days of neural network training, everything was extremely delicate. You had to be extremely careful with the activations, the ingredients and their ranges, and their histograms. It was crucial to be very precise with the setting of gains and the scrutinizing of the nonlinearities used. Everything was very finicky, fragile, and had to be properly arranged for the neural network to train, especially if your neural network was very deep.

However, a number of modern innovations have made everything significantly more stable and well-behaved. It has become less important to initialize these networks exactly right. Some of these modern innovations include residual connections, the use of a number of normalization layers like batch normalization layer, group normalization, and much better optimizers, not just stochastic gradient descent. We're basically using slightly more complex optimizers like RMS prop and especially Adam. All of these modern innovations make it less important for you to precisely calibrate the initialization of the neural net.

In practice, when initializing these neural networks, I basically just normalize my weights by the square root of the fan-in. Roughly what we did here is what I do. Now, if we want to be exactly accurate, we should go by the init of the common normal. This is how it should be implemented: we want to set the standard deviation to be gain over the square root of fan-in.

When we have a torch that runs and let's say I just create a thousand numbers, we can look at the standard deviation of this. That's the amount of spread. When you take these and you multiply by say 0.2, that basically scales down the Gaussian and that makes its standard deviation 0.2. The number that you multiply by here ends up being the standard deviation of this Gaussian.

When we sample rw1, we want to set the standard deviation to gain over the square root of fan-in, which is valid. In other words, we want to multiply by gain which for 10 H is 5 over 3. Then, divide by the square root of the fan-in. In this example here, the fan-in for W1 is actually an embed times block size which is 30. That's because each character is 10 dimensional but then we have three of them and we concatenate them. So actually, the fan-in here was 30 and I should have used 30 here probably.

We want 30 square root so this is the number we want our standard deviation to be and this number turns out to be 0.3. Whereas here, just by fiddling with it and looking at the distribution and making sure it looks okay, we came up with 0.2. So instead, what we want to do here is we want to make the standard deviation be 5 over 3 which is our gain, divided by this amount, times 0.2 square root. These brackets here are not that necessary but I'll just put them here for clarity.
# Training Neural Networks with Batch Normalization

In this article, we will discuss the process of training a neural network and the importance of initialization. We will also introduce a modern innovation in the field, batch normalization, which has significantly improved the training of deep neural networks.

To begin with, let's consider the initialization of the neural network. In our case, we are using a 10h nonlinearity. We initialize the neural network by multiplying by 0.3 instead of 0.2. This method of initialization allows us to train the neural network and evaluate the results.

After training the neural network, we end up in roughly the same spot. Looking at the validation loss, we now get 2.10, which is similar to the previous result. There's a slight difference, but that's likely due to the randomness of the process.

The significant aspect of this process is that we reach the same spot without having to introduce any magic numbers that we got from just looking at histograms and guessing. We have something that is semi-principled and will scale to much larger networks. It serves as a guide for future network training.

The precise setting of these initializations is not as important today due to some modern innovations. One of these innovations is batch normalization.

Batch normalization was introduced in 2015 by a team at Google. It was an extremely impactful paper because it made it possible to train very deep neural networks quite reliably. It basically just worked.

So, what does batch normalization do? We have these hidden states, HP_act, and we don't want these pre-activation states to be too small because then the 10h is not doing anything. But we also don't want them to be too large because then the 10h is saturated. In fact, we want them to be roughly Gaussian, with zero mean and a unit or one standard deviation, at least at initialization.

The insight from the batch normalization paper is that if you want these hidden states to be roughly Gaussian, then why not take the hidden states and just normalize them to be Gaussian? It sounds kind of crazy, but you can just do that because standardizing hidden states so that they're unit Gaussian is a perfectly differentiable operation.

The idea is to make these roughly Gaussian because if these are way too small numbers, then the 10h here is kind of inactive. But if these are very large numbers, then the 10h is way too saturated, and gradients do not flow. So, we'd like this to be roughly Gaussian.

The insight in batch normalization is that we can just standardize these activations so they are exactly Gaussian. We can take HP_act and calculate the mean across the zeroth dimension, keeping the dimensions as true so that we can easily broadcast this. The shape of this is 1 by 200, meaning we are doing the mean over all the elements in the batch. Similarly, we can calculate the standard deviation.
# Understanding Neural Network Normalization

In this paper, a prescription is provided for calculating the mean and standard deviation of neuron activation in a neural network. The mean is simply the average value of any neuron's activation. The standard deviation, on the other hand, is a measure of the spread of these values. It is calculated by determining the distance of each value from the mean, squaring these distances, and then averaging them. This gives us the variance. The standard deviation is then obtained by taking the square root of the variance.

These two values, the mean and standard deviation, are used to normalize or standardize the neuron activations. This is done by subtracting the mean from each activation and then dividing by the standard deviation. This process is applied to the pre-activation values of each neuron.

The symbols used in this calculation are as follows: the mean is represented by the symbol 'mu', and the variance by 'sigma squared'. The standard deviation, usually represented by 'sigma', is the square root of the variance.

By standardizing these values, every single neuron and its firing rate will be exactly unit Gaussian on these 32 examples, at least for this batch. This is why it's called batch normalization - we are normalizing these batches.

In principle, we could train this network as is. Calculating the mean and standard deviation are just mathematical formulas that are perfectly differentiable. However, this approach won't achieve a very good result. While we want the neuron activations to be roughly Gaussian at initialization, we don't want them to be forced to always be Gaussian. We would actually like the neural network to move this distribution around, to potentially make it more diffuse or sharper. We'd like the back propagation to tell us how that distribution should move around.

To address this, the paper introduces an additional component: the scale and shift. This involves taking the normalized inputs and scaling them by some gain and offsetting them by some bias to get the final output from this layer.

This is implemented by allowing a batch normalization gain, initialized at one, and a batch normalization bias, initialized at zero. Both of these are in the shape of 1 by n hidden. The batch normalization gain multiplies the normalized inputs, and the batch normalization bias offsets them.

At initialization, each neuron's firing values in this batch will be exactly unit Gaussian. This is roughly what we want, at least at initialization. During optimization, we'll be able to back propagate to the batch normalization gain and bias, allowing the distribution of neuron activations to move around as needed.
# Understanding Neural Networks and Batch Normalization

In this article, we will discuss the concept of bias in neural networks and how to change them to give the network full ability to perform its tasks. We will also delve into the initialization of these networks and the importance of including these parameters in the neural network as they will be trained with back propagation.

Let's start by initializing the network. We will also copy the line which is the batch normalization layer. This is done in a single line of code. We will then proceed to do the exact same thing at the test. Similar to training time, we will normalize and then scale. This will give us our train and validation loss. 

We will see in a second that we're actually going to change this a little bit, but for now, let's keep it this way. So, let's wait for this to converge. Once the neural networks converge, we see that our validation loss here is roughly 2.10. This is actually kind of comparable to some of the results that we've achieved previously.

Now, I'm not actually expecting an improvement in this case. That's because we are dealing with a very simple neural network that has just a single hidden layer. In this very simple case of just one hidden layer, we were able to actually calculate what the scale of 'w' should be to make these pre-activations already have a roughly Gaussian shape. So, the batch normalization is not doing much here.

However, you might imagine that once you have a much deeper neural network that has lots of different types of operations and there's also, for example, residual connections which we'll cover later, it will become very difficult to tune those scales of your weight matrices such that all the activations throughout the neural networks are roughly Gaussian. 

That's going to become very quickly intractable. But compared to that, it's going to be much easier to sprinkle batch normalization layers throughout the neural network. In particular, it's common to look at every single linear layer like this one. This is a linear layer multiplying by a weight matrix and adding the bias. Or for example, convolutions which we'll cover later and also perform basically a multiplication with the weight matrix but in a more spatially structured format. 

It's customary to take these linear layers or convolutional layers and append a batch normalization layer right after it to control the scale of these activations at every point in the neural network. So, we'd be adding these batch normalization layers throughout the neural network and then this controls the scale of these activations throughout the neural network. 

It doesn't require us to do perfect mathematics and care about the activation distributions for all these different types of neural network Lego building blocks that you might want to introduce into your neural network. It significantly stabilizes the training and that's why these layers are quite popular.

Now, the stability offered by batch normalization actually comes at a terrible cost. If you think about what's happening here, something terribly strange and unnatural is happening. It used to be that we have a single example feeding into a neural network and then we calculate these activations and its logits. This is a deterministic process so you arrive at some logits for this example. 

Then, because of efficiency of training, we suddenly started to use batches of examples. But those batches of examples were processed independently and it was just an efficiency thing. But now, suddenly in batch normalization, something different is happening.
# The Impact of Normalization in Neural Networks

Because of the normalization through the batch, we are coupling these examples mathematically in the forward pass and the backward pass of the neural land. Now, the hidden state activations, `hpact`, and your logits for any one input example are not just a function of that example and its input, but they're also a function of all the other examples that happen to come for a ride in that batch. 

These examples are sampled randomly. So, what's happening is, for example, when you look at each `preact` that's going to feed into `H` (the hidden state activations), for any one of these input examples, it's going to actually change slightly depending on what other examples there are in a batch. Depending on what other examples happen to come for a ride, `H` is going to change subtly. 

If you imagine sampling different examples, the statistics of the mean and the standard deviation are going to be impacted. So, you'll get a jitter for `H` and you'll get a jitter for logits. You might think that this would be a bug or something undesirable, but in a very strange way, this actually turns out to be good in neural network training. 

As a side effect, you can think of this as a kind of regularizer. What's happening is you have your input and you get your `H`, and then depending on the other examples, this is generating a bit. So, what that does is that it's effectively padding out any one of these input examples and it's introducing a little bit of entropy. 

Because of the padding out, it's actually kind of like a form of data augmentation. It's kind of like augmenting the input a little bit and it's jittering it. That makes it harder for the neural nets to overfit to these concrete specific examples. By introducing all this noise, it actually pads out the examples and it regularizes the neural net. That's one of the reasons why, deceivingly as a second-order effect, this is actually a regularizer. 

This has made it harder for us to remove the use of batch normalization. Basically, no one likes this property that the examples in the batch are coupled mathematically and in the forward pass. It leads to a lot of bugs and strange results. 

People have tried to deprecate the use of batch normalization and move to other normalization techniques that do not couple the examples of a batch. Examples are layer normalization, instance normalization, group normalization, and so on. 

Long story short, batch normalization was the first kind of normalization layer to be introduced. It worked extremely well, it happens to have this regularizing effect, it stabilized training. People have been trying to remove it and move to some of the other normalization techniques, but it's been hard because it just works quite well. Some of the reason that it works quite well is again because of this regularizing effect and because it is quite effective at controlling the activations and their distributions. 

One of the strange outcomes of this coupling is when evaluating the loss on the validation side. Once we've trained a neural net, we'd like to deploy it in some kind of a setting and we'd like to be able to.
# Batch Normalization in Neural Networks

In this article, we will discuss how to feed a single individual example into a neural network and get a prediction out. This process becomes a bit complex when our neural network, in a forward pass, estimates the statistics of the mean and standard deviation of a batch. The neural network expects batches as an input, so how do we feed in a single example and get sensible results out?

The proposal in the batch normalization paper is to have a step after training that calculates and sets the batch mean and standard deviation a single time over the training set. We call this process "calibrating the batch statistics". 

Here's how it works: We use the `no_grad` function in PyTorch, which tells the system that none of this will call the `backward` function, making it more efficient. We take the training set, get the pre-activations for every single training example, and then one single time estimate the mean and standard deviation over the entire training set. We then get the batch mean and standard deviation. These are now fixed numbers representing the mean of the entire training set. 

Instead of estimating it dynamically, we use the batch mean and standard deviation. At this time, we fix these values and use them during inference. The result is basically identical to the previous method, but the benefit is that we can now also forward a single example because the mean and standard deviation are now fixed tensors.

However, estimating the mean and standard deviation as a second stage after neural network training is not ideal because it adds an extra step. The batch normalization paper introduced one more idea: we can estimate the mean and standard deviation in a running manner during training of the neural network. This way, we have a single stage of training and on the side of that training, we are estimating the running mean and standard deviation.

Here's how it works: We take the mean that we are estimating on the batch and call this the batch mean on the `i` iteration. The mean comes here and the standard deviation comes here. We then keep a running mean of both of these values during training. 

In PyTorch, these running mean and standard deviation are not actually part of the gradient-based optimization. We never derive gradients with respect to them; they are updated on the side of training. We use the `torch.no_grad` function to tell PyTorch that the update here is not supposed to be building out. 

In the beginning, because of the way we initialized `W1` and `B1`, each react will be roughly unit Gaussian so the mean will be roughly zero and the standard deviation roughly one. We initialize these that way, but then we update these during the training process. 

This is how we implement batch normalization in neural networks. It allows us to feed in a single example and get sensible results out, making the process more efficient and streamlined.
# Understanding Batch Normalization

We can visualize the process of batch normalization using a graph. This will help us understand the process without any doubt. The process is essentially running backward. However, this running is going to be 0.99 times the current value plus 0.001 times the new mean value. 

In the same way, the standard deviation (STD) running will be updated, but it will receive a small update in the direction of what the current standard deviation is. As you can see, this update is outside and on the side of the gradient-based optimization. It's simply being updated not using gradient descent, but rather in a gen key-like smooth running mean manner. 

While the network is training and these pre-activations are changing and shifting around during backpropagation, we are keeping track of the typical mean and standard deviation. We're estimating them once and keeping track of this in a running manner. 

What we're hoping for, of course, is that the mean running and standard deviation running are going to be very similar to the ones that we calculated before. That way, we don't need a second stage because we've sort of combined the two stages and put them on the side of each other. 

This is also how it's implemented in the batch normalization layer in PyTorch. During training, the exact same thing will happen. Then later, when you're using inference, it will use the estimated running mean and standard deviation of those hidden states. 

Let's wait for the optimization to converge. Hopefully, the running mean and standard deviation are roughly equal to these two. Then we can simply use it here and we don't need this stage of explicit calibration at the end. 

After the optimization finishes, I'll rerun the explicit estimation. The mean from the explicit estimation is very similar to the mean from the running estimation during the optimization. It's not identical but it's pretty close. In the same way, the standard deviation is similar to the standard deviation running. They are fairly similar values, not identical but pretty close. 

Then here, instead of using the mean, we can use the running mean. Instead of using the standard deviation, we can use the running standard deviation. Hopefully, the validation loss will not be impacted too much. It's basically identical and this way we've eliminated the need for this explicit stage of calibration because we are doing it in line over here. 

We're almost done with batch normalization. There are only two more notes that I'd like to make. 

First, I've skipped a discussion over what is this plus Epsilon doing here. This Epsilon is usually like some small fixed number, for example, one a negative five by default. What it's doing is that it's basically preventing a division by zero in the case that the variance over your batch is exactly zero. In that case, we normally have a division by zero but because of the plus Epsilon, this is going to become a small number in the denominator instead and things will be more well behaved. So feel free to also add a plus Epsilon here of a very small number. It doesn't actually substantially change the result. 

Second, I want you to notice that we're being methodical and taking this step by step. This is unlikely to happen in our very simple example here.
# Understanding Batch Normalization in Neural Networks

In this article, we will delve into the concept of batch normalization in neural networks. This is a subtle yet crucial aspect that can sometimes be overlooked. 

Let's start by looking at a common scenario where we add bias into `hpact`. However, these biases are essentially useless because we add them to `hpact` and then calculate the mean for each of these neurons, subtracting it. So, whatever bias you add here is going to be subtracted right away. These biases are not doing anything; in fact, they're being subtracted out and don't impact the rest of the calculation. If you look at `b1.grad`, it's actually going to be zero because it's being subtracted out and doesn't have any effect.

Whenever you're using batch normalization layers, if you have any weight layers before, like a linear or a comma, you're better off not using bias. You don't want to add it because it's spurious. Instead, we have this batch normalization bias, which is now in charge of the biasing of this distribution instead of the `B1` we had originally. 

The rationalization layer has its own bias, and there's no need to have a bias in the layer before it because that bias is going to be subtracted anyway. This is a small detail to be careful with. Sometimes it's not going to do anything catastrophic. This `B1` will just be useless; it will never get any gradient, it will not learn, it will stay constant, and it's just wasteful. But it doesn't really impact anything otherwise.

## Summary of Batch Normalization Layer

Batch normalization is used to control the statistics of activations in the neural net. It is common to sprinkle batch normalization layers across the neural net, usually placing them after layers that have multiplications, like a linear layer or a convolutional layer.

The batch normalization internally has parameters for the gain and the bias, which are trained using backpropagation. It also has two buffers: the mean and the standard deviation (the running mean and the running mean of the standard deviation). These are not trained using backpropagation; instead, they are trained using a running mean update.

The batch normalization layer calculates the mean and standard deviation of the activations feeding into it over that batch. It then centers that batch to be a unit Gaussian and offsets and scales it by the learned bias and gain. On top of that, it keeps track of the mean and standard deviation of the inputs, maintaining a running mean and standard deviation. This will later be used at inference so that we don't have to re-estimate the mean and standard deviation all the time. This also allows us to forward individual examples at test time.

Batch normalization is a fairly complicated layer, but understanding its internal workings is crucial. For instance, ResNet, a residual neural network, uses convolutional neural networks, which heavily rely on batch normalization.
# Image Classification and Residual Networks

In this article, we will delve into the concept of image classification using Residual Networks (ResNets). We won't go into the intricate details of ResNets, but it's important to note that an image feeds into a ResNet at the top, and there are many layers with a repeating structure all the way to the predictions of what's inside that image. 

This repeating structure is made up of blocks, and these blocks are sequentially stacked up in this deep neural network. The code for this block, which is used and repeated sequentially in series, is called the "bottleneck block". 

## Understanding the Bottleneck Block

There's a lot to unpack in the bottleneck block, which is written in PyTorch. We won't cover all of it, but we will point out some key pieces. 

In the `init` section, we initialize the neural net. This block of code is where we initialize all the layers. In the `forward` section, we specify how the neural net acts once you actually have the input. 

These blocks are replicated and stacked up serially, forming the structure of a residual network. 

## Convolutional Layers

Notice the convolutional layers, denoted as `conv1`. Convolutional layers are similar to linear layers, except they are used for images and have spatial structure. Essentially, they perform the linear multiplication and bias offset on patches of the input, rather than the full input. 

## Normalization and Nonlinearity

After the convolutional layers, we have the normalization layer, which is initialized to be a two-dimensional batch normalization layer by default. Following this, we have a nonlinearity like ReLU. In our case, we are using `tanh`, but both are just nonlinearities and can be used relatively interchangeably. For very deep networks, ReLU typically works a bit better empirically. 

## The Repeating Motif

The repeating motif in this structure is a weight layer (like a convolution or linear layer), a normalization layer, and a nonlinearity. This is the motif that you would be stacking up when you create these deep neural networks. 

When initializing the convolutional layers, like `conv1x1`, the depth for that is specified. It's initializing an `nn.Conv2d`, which is a convolutional layer in PyTorch. There are several keyword arguments here that we won't delve into, but it's important to note that `bias=False` is specified. This is because after the weight layer, there's a batch normalization which subtracts the bias and then has its own bias. Therefore, there's no need to introduce these spurious parameters. It wouldn't hurt performance, but it's simply unnecessary. 

In conclusion, the motif of convolution, batch normalization, and ReLU eliminates the need for a bias in the convolutional layer because there's a bias inside the batch normalization. This is the fundamental structure of a Residual Network used for image classification.
# Understanding PyTorch Implementation of Residual Neural Network

By the way, this example here is very easy to find. Just search for "ResNet PyTorch" and you'll find this example. This is the stock implementation of a Residual Neural Network in PyTorch. You can find it here, but of course, I haven't covered many of these parts yet. 

I would also like to briefly descend into the definitions of these PyTorch layers and the parameters that they take. Now, instead of a convolutional layer, we're going to look at a linear layer because that's the one that we're using here. This is a linear layer and I haven't covered convolutions yet. But as I mentioned, convolutions are basically linear layers except on patches. 

A linear layer performs a `Wx + b` operation, except here they're calling the `W` as `A transpose`. To initialize this layer, you need to know the fan-in and the fan-out. This is so that they can initialize this `W`. This is the fan-in and the fan-out, so they know how big the weight matrix should be. 

You also need to pass in whether or not you want a bias. If you set it to false, then no bias will be inside this layer. You may want to do that, exactly like in our case, if your layer is followed by a normalization layer such as BatchNorm. This allows you to basically disable bias. 

In terms of the initialization, this is reporting the variables used inside this linear layer. Our linear layer here has two parameters: the weight and the bias. They're talking about how they initialize it by default. By default, PyTorch initializes your weights by taking the fan-in and then doing `1 / sqrt(fan-in)`. Instead of a normal distribution, they are using a uniform distribution. 

It's very much the same thing, but they are using a `1` instead of `sqrt(5/3)`. So there's no gain being calculated here, the gain is just one. But otherwise, it's exactly `1 / sqrt(fan-in)` exactly as we have here. So `1 / sqrt(K)` is the scale of the weights. But when they are drawing the numbers, they're not using a Gaussian by default, they're using a uniform distribution by default. They draw uniformly from `-sqrt(K)` to `sqrt(K)`. 

The reason they're doing this is if you have a roughly Gaussian input, this will ensure that out of this layer you will have a roughly Gaussian output. You basically achieve that by scaling the weights by `1 / sqrt(fan-in)`. 

The second thing is the Batch Normalization layer. Let's look at what that looks like in PyTorch. Here we have a one-dimensional Batch Normalization layer, exactly as we are using here. There are a number of keyword arguments going into it as well. 

We need to know the number of features. For us, that is `200` and that is needed so that we can initialize these parameters here: the gain, the bias, and the buffers for the running mean and standard deviation. 

Then they need to know the value of Epsilon here and by default, this is `1e-5`. You don't typically change this too much. Then they need to know the momentum. The momentum here is basically used for the running mean and running standard deviation.
# Understanding Activations and Gradients in Neural Networks

In this lecture, we will discuss the importance of understanding the activations and gradients in neural networks. This becomes increasingly important, especially as you make your neural networks bigger, larger, and deeper.

## The Importance of Activations

We will start by looking at the distributions at the output layer. If you have two confident mispredictions because the activations are too messed up at the last layer, you can end up with these hockey stick losses. If you fix this, you get a better loss at the end of training because your training is not doing wasteful work.

We also need to control the activations. We don't want them to squash to zero or explode to infinity. If that happens, you can run into a lot of trouble with all of these non-linearities in these neural networks. Basically, you want everything to be fairly homogeneous throughout the neural network. You want roughly Gaussian activations throughout the neural network.

## Weight Matrices and Biases

Next, we will talk about how to scale these weight matrices and biases during the initialization of the neural network. This is important so that we don't get everything as controlled as possible. This gave us a large boost in improvement.

## Batch Normalization

We will also discuss batch normalization. By default, the momentum here is 0.1. The momentum we are using here in this example is 0.001. You may want to change this sometimes. 

If you have a very large batch size, typically what you'll see is that when you estimate the mean and the standard deviation for every single batch size, if it's large enough, you're going to get roughly the same result. Therefore, you can use slightly higher momentum like 0.1. 

But for a batch size as small as 32, the mean and standard deviation here might take on slightly different numbers because there's only 32 examples we are using to estimate the mean of standard deviation. So the value is changing around a lot and if your momentum is 0.1, that might not be good enough for this value to settle and converge to the actual mean and standard deviation over the entire training set.

If your batch size is very small, a momentum of 0.1 is potentially dangerous. It might make it so that the running mean and standard deviation is thrashing too much during training and it's not actually converging properly.

The `alpha` parameter determines whether the batch normalization layer has these learnable affine parameters, the gain, and the bias. This is almost always kept at true. I'm not actually sure why you would want to change this to false.

The `track_running_stats` parameter determines whether or not the batch normalization layer of PyTorch will be doing this. One reason you may want to skip the running stats is because you may want to, for example, estimate them at the end as a stage two. In that case, you don't want the batch normalization layer to be doing all this extra compute that you're not going to use.

Finally, we need to know which device we're going to run this batch normalization on, a CPU or a GPU, and what the data type should be, half precision, single precision, double precision, and so on.

That's the batch normalization layer. The link to the paper is the same formula we've implemented and everything is the same exactly as we've done here.

## Conclusion

In conclusion, understanding the activations and their statistics in neural networks is crucial. We need to control the activations and scale the weight matrices and biases during initialization. We also need to understand the importance of batch normalization and its parameters. This understanding will help us build better and more efficient neural networks.
# Deep Learning: Neural Networks and Normalization Layers

That strategy is not actually possible for much deeper neural networks. When you have much deeper neural networks with lots of different types of layers, it becomes really hard to precisely set the weights and the biases in such a way that the activations are roughly uniform throughout the neural network.

So then, I introduced the notion of the normalization layer. Now, there are many normalization layers that people use in practice: batch normalization layer, constant normalization, group normalization. We haven't covered most of them, but I've introduced the first one, and also the one that I believe came out first, and that's called batch normalization.

We saw how batch normalization works. This is a layer that you can sprinkle throughout your deep neural network. The basic idea is if you want roughly Gaussian activations, well then take your activations and take the mean and standard deviation and center your data. You can do that because the centering operation is differentiable.

But, on top of that, we actually had to add a lot of bells and whistles. That gave you a sense of the complexities of the batch normalization layer. Because now we're centering the data, that's great, but suddenly we need the gain and the bias, and now those are trainable. 

And then, because we are coupling all the training examples, now suddenly the question is how do you do the inference? To do the inference, we need to now estimate these mean and standard deviation once for the entire training set and then use those at inference. But then, no one likes to do stage two, so instead, we fold everything into the batch normalization layer during training and try to estimate these in a running manner so that everything is a bit simpler. That gives us the batch normalization layer.

As I mentioned, no one likes this layer. It causes a huge amount of bugs. Intuitively, it's because it is coupling examples in the forward pass of a neural network. I've shot myself in the foot with this layer over and over again in my life, and I don't want you to suffer the same. So, basically, try to avoid it as much as possible. 

Some of the other alternatives to these layers are, for example, group normalization or layer normalization, and those have become more common in more recent deep learning. But we haven't covered those yet. Definitely, batch normalization was very influential at the time when it came out in roughly 2015 because it was the first time that you could train reliably much deeper neural networks. Fundamentally, the reason for that is because this layer was very effective at controlling the statistics of the activations in a neural network.

So that's the story so far. In the future lectures, hopefully, we can start going into recurring neural networks. Recurring neural networks, as we'll see, are just very deep networks because you unroll the loop. When you actually optimize these neurons, that's where a lot of this analysis around the activation statistics and all these normalization layers will become very important for good performance. We'll see that next time.

As a bonus, I would like us to do one more summary here. It's useful to have one more summary of everything I've presented in this lecture. But also, I would like us to start by modifying our code a little bit so it looks much more like what you would encounter in PyTorch. You'll see that I will structure our code into these modules like a linear module and a...
# Building Neural Networks with PyTorch

In this tutorial, we will be creating a linear layer and a batch normalization layer using PyTorch. We will then construct a neural network and run an optimization loop. Additionally, we will examine the activation statistics in both the forward and backward pass. Finally, we will evaluate and sample the network.

## Creating a Linear Layer

PyTorch's `torch.nn` module provides a variety of layer types, one of which is the linear layer. The linear layer takes a number of input features, output features, whether or not we should have bias, and the device that we want to place this layer on, as well as the data type. 

In our implementation, we will omit the device and data type, but we will include the number of inputs (`fan_in`), the number of outputs (`fan_out`), and whether or not we want to use a bias. 

Internally, this layer contains a weight and a bias. It is common to initialize the weight using random numbers drawn from a Gaussian distribution. The Kaiming initialization, which we have discussed in this lecture, is a good default and is also the default used by PyTorch. By default, the bias is usually initialized to zeros.

When you call this module, it will calculate `W * X + B` if you have `B`. When you call `parameters` on this module, it will return the tensors that are the parameters of this layer.

## Creating a Batch Normalization Layer

Next, we will create a batch normalization layer. This is very similar to PyTorch's `nn.BatchNorm1d` layer. We will take three parameters: the dimensionality, the epsilon that we'll use in the division, and the momentum that we will use in keeping track of the running stats (the running mean and the running variance).

In our implementation, `affine` will be true, meaning that we will be using a gamma and beta after the normalization. `track_running_stats` will also be true, so we will be keeping track of the running mean and the running variance in the pattern. By default, our device is the CPU and the data type is `float32`.

Batch normalization layers have a different behavior depending on whether you are training your model or running it in evaluation mode. When we are training, we use the mean and the variance estimated from the current batch. During inference, we use the running mean and running variance. If we are training, we update the mean and variance, but if we are testing, these are kept fixed. 

To handle this, we use a `training` flag, which is true by default, just like in PyTorch. The parameters in `BatchNorm1d` are the gamma and the beta, and the running mean and running variance are called buffers in PyTorch.

In the next part of this tutorial, we will construct our neural network and run the optimization loop. Stay tuned!
# Understanding the Initialization of Batch Normalization and Tanh Layers

In this article, we will discuss the initialization of batch normalization and Tanh layers in a neural network. 

## Batch Normalization Layer

The batch normalization layer is initialized with two parameters, gamma and beta. These parameters are initialized with ones and zeros respectively. The layer also has two buffers, running mean and running variance, which are initialized with zeros. 

The nomenclature and these buffers are trained using an exponential moving average. It's important to note that these buffers are not part of the back propagation and stochastic gradient descent. They are not parameters of this layer. That's why when we calculate the parameters, we only return gamma and beta, not the mean and the variance. These buffers are trained internally during every forward pass using an exponential moving average.

## Forward Pass

In a forward pass, if we are training, we use the mean and the variance estimated by the batch. We calculate the mean and the variance. In the paper, they calculate the variance, which is the standard deviation squared, and that's what's kept track of in the running variance instead of a running standard deviation. 

If we are not training, then we use the running mean and variance. We normalize and then calculate the output of this layer, assigning it to an attribute called `dot out`. This attribute is used in our modules for maintaining all those variables so that we can create statistics of them and plot them. 

Finally, we update the buffers using the provided momentum. Importantly, we use the `torch.no_grad` context manager. This is because if we don't use this, PyTorch will start building out an entire computational graph out of these tensors because it is expecting that we will eventually call `backward`. But we are never going to be calling `backward` on anything that includes running mean and running variance. This context manager makes the process more efficient and tells PyTorch that we only want to update the tensors.

## Tanh Layer

The Tanh layer is very similar to `torch.tanh` and doesn't do too much. It just calculates the Tanh function as you might expect. There are no parameters in this layer.

## Stacking Layers

Because these are layers, it now becomes very easy to stack them up into a list. We can do all the initializations that we're used to. We have the initial embedding matrix and our layers, and we can call them sequentially. 

With `torch.no_grad`, there are some initializations here. We want to make the output softmax a bit less confident. In addition to that, because we are using a six-layer multi-layer perceptron here, we are going to be using the game here and playing with this in a second. 

Finally, the parameters are the embedding matrix and all the parameters in all the layers. Notice here we're using a double list comprehension. For every layer in layers and for every parameter in the layer, we return the parameters.
# Deep Learning: Understanding Layers and Parameters

In this deep learning model, we are stacking up multiple layers, each with its own set of parameters. In total, we have 46,000 parameters. 

We are using a standard approach where we sample a batch and perform a forward pass. The forward pass is simply a linear application of all the layers in order, followed by the cross-entropy. 

During the backward path, for every single layer, we iterate over all the outputs and instruct PyTorch to retain the gradient of them. We then set all the gradients to None, perform the backward pass to fill in the gradients, and update using the gradient descent. We also track some statistics and break after a single iteration.

## Visualizing the Forward Pass Activations

In this section, we visualize the histograms of the forward pass activations at the Tanh layers. We iterate over all the layers except for the very last one, which is essentially just the Softmax layer. 

We use Tanh layers because they have a finite output range from -1 to 1, making them easy to visualize. We calculate the mean, standard deviation, and percent saturation of the output tensor from that layer. 

The percent saturation is defined as the proportion of the tensor's absolute value that is greater than 0.97. We want to avoid high saturation because when we are in the tails of the Tanh function, it will stop the gradients.

We then call `torch.histogram` and plot this histogram. Each different type of layer is represented by a different color. We are looking at how many values in these tensors take on any of the values on the x-axis. 

The first layer is fairly saturated, but then everything stabilizes. If we had more layers, they would stabilize at around a standard deviation of about 0.65, and the saturation would be roughly five percent. This stabilization and nice distribution occur because the gain is set to 5/3.

## The Role of Gain in Initialization

During initialization, we iterate over all the layers. If it's a linear layer, we boost the initialization by the gain. By default, we initialize with 1 over the square root of fan-in. 

If we don't use a gain, the standard deviation shrinks, and the saturation comes to zero. The first layer is decent, but further layers shrink down to zero. This shrinking happens because when we have a sandwich of linear layers alone, initializing our weights in this manner would have conserved the standard deviation of one. 

However, because we have interspersed Tanh layers, which are squashing functions, they slightly squash the distribution. Therefore, some gain is necessary to keep expanding it to fight the squashing. It turns out that 5/3 is a good value for the gain. If we have something too small like one, the standard deviation and saturation will shrink to zero.
# Understanding PyTorch Initialization and Activation

Let's start by examining the concept of "coming towards zero". If something is too high, let's say two, we can observe that the saturations are trying to be way too large. For instance, a value of three would create overly saturated activations. 

A good setting for a sandwich of linear layers with 10-inch activations is five over three. This setting roughly stabilizes the standard deviation at a reasonable point. However, I must admit, I have no idea where the value of five over three came from in PyTorch when we were looking at the coming initialization. 

Empirically, I can see that it stabilizes this sandwich of linear and 10h, and that the saturation is in a good range. But I don't actually know if this came out of some mathematical formula. I tried searching briefly for where this comes from, but I wasn't able to find anything. 

Certainly, we see that empirically, these are very nice ranges. Our saturation is roughly five percent, which is a pretty good number, and this is a good setting of the gain in this context.

We can do the exact same thing with the gradients. Here is the very same loop if it's a 10h, but instead of taking the layer that's out, I'm taking the grad. I'm also showing the mean and the standard deviation and plotting the histogram of these values. 

You'll see that the gradient distribution is fairly reasonable. In particular, what we're looking for is that all the different layers in this sandwich have roughly the same gradient. Things are not shrinking or exploding. 

For example, if we take a look at what happens if this is too high, you'll see that the activations are shrinking to zero, but also the gradients are doing something weird. The gradient started out here and then now they're expanding out. 

Similarly, if we have a too high gain, like three, then we see that also the gradients have some asymmetry going on. As you go into deeper and deeper layers, the activations are also changing. That's not what we want. 

In this case, we saw that without the use of Batch Normalization, we have to very carefully set those gains to get nice activations in both the forward pass and the backward pass. 

Now, before we move on to Batch Normalization, I would also like to take a look at what happens when we have no 10h units here. So, erasing all the 10h nonlinearities but keeping the gain at 5 over 3, we now have just a giant linear sandwich. 

Let's see what happens to the activations. As we saw before, the correct gain here is one. That is the standard deviation preserving gain. So, 1.667 is too high and so what's going to happen now is the following. 

The activations started out on the blue and have by layer 4 become very diffuse. So, what's happening to the activations is this. With the gradients on the top layer, the activation gradient statistics are the purple and then they diminish as you go down deeper in the layers. 

Basically, you have an asymmetry like in the neural net and you might imagine that if you have very deep neural networks, say like 50 layers or something like that, this just isn't a good place to be. That's why before Batch Normalization, this was incredibly tricky to set. In particular, if this is too large of a gain, this happens and if it's too small, the opposite happens.
# Understanding Neural Networks: A Deep Dive into Initialization and Optimization

When training a neural network, there are several factors to consider. One of these is the gain, which can significantly impact the performance of the network. If the gain is too high, the network can explode, and if it's too low, it can vanish. This is not what you want. In this case, the correct setting of the gain is exactly one, just like we're doing at initialization. 

When the gain is set correctly, the statistics for the forward and the backward pass are well behaved. This is crucial for getting neural networks to train before the use of normalization layers and advanced optimizers like Adam, which we still have to cover, and residual connections. 

Training neural networks is like a total balancing act. You have to make sure that everything is precisely orchestrated. You have to care about the activations and the gradients and their statistics. Then, maybe you can train something. However, it was basically impossible to train very deep networks, and this is fundamentally the reason for that. You'd have to be very careful with your initialization.

You might be asking yourself, why do we need these tanh layers at all? Why do we include them and then have to worry about the gain? The reason for that is that if you just have a stack of linear layers, then certainly we're getting very easily nice activations and so on. But this is just a massive linear sandwich and it collapses to a single linear layer in terms of its representation power. 

If you were to plot the output as a function of the input, you're just getting a linear function. No matter how many linear layers you stack up, you still just end up with a linear transformation. All the WX plus B's just collapse into a large WX plus B with slightly different W's and B's. 

Interestingly, even though the forward pass collapses to just a linear layer, because of back propagation and the dynamics of the backward pass, the optimization is not identical. You actually end up with all kinds of interesting dynamics in the backward pass because of the way the chain rule is calculating it. 

Optimizing a linear layer by itself and optimizing a sandwich of 10 linear layers, in both cases, those are just a linear transformation in the forward pass. But the training dynamics would be different. There are entire papers that analyze infinitely layered linear layers and so on. 

The technical linearities allow us to turn this sandwich from just a linear function into a neural network that can, in principle, approximate any arbitrary function.

Now, let's reset the code to use the linear tanh sandwich like before and reset everything so the gain is five over three. We can run a single step of optimization and look at the activation statistics of the forward pass and the backward pass. 

But there's one more plot here that is really important to look at when you're training your neural networks. Ultimately, what we're doing is updating the parameters of the neural network. So we care about the parameters and their values and their gradients. 

Here, we're iterating over all the parameters available and restricting it to the two-dimensional parameters, which are basically the weights of these linear layers. We're skipping the biases and skipping the one-dimensional parameters. This is a crucial step in understanding and optimizing your neural networks.
# Understanding Neural Network Weights

In this article, we will delve into the intricacies of neural network weights. We will examine the different weights, their shapes, and their implications. This will include the embedding layer, the first linear layer, all the way to the very last linear layer. We will also look at the mean and the standard deviation of all these parameters.

## Histogram of Weights

When we look at the histogram of these weights, it doesn't look that amazing. There seems to be some trouble in paradise. Even though the gradients look okay, there's something weird going on here. We will get to that in a second.

## Gradient to Data Ratio

The last thing we will look at is the gradient to data ratio. Sometimes, it's helpful to visualize this as well. This gives you a sense of the scale of the gradient compared to the scale of the actual values. This is important because we're going to end up taking a step update that is the learning rate times the gradient onto the data. 

If the gradient has too large of magnitude, if the numbers in there are too large compared to the numbers in data, then you'd be in trouble. But in this case, the gradient to data is our loan numbers. The values inside the gradient are 1000 times smaller than the values inside data in these weights, for most of them.

## The Last Layer

Notably, that is not true about the last layer. The last layer, the output layer, is a bit of a troublemaker in the way that this is currently arranged. The last layer here in pink takes on values that are much larger than some of the values inside the neural net. 

The standard deviations are roughly 1 and negative three throughout, except for the last layer which actually has roughly one e negative two a standard deviation of gradients. So, the gradients on the last layer are currently about 10 times greater than all the other weights inside the neural net. 

That's problematic because in the simple stochastically setup, you would be training this last layer about 10 times faster than you would be training the other layers at initialization. This actually kind of fixes itself a little bit if you train for a bit longer.

## Training for Longer

For example, if we train for more than 1000 steps, we can look at the forward pass. You see how the neurons are a bit saturating a bit and we can also look at the backward pass. But otherwise, they look good. They're about equal and there's no shrinking to zero or exploding to infinities.

In the weights, things are also stabilizing a little bit. The tails of the last pink layer are actually coming down during the optimization. But certainly, this is a little bit troubling, especially if you are using a very simple update rule like stochastic gradient descent instead of a modern optimizer like Adam.

## Update to Data Ratio

I'd like to show you one more plot that I usually look at when I train neural networks. Basically, the gradient to data ratio is not actually that informative because what matters at the end is not the gradient to date ratio but the update to the data ratio. That is the amount by which we will actually change the data in these tensors.

So, I'd like to introduce a new update to data ratio. We're going to build it out every single iteration and keep track of it.
# Understanding the Ratio of Updates to Values in Tensors

In this article, we will be discussing the ratio of updates to values in tensors. This ratio is calculated at every single iteration. Without any gradients, we are comparing the update, which is the learning rate times the gradient. This update is what we're going to apply to every parameter. 

We then take the standard deviation of the update we're going to apply and divide it by the actual content of that parameter and its standard deviation. This gives us the ratio of how great the updates are to the values in these tensors. 

To make it easier to visualize, we're going to take a log of it, specifically a log base 10. We're going to be looking at the exponents of this division. We then convert it to a float and keep track of this for all the parameters, adding it to the UD tensor. 

After reinitializing and running a thousand iterations, we can look at the activations, the gradients, and the parameter gradients as we did before. But now, we have one more plot to introduce. 

In this plot, we're going to parameters every interval and constraining it to just the weights. The number of dimensions in these tensors is two. We're plotting all of these update ratios over time. 

During initialization, these ratios evolve to take on certain values. These updates then start stabilizing usually during training. 

We're also plotting an approximate value that is a rough guide for what it roughly should be. It should be roughly one in negative three. This means that there are some values in this tensor and they take on certain values. The updates to them at every single iteration are no more than roughly 1/1000 of the actual magnitude in those tensors. 

If this was much larger, for example, if the log of this was negative one, this would mean that the values are undergoing a lot of change. 

The final layer here is an outlier because this layer was artificially shrunk down to keep the softmax unconfident. We multiplied the weight by 0.1 in the initialization to make the last layer prediction less confident. This artificially made the values inside that tensor way too low. That's why we're getting temporarily a very high ratio. But this stabilizes over time once that weight starts to learn. 

I like to look at the evolution of this update ratio for all my parameters usually. I like to make sure that it's not too much above one in negative three roughly. If it's below negative three usually, that means that the parameters are not training fast enough. 

For example, if our learning rate was very low, let's initialize and then let's actually do a learning rate of say y negative.
# Understanding Neural Network Calibration

In this article, we will discuss the calibration of neural networks, focusing on the learning rate, activation plot, and the introduction of batch normalization layers.

Firstly, let's consider the learning rate. If you observe that the updates are way too small, it could be a symptom of training too slow. This could be due to the size of the update being 10,000 times in magnitude to the size of the numbers in the tensor. This is one way to set the learning rate and get a sense of what that learning rate should be. 

If the learning rate is a little bit on the higher side, you might see that we're above the black line of negative three, somewhere around negative 2.5. However, if everything is somewhat stabilizing, this could be a decent setting of learning rates. 

When things are miscalibrated, you will notice very quickly. For example, if we forgot to apply the fan-in normalization, the weights inside the linear layers would just be a sample from a Gaussian in all those stages. The activation plot will tell you that your neurons are way too saturated and the gradients are going to be all messed up. The histogram for these weights will also be messed up, with a lot of asymmetry. 

There could also be a lot of discrepancy in how fast these layers are learning. Some of them might be learning way too fast, with ratios of negative one or negative 1.5. These aren't very large numbers in terms of this ratio. Ideally, you should be somewhere around negative three and not much more than that. 

These kinds of plots are a good way of bringing miscalibrations to your attention so you can address them. 

When we have a linear 10h sandwich, we can precisely calibrate the gains and make the activations, the gradients, and the parameters and the updates all look pretty decent. However, it definitely feels a little bit like balancing a pencil on your finger because this gain has to be very precisely calibrated.

To help fix this problem, we can introduce batch normalization layers. The standard typical place you would place it is between the linear layer, right after it but before the nonlinearity. However, people have definitely played with that and you can get very similar results even if you place it after the nonlinearity. 

It's also totally fine to place it at the end, after the last linear layer and before the loss function. In this case, the output would be world cup size. Now, because the last layer is mushroom, we would not be changing the weight to make the softmax less. 

In conclusion, understanding the calibration of neural networks is crucial in ensuring the efficiency and effectiveness of your model. By paying attention to the learning rate, activation plot, and the introduction of batch normalization layers, you can ensure that your model is well-calibrated and ready to deliver accurate results.
# Understanding the Impact of Gamma in Neural Networks

In our exploration of neural networks, we've come to understand that the gamma variable plays a significant role in the output of the normalization process. This variable multiplicatively interacts with the output, and by changing the gamma, we can significantly alter the results of our network.

Let's consider an example where we initialize a network and train it. The activations, or the outputs of the neurons in the network, are going to look very good. This is because before every single 10H layer, there is a normalization process taking place. As a result, we can expect a standard deviation of roughly 0.65 percent throughout the entire layers, making everything look very homogeneous.

The gradients and the weights also look good in their distributions. The updates also look pretty reasonable, going above negative three a little bit, but not by too much. All the parameters are training at roughly the same rate.

However, what we've gained from this process is a network that is slightly less brittle with respect to the gain of these layers. For example, we can make the gain be 0.2, which is much slower than what we had with the 10H layer. The activations will actually be exactly unaffected due to the explicit normalization process. The gradients and weight gradients will look okay, but the updates will change.

Even though the forward and backward pass to a large extent look okay because of the backward pass of the batch normalization and how the scale of the incoming activations interacts in the batch normalization and its backward pass, this is actually changing the scale of the updates on these parameters. So, the gradients of these weights are affected.

We still don't get a completely free pass to pass arbitrary weights here, but everything else is significantly more robust in terms of the forward and backward pass and the weight gradients. It's just that you may have to retune your learning rate if you are changing the scale of the activations that are coming into the batch normalization.

For example, if we changed the gains of these linear layers to be greater, we would see that the updates are coming out lower as a result. Finally, if we are using batch normalization, we don't actually need to normalize by fan-in sometimes. If we take out the fan-in, these are just now random Gaussian.

Because of batch normalization, this will actually be relatively well. The forward pass looks good, the gradients look good, the weight updates look okay. A little bit of fat tails in some of the layers, and this looks okay as well. But as you can see, we're significantly below negative three, so we'd have to bump up the learning rate of this batch normalization so that we are training more properly.

In particular, looking at this, it roughly looks like we have to 10x the learning rate to get to about 20 negative three. So, we would change this to be an update of 1.0. Then we'll see that everything still of course looks good, and now we are roughly here and we expect this to be an okay training run.

In conclusion, we are significantly more robust to the gain of these linear layers, whether or not we have to apply batch normalization.
# Understanding Batch Normalization in Neural Networks

In this section, we aim to achieve three things. First, we want to introduce you to batch normalization, a modern innovation that has significantly stabilized the training of very deep neural networks. Second, we aim to simplify our code by wrapping it up into modules such as linear, BatchNorm1D, and Tanh. These layers or modules can be stacked up into neural networks like Lego building blocks. Lastly, we want to introduce you to the diagnostic tools that you can use to understand whether your neural network is in a good state dynamically.

## Batch Normalization

Batch normalization is one of the first modern innovations that we're looking into that has helped stabilize very deep neural networks and their training. It's important to understand how batch normalization works and how it can be used in a neural network.

## Simplifying Code with Modules

We've simplified our code by wrapping it up into modules like linear, BatchNorm1D, and Tanh. These are layers or modules that can be stacked up into neural networks like Lego building blocks. These layers actually exist in PyTorch. If you import torch, you can use PyTorch by prepending `nn.` to all these different layers. The API that we've developed here is identical to the API that PyTorch uses, and the implementation is also identical to the one in PyTorch.

## Diagnostic Tools for Neural Networks

We've introduced some diagnostic tools that you can use to understand whether your neural network is in a good state dynamically. We're looking at the statistics and histograms of the forward pass activations, the backward pass gradients, and the weights that are going to be updated as part of stochastic gradient descent. We're looking at their means, standard deviations, and also the ratio of gradients to data or even better, the updates to data.

Typically, we don't look at it as a single snapshot frozen in time at some particular iteration. People usually look at this over time, just like we've done here, and they look at these updated data ratios and make sure everything looks okay. In particular, -3 on the log scale is a good rough heuristic for what you want this ratio to be. If it's way too high, then probably the learning rate or the updates are a little too big. If it's way too small, then the learning rate is probably too small.

## Performance of Neural Networks

We did not aim to beat our previous performance by introducing the batch normalization layer. In fact, we found that the performance now is not bottlenecked by the optimization, which is what batch normalization is helping with. The performance at this stage is bottlenecked by what we suspect is the context length of our context. Currently, we are taking three characters to predict the fourth one, and we think we need to go beyond that. We need to look at more powerful architectures like recurrent neural networks and transformers in order to further push the log probabilities that we're obtaining.
# Understanding Neural Network Initialization and Back Propagation

In this lecture, we delved into the complexities of neural network initialization and back propagation. However, I must admit that I did not fully explore every aspect of these topics. For instance, I did not provide a comprehensive explanation of all the activations, gradients, backward paths, and the statistics of all these gradients. 

You might find some parts of the lecture slightly unintuitive or confusing. For example, you might wonder how changing the gain affects the need for a different learning rate. To fully understand these concepts, you would need to examine the backward pass of all the different layers and develop an intuitive understanding of how that works. Unfortunately, I did not delve into that in this lecture.

The primary purpose of this lecture was to introduce you to the diagnostic tools and give you a glimpse of what they look like. However, there is still a lot of work to be done on an intuitive level to understand initialization, the backward pass, and how all of these elements interact.

But don't feel too bad about any confusion you might have. We are currently at the cutting edge of the field. We haven't fully solved the problems of initialization and back propagation. These are still very much active areas of research. People are still trying to figure out the best way to initialize these networks and the most effective update rule to use.

None of these issues have been definitively solved, and we don't have all the answers to all these cases. However, we are making progress. We now have some tools to tell us whether or not things are on the right track.

In conclusion, I believe we've made positive progress in this lecture. I hope you found it enlightening and enjoyable.