---
layout: default
title: 5 - Building a WaveNet
parent: Andrej Karpathy
has_children: false
nav_order: 5
---

# Implementing a Character Level Language Model

Hello everyone! Today, we are continuing our implementation of our favorite character-level language model. You'll notice that the background behind me is different. That's because I am in Kyoto, and it is awesome. I'm currently in a hotel room here.

Over the last few lectures, we've built up to this architecture that is a multi-layer perceptron character-level language model. We see that it receives three previous characters and tries to predict the fourth character in a sequence using a very simple multi-perceptron using one hidden layer of neurons with 10ational neuralities.

What we'd like to do now in this lecture is to complexify this architecture. In particular, we would like to take more characters in a sequence as an input, not just three. In addition to that, we don't just want to feed them all into a single hidden layer because that squashes too much information too quickly. Instead, we would like to make a deeper model that progressively fuses this information to make its guess about the next character in a sequence.

As we make this architecture more complex, we're actually going to arrive at something that looks very much like a WaveNet. WaveNet is a paper published by DeepMind in 2016. It is also a language model, but it tries to predict audio sequences instead of character-level sequences or word-level sequences. Fundamentally, the modeling setup is identical. It is an autoregressive model and it tries to predict the next character in a sequence. The architecture actually takes this interesting hierarchical approach to predicting the next character in a sequence with a tree-like structure. This is the architecture we're going to implement in the course of this video.

The starter code for part five is very similar to where we ended up in part three. Recall that part four was the manual backpropagation exercise that is kind of an aside. So we are coming back to part three, copy-pasting chunks out of it, and that is our starter code for part five. I've changed very few things otherwise, so a lot of this should look familiar if you've gone through part three.

In particular, we are doing imports, reading our dataset of words, and processing the set of words into individual examples. None of this data generation code has changed. We have lots and lots of examples. In particular, we have 182,000 examples of three characters trying to predict the fourth one. We've broken up every one of these words into little problems of given three characters, predict the fourth one. This is our dataset and this is what we're trying to get the neural network to do.

In part three, we started to develop our code around these layer modules that are, for example, like class linear. We're doing this because we want to think of these modules as building blocks, like Lego bricks that we can stack up into neural networks and feed data between these layers and stack them up into a sort of graphs.

We also developed these layers to have APIs and signatures very similar to those that are found in PyTorch. So we have `torch.nn` and it's got all these layer building blocks that you would use in practice. We were developing all of these to mimic the APIs of these. For example, we have linear so there will also be a `torch.nn.Linear` and its signature will be very similar to our signature and the functionality will be also quite identical as far as I'm aware.

So we have the linear layer with the `Bass from 1D` layer and the `10h` layer that we developed previously. Linear just does a matrix multiply in the forward pass of this module batch.
# Understanding the Crazy Layer in Neural Networks

In the previous lecture, we developed a crazy layer in our neural network. What's crazy about it? Well, there are many things. 

Firstly, it has these running mean and variances that are trained outside of backpropagation. They are trained using an exponential moving average inside this layer when we call the forward pass. 

Secondly, there's this training flag because the behavior of batch normalization is different during train time and evaluation time. So, we have to be very careful that batch normalization is in its correct state. It's either in the evaluation state or training state. This is something to keep track of and it sometimes introduces bugs because you forget to put it into the right mode. 

Finally, batch normalization couples the statistics or the activations across the examples in the batch. Normally, we thought of the batch as just an efficiency thing, but now we are coupling the computation across batch elements. It's done for the purposes of controlling the activation statistics. 

This is a very weird layer that leads to a lot of bugs, partly because you have to modulate the training and evaluation phase. For example, you have to wait for the mean and the variance to settle and to actually reach a steady state. There's state in this layer and state is usually harmful. 

In the previous example, we had a generator object inside these layers. I've discarded that in favor of just initializing the torch RNG globally for simplicity. 

We are starting to build out some of the neural network elements. This should look very familiar. We have our embedding table C and then we have a list of layers. It's a linear feed to batch normalization, feed to tanh, and then a linear output layer. Its weights are scaled down so we are not confidently wrong at the initialization. 

We see that this is about 12,000 parameters. We're telling PyTorch that the parameters require gradients. The optimization is, as far as I'm aware, identical and should look very familiar. Nothing changed here. 

The loss function looks very jagged because 32 batch elements are too few. You can get very lucky or unlucky in any one of these batches and it creates a very jagged loss function. We're going to fix that soon. 

When we want to evaluate the trained neural network, we need to remember to set all the layers to be training equals false. This only matters for the batch normalization layer so far. 

Currently, we have a validation loss of 2.10 which is fairly good but there's still ways to go. But even at 2.10, we see that when we sample from the model we actually get relatively name-like results that do not exist in the training set. For example, Yvonne, Kilo Pros, Alaia, etc. They are certainly not unreasonable but not amazing. We can still push this validation loss even lower and get much better samples that are even more name-like. 

Let's improve this model. 

First, let's fix this graph because it is daggers in my eyes. The loss is a Python list of floats. We need to average up.
# PyTorch Tutorial: Tensor Operations and Layers

In this tutorial, we will explore some of the tensor operations in PyTorch and how to create layers for our neural network model. 

Firstly, let's consider a tensor of the first 10 numbers. This is currently a one-dimensional array. However, we can view this array as two-dimensional. For example, we can use it as a 2 by 5 array. This is now a 2D tensor, 2 by 5. The first row of this tensor is the first five elements and the second row is the second five elements. We can also view it as a 5 by 2 array.

In PyTorch, we can use negative one in place of one of these numbers, and PyTorch will calculate what that number must be in order to make the number of elements work out. This allows us to spread out some of the consecutive values into rows.

This is very helpful because we can create a PyTorch tensor out of a list of floats and then view it as whatever it is, but we're going to stretch it out into rows of 1000 consecutive elements. The shape of this now becomes 200 by 1000. Each row is one thousand consecutive elements in this list. 

We can then calculate the mean along the rows. The shape of this will just be 200. We've taken the mean on every row. This gives us a much nicer plot. We see that we made a lot of progress and then the learning rate decay subtracted a ton of energy out of the system and allowed us to settle into the local minimum in this optimization.

Next, let's consider our forward pass. Currently, it is a bit gnarly and takes too many lines of code. We see that we've organized some of the layers inside the layers list but not all of them. For no reason, we still have the embedding table as a special case outside of the layers. In addition to that, the viewing operation is also outside of our layers. 

Let's create layers for these and then we can add those layers to our list. In particular, we need to create layers for the embedding operation and the flattening operation. 

The embedding operation is simply an embedding table lookup done with indexing. The flattening operation rearranges the character embeddings and stretches them out into a row. This is effectively a concatenation operation, except it's free because viewing is very cheap in PyTorch. No memory is being copied, we're just re-representing how we view that tensor.

To save some time, I have already written the code for these modules. We have a module `Embedding` and a module `Flatten`. Both of them simply do the indexing operation in the forward pass and the flattening operation. The `Embedding` module will replace the special case of the embedding table outside of the layers list. The `Flatten` module will replace the viewing operation outside of the layers list. 

In conclusion, we have explored some tensor operations in PyTorch and how to create layers for our neural network model. We have also seen how to clean up our forward pass by organizing all the layers inside the layers list.
# PyTorch Embedding and Flatten

In this tutorial, we will be specifically embedding a platinum because it turns out that both of them actually exist in PyTorch. In PyTorch, we have `nn.Embedding` and it also takes the number of embeddings and the dimensionality of the embedding, just like we have here. But in addition, PyTorch takes in a lot of other keyword arguments that we are not using for our purposes yet. 

For `flatten`, that also exists in PyTorch and it also takes additional keyword arguments that we are not using. So we have a very simple platform, but both of them exist in PyTorch. They're just a bit more simpler and now that we have these, we can simply take out some of these special cased things. 

Instead of `C`, we're just going to have an embedding of a cup size and `N embed`. And then after the embedding, we are going to flatten. So let's construct those modules and now I can take out this `the`. Here, I don't have to special case anymore because now `C` is the embeddings weight and it's inside layers, so this should just work. 

Our forward pass simplifies substantially because we don't need to do these now outside of these layers. They're now inside layers, so we can delete those. But now to kick things off, we want this little `X` which in the beginning is just `XB`, the tensor of integers specifying the identities of these characters at the input. 

These characters can now directly feed into the first layer and this should just work. So let me insert a break here, just to make sure that the first iteration of this runs and then there's no mistake. That ran properly and basically, we substantially simplified the forward pass here. 

Now, one more thing that I would like to do in order to PyTorchify our code even further is that right now we are maintaining all of our modules in a naked list of layers. We can also simplify this because we can introduce the concept of PyTorch containers. 

In `torch.nn`, which we are basically rebuilding from scratch here, there's a concept of containers. These containers are basically a way of organizing layers into lists or dicts and so on. In particular, there's a `Sequential` which maintains a list of layers and is a module class in PyTorch. It basically just passes a given input through all the layers sequentially, exactly as we are doing here. 

So let's write our own `Sequential`. The code for `Sequential` is quite straightforward. We pass in a list of layers which we keep here and then given any input in a forward pass, we just call all the layers sequentially and return the result. In terms of the parameters, it's just all the parameters of the child modules. 

We can run this and we can again simplify this substantially because we don't maintain this naked list of layers. We now have a notion of a model which is a module and in particular is a `Sequential`. Now, parameters are simply just a model about parameters and so that list comprehension now lives here. 

Here, the code again simplifies substantially because we don't have to do this forwarding here. Instead, we just call the model on the input data and the input data here are the integers inside `xB`. So we can simply do `logits` which are the outputs of our model are simply the model called on `xB`.
# Understanding Cross Entropy and Neural Networks

The cross entropy here takes the logits and the targets, simplifying the process substantially. This looks good, so let's just make sure this runs. That looks good. Now, here we actually have some work to do still, but I'm going to come back later. For now, there's no more layers. There's a model that layers, but it's not a to access attributes of these classes directly, so we'll come back and fix this later.

Here, of course, this simplifies substantially as well because logits are the model called on x. These logits come here, so we can evaluate the train and validation loss, which currently is terrible because we just initialized the neural net. We can also sample from the model and this simplifies dramatically as well. We just want to call the model onto the context and outcome logits. These logits go into softmax and get the probabilities, etc. so we can sample from this model.

I fixed the issue and we now get the result that we expect, which is gibberish because the model is not trained because we re-initialized it from scratch. The problem was that when I fixed this cell to be modeled out layers instead of just layers, I did not actually run the cell and so our neural net was in a training mode. What caused the issue here is the batch norm layer, as batch norm layer likes to do. 

Batch norm was in a training mode and here, we are passing in an input which is a batch of just a single example made up of the context. If you are trying to pass in a single example into a batch norm that is in the training mode, you're going to end up estimating the variance using the input. The variance of a single number is not a number because it is a measure of a spread. For example, the variance of just the single number five is not a number. That's what happened in the master, basically causing an issue and then that polluted all of the further processing.

All that we had to do was make sure that this runs and we basically made the issue of again, we didn't actually see the issue with the loss. We could have evaluated the loss but we got the wrong result because batch norm was in the training mode. We still get a result, it's just the wrong result because it's using the sample statistics of the batch, whereas we want to use the running mean and running variants inside the batch norm. 

Again, this is an example of introducing a bug inline because we did not properly maintain the state of what is training or not. So, I rewrote everything and here's where we are. As a reminder, we have the training loss of 2.05 and validation 2.10. Now, because these losses are very similar to each other, we have a sense that we are not overfitting too much on this task and we can make additional progress in our performance by scaling up the size of the neural network and making everything bigger and deeper.

Currently, we are using this architecture here where we are taking in some number of characters going into a single hidden layer and then going to the prediction of the next character. The problem here is we don't have a naive way of making this bigger in a productive way. We could, of course, use our layers sort of building blocks and materials to introduce additional layers here and make the network deeper. But it is still the case that we are crushing all of the characters into a single layer all the way at the beginning. Even if we make this a bigger layer and add neurons, it's still kind of silly to squash all that information so fast in a single step.
# Implementing a Hierarchical Network for Text Generation

What we'd like to do instead is to design our network to look a lot more like the WaveNet model. In the WaveNet case, when we are trying to make the prediction for the next character in the sequence, it is a function of the previous characters that feed in. However, not all of these different characters are crushed to a single layer and then sandwiched. They are crushed slowly.

In particular, we take two characters and fuse them into a sort of diagram representation. We do that for all these characters consecutively. Then, we take the bigrams and fuse those into four-character level chunks. We then fuse that again. We do that in a tree-like hierarchical manner, fusing the information from the previous context slowly into the network as it gets deeper. This is the kind of architecture that we want to implement.

In the WaveNet case, this is a visualization of a stack of dilated causal convolution layers. This might sound very technical, but the idea is very simple. The fact that it's a dilated causal convolution layer is really just an implementation detail to make everything fast. We'll see that later. For now, let's just keep the basic idea of it, which is this Progressive Fusion. We want to make the network deeper and at each level, we want to fuse only two consecutive elements: two characters, then two bigrams, then two four-grams, and so on.

Let's implement this. First, let's change the block size from 3 to 8. We're going to be taking eight characters of context to predict the ninth character. The dataset now looks like this: we have a lot more context feeding in to predict any next character in a sequence. These eight characters are going to be processed in this tree-like structure.

If we redefine the network, you'll see the number of parameters has increased by 10,000. That's because the block size has grown. This first linear layer is much bigger. Our linear layer now takes eight characters into this middle layer, so there's a lot more parameters there. This runs just fine, but this network doesn't make too much sense. We're crushing way too much information way too fast.

Before we dive into the detail of the re-implementation, I was curious to see where we are in terms of the baseline performance of just lazily scaling up the context length. After letting it run, we get a nice loss curve. Evaluating the loss, we actually see quite a bit of improvement just from increasing the context line length.

Previously, we were getting a performance of 2.10 on the validation loss. Now, simply scaling up the contact length from 3 to 8 gives us a performance of 2.02. Quite a bit of an improvement here. Also, when you sample from the model, you see that the names are definitely improving qualitatively as well.

We could, of course, spend a lot of time here tuning things and making the network even bigger, even with the simple setup. But let's continue and implement our model and treat this as just a rough baseline performance. There's a lot of optimization left on the table in terms of some of the hyperparameters that you're hopefully getting a sense of.
# Debugging a Neural Network

Let's scroll up and take a look at what I've done here. I've created a bit of a scratch space for us to inspect the forward pass of the neural network and examine the shape of the tensor along the way.

For debugging purposes, I'm temporarily creating a batch of four examples, which are four random integers. I'm plucking out those rows from our training set and then passing them into the model as the input `xB`.

The shape of `xB` here is four by eight because we have only four examples. This eight is the current block size. Inspecting `xB`, we see that we have four examples, each one of them is a row of `xB`, and we have eight characters here. This integer tensor just contains the indices of the characters.

The first layer of our neural network is the embedding layer. Passing `xB`, this integer tensor, through the embedding layer creates an output that is four by eight by ten. Our embedding table has a 10-dimensional vector for each character that we are trying to learn. 

What the embedding layer does here is it plucks out the embedding vector for each one of these integers and organizes it all in a four by eight by ten tensor. All of these integers are translated into 10-dimensional vectors inside this three-dimensional tensor.

Passing that through the flattened layer, as you recall, views this tensor as just a 4 by 80 tensor. What that effectively does is that all these 10-dimensional embeddings for all these eight characters end up being stretched out into a long row. It looks kind of like a concatenation operation. By viewing the tensor differently, we now have a four by eighty tensor, and inside this 80, it's all the 10-dimensional vectors concatenated next to each other.

The linear layer then takes the 80 and creates 200 channels via matrix multiplication. 

Now, let's look at the insides of the linear layer and remind ourselves how it works. The linear layer, in the forward pass, takes the input `X`, multiplies it with a weight, and then optionally adds a bias. The weight here is two-dimensional, and the bias is one-dimensional. 

In terms of the shapes involved, what's happening inside this linear layer looks like this: a 4 by 80 input comes into the linear layer, that's multiplied by this 80 by 200 weight matrix inside, and there's a plus 200 bias. The shape of the whole thing that comes out of the linear layer is four by two hundred. 

Notice here that this operation will create a 4x200 tensor, and then plus 200, there's a broadcasting happening here. A 4 by 200 broadcasts with 200, so everything works here.

The surprising thing that you may not expect is that this input here that is being multiplied doesn't actually have to be two-dimensional. The matrix multiplication operator in PyTorch is quite powerful, and in fact, you can pass in higher-dimensional arrays or tensors, and everything works fine. 

For example, this could be four by five by eighty, and the result in that case will become four by five by two hundred. You can add as many dimensions as you like on the left here. Effectively, what's happening is that the matrix multiplication only works on the last dimension, and the dimensions before it in the input tensor are preserved.
# Understanding Matrix Multiplication in Neural Networks

In this article, we will discuss the concept of matrix multiplication in neural networks. Specifically, we will focus on how dimensions are treated in these calculations and how we can manipulate them to our advantage.

The dimensions on the left are all treated as a batch dimension. This means we can have multiple batch dimensions and perform matrix multiplication on the last dimension in parallel over all these dimensions. This is quite convenient because we can use this in our network.

Let's consider an example where we have eight characters coming in. We don't want to flatten all of it out into a large eight-dimensional vector because we don't want to perform matrix multiplication immediately. Instead, we want to group these characters. 

Every consecutive two elements, one and two, three and four, five and six, and seven and eight, should be flattened out and multiplied by a weight matrix. We'd like to process all of these four groups in parallel, introducing a kind of batch dimension. This allows us to process all of these bigram groups in the four batch dimensions of an individual example and also over the actual batch dimension of the four examples in our example.

Effectively, what we want is to take a 4 by 80 and multiply it by 80 by 200 in the linear layer. But instead, we want only two characters to come in on the very first layer and those two characters should be fused. In other words, we just want 20 numbers to come in. 

We don't want a 4 by 80 to feed into the linear layer, we want these groups of two to feed in. So instead of four by eighty, we want this to be a 4 by 4 by 20. These are the four groups of two and each one of them is a ten-dimensional vector.

We need to change the flattened layer so it doesn't output a four by eighty but it outputs a four by four by twenty. Every two consecutive characters are packed in on the very last dimension and these four is the first batch dimension and this four is the second batch dimension referring to the four groups inside every one of these examples.

This will just multiply like this. So we're going to have to change the linear layer in terms of how many inputs it expects. It shouldn't expect 80, it should just expect 20 numbers. We have to change our flattened layer so it doesn't just fully flatten out this entire example. It needs to create a 4x4 by 20 instead of four by eighty.

Currently, we have an input that is a four by eight by ten that feeds into the flattened layer. The flattened layer just stretches it out. It takes RX and it just views it as whatever the batch dimension is and then negative one. So effectively what it does right now is it does e dot view of 4 negative one and the shape of this of course is 4 by 80. 

We instead want this to be a four by four by twenty where these consecutive ten-dimensional vectors get concatenated. This is similar to how in Python you can take a list of range of 10, so we have numbers from zero to nine and we can index like this to get all the even parts.
# Indexing and Concatenating Tensors in PyTorch

We can index starting at one and going in steps up to two to get all the odd parts. One way to implement this would be as follows: we can take a tensor `e` and index into it for all the batch elements, and then just even elements in this dimension. So, at indexes 0, 2, 4, and 8, and then all the parts here from this last dimension. This gives us the even characters. 

This gives us all the odd characters and basically what we want to do is make sure that these get concatenated in PyTorch. We want to concatenate these two tensors along the second dimension. The shape of it would be four by four by twenty. This is definitely the result we want. We are explicitly grabbing the even parts and the odd parts and we're arranging those four by four by ten right next to each other and concatenate.

This works, but it turns out that what also works is you can simply use a view again and just request the right shape. It just so happens that in this case, those vectors will again end up being arranged in exactly the way we want. In particular, if we take `e` and we just view it as a four by four by twenty, which is what we want, we can check that this is exactly equal to the explicit concatenation.

So, the shape of the explicit concatenation is 4x4 by 20. If you just view it as 4x4 by 20, you can check that when you compare to explicit, you get a big element-wise operation, making sure that all of them are true. 

Long story short, we don't need to make an explicit call to concatenate. We can simply take this input tensor to flatten and we can just view it in whatever way we want. In particular, you don't want to stretch things out with negative one, we want to actually create a three-dimensional array. Depending on how many vectors that are consecutive we want to fuse, like for example two, then we can just simply ask for this dimension to be 20. 

Let's now go into flatten and implement this. We'd like to change it now, so let me create a constructor and take the number of elements that are consecutive that we would like to concatenate now in the last dimension of the output. Here we're just going to remember `self.n = n`.

I want to be careful here because PyTorch actually has a `torch.flatten` and its keyword arguments are different and they function differently. Our flatten is going to start to depart from PyTorch flatten, so let me call it `flatten_consecutive` just to make sure that our APIs are about equal. This function basically flattens only some `n` consecutive elements and puts them into the last dimension.

The shape of `X` is `B` by `T` by `C`, so let me pop those out into variables. Now, instead of doing `x.view(B, -1)`, we want this to be `B` by `-1` by `n`. Here we want `C` times `n`, that's how many consecutive elements we want. Instead of negative one, I don't super love the use of negative one because I like to be very explicit so that you get error messages when things don't match up.
# Implementing a Hierarchical Model

Sometimes, things don't go according to your expectations. So, what do we expect here? We expect this to become `t`, and divide `n` using integer division here. That's what I expect to happen. 

There's one more thing I want to do here. Remember previously, all the way in the beginning, `n` was three and we were basically concatenating all the three characters that existed there. So, we basically concatenated everything. 

Sometimes, this can create a spurious dimension of one. So, if it is the case that `x.shape` at one is one, then it's kind of like a spurious dimension. We don't want to return a three-dimensional tensor with a one here, we just want to return a two-dimensional tensor exactly as we did before. 

In this case, we will just say `x = x.squeeze()`. This is a PyTorch function. `squeeze()` takes a dimension that it either squeezes out all the dimensions of a tensor that are one, or you can specify the exact dimension that you want to be squeezed. I like to be as explicit as possible always, so I expect to squeeze out the first dimension only of this three-dimensional tensor. If this dimension here is one, then I just want to return `B` by `C` times `n`. 

So, `self.out` will be `X` and then we return `self.out`. That's the candidate implementation. Of course, this should be `self.n` instead of just `n`. 

Let's run this and take it for a spin. So, `flatten_consecutive` and in the beginning, let's just use eight. This should recover the previous behavior. So, `flatten_consecutive` of eight, which is the current block size, we can do this. That should recover the previous behavior. 

We should be able to run the model and here we can inspect. I have a little code snippet here where I iterate over all the layers. I print the name of this class and the shape. We see the shapes as we expect them after every single layer in the top bit. 

Now, let's try to restructure it using our `flatten_consecutive` and do it hierarchically. In particular, we want to `flatten_consecutive` not just block size but just two. Then we want to process this with `linear`. Now, the number of inputs to this `linear` will not be `n_embed` times block size, it will now only be `n_embed` times two. 

This goes through the first layer and now we can, in principle, just copy-paste this. The next `linear` layer should expect `n_hidden` times two and the last piece of it should expect `n_hidden` times two again. This is sort of like the naive version of it. 

Running this, we now have a much bigger model and we should be able to basically just forward the model. Now, we can inspect the numbers in between. 

So, `4` by `20` was `flatten_consecutive` into `4` by `4` by `20`. This was projected into `4` by `4` by `200` and then `BatchNorm` just worked out of the box. We have to verify that `BatchNorm` does the correct thing even though it takes a three-dimensional input that is two-dimensional. 

Then we have `Tanh` which is element-wise. Then we crushed it again. So, if we `flatten_consecutive`, we ended up with a `4` by `2` by `400` now.
# Understanding the Hierarchical Architecture of Neural Networks

In this article, we will delve into the hierarchical architecture of neural networks. We will start by examining the linear layer, which brings the dimension down to 200. This is followed by the batch room 10h, and finally, we get a 4 by 400. 

When we flatten this, we see that the last flatten operation squeezes out the dimension of one, leaving us with a four by four hundred. The linear batch then brings it down to 10h. The last linear layer gives us our logits. The logits end up in the same shape as they were before, but now we have a three-layer neural network. 

This network corresponds exactly to the previous network, except for this piece here because we only have three layers. In the previous example, there were four layers with a total receptive field size of 16 characters instead of just eight characters. The block size here is 16. This piece is basically implemented here. 

Now, we just have to figure out some good channel numbers to use. In particular, I changed the number of hidden units to be 68 in this architecture because when I use 68, the number of parameters comes out to be 22,000. This is exactly the same as we had before, and we have the same amount of capacity in this neural network in terms of the number of parameters. 

The question is whether we are utilizing those parameters in a more efficient architecture. To test this, I got rid of a lot of the debugging cells and reran the optimization. The result showed that we get identical performance. Our validation loss now is 2.029, and previously it was 2.027. So, controlling for the number of parameters, changing from the flat to hierarchical is not giving us anything yet. 

That said, there are two things to point out. First, we didn't really test the architecture very much. This is just my first guess, and there's a bunch of hyperparameters search that we could do in terms of how we allocate our budget of parameters to what layers. Second, we still may have a bug inside the batchnorm 1D layer. 

Let's take a look at that because it runs but does it do the right thing? I pulled up the layer inspector and printed out the shape along the way. Currently, it looks like the batchnorm is receiving an input that is 32 by 4 by 68. 

Now, this batchnorm assumed that X is two-dimensional, so it was n by D where n was the batch size. That's why we only reduced the mean and the variance over the zeroth dimension. But now, X will basically become three-dimensional. 

So, what's happening inside the batchnorm right now and how come it's working at all and not giving any errors? The reason for that is basically because everything broadcasts properly, but the batchnorm is not doing what we need it to do. 

In particular, let's think through what's happening inside the batchnorm. We're receiving an input of 32 by 4 by 68, and then we are doing the mean over zero, which is giving us 1 by 4 by 68. We're doing the mean only over the very first dimension, and it's giving us a mean and a variance that still maintain this dimension. 

These means are only taking over 32 numbers in the first dimension, and then when we perform this, everything broadcasts correctly. But basically, what ends up happening is the shape of it. I'm looking at the model that layers at three, which is the batchnorm.
# Understanding Batch Normalization in Deep Learning

In the first layer of the bathroom, we are looking at the running mean and its shape. The shape of this running mean is 1 by 4 by 68. Instead of it being just a size of dimension, we have 68 channels. We expect to have 68 means and variances that we're maintaining. However, we actually have an array of 4 by 68. 

This tells us that this batch normalization is currently working in parallel over 4 times 68 instead of just 68 channels. We are maintaining statistics for every one of these four positions individually and independently. What we want to do is treat this four as a batch dimension, just like the zeroth dimension. 

As far as the batch normalization is concerned, we don't want to average over 32 numbers. We want to now average over 32 times four numbers for every single one of these 68 channels. 

When you look at the documentation of `torch.mean`, one of its signatures specifies the dimension. The dimension here can be an integer or it can also be a tuple of integers. We can reduce over multiple dimensions at the same time. Instead of just reducing over zero, we can pass in a tuple (0,1). 

The output will be the same, but now we've reduced over both the zeroth and the first dimension. We're just getting 68 numbers and a bunch of spurious dimensions. This becomes 1 by 1 by 68 and the running mean and the running variance will become 1 by 1 by 68. 

We are only maintaining means and variances for 68 channels and we're not calculating the mean variance across 32 times 4 dimensions. That's exactly what we want. 

Let's change the implementation of batch normalization 1D that we have so that it can take in two-dimensional or three-dimensional inputs and perform accordingly. The fix is relatively straightforward. The dimension we want to reduce over is either 0 or the tuple (0,1) depending on the dimensionality of X. 

If `x.dim` is two, it's a two-dimensional tensor, then the dimension we want to reduce over is just the integer zero. If `x.dim` is three, it's a three-dimensional tensor, then the dimensions we're going to assume are zero and one that we want to reduce over. 

We're actually departing from the API of PyTorch here a little bit. When you come to batch normalization 1D in PyTorch, the input to this layer can either be n by C where n is the batch size and C is the number of features or channels. It does accept three-dimensional inputs but it expects it to be n by C by L where L is the sequence length. This is a problem because you see how C is.
# Improving Neural Network Performance

In this article, we will discuss how we improved the performance of our neural network from a starting point of 2.1 down to 1.9. However, it's important to note that we are still in the dark with respect to the correct setting of the hyperparameters and the learning rates. The experiments are starting to take longer to train, and we are missing an experimental harness on which we could run a number of experiments and really tune this architecture very well.

## The General Architecture

We have implemented a more general architecture in place, which is now set up to push the performance further by increasing the size of the network. For example, we bumped up the number of embeddings to 24 instead of 10 and also increased the number of hidden units. Using the exact same architecture, we now have 76,000 parameters. The training takes a lot longer, but we do get a nice curve. 

When we evaluate the performance, we are now getting validation performance of 1.993. So, we've crossed over the 2.0 territory and are right about 1.99. However, we are starting to have to wait quite a bit longer, and we're a little bit in the dark with respect to the correct setting of the hyperparameters and the learning rates.

## The Bug Fix

We also fixed a bug inside the batch normalization layer, which was holding us back a little bit. After the bug fix, we saw a slight improvement in the validation performance from 2.029 to 2.022. The reason we slightly expect an improvement is because we're not maintaining so many different means and variances that are only estimated using 32 numbers. Now, we are estimating them using 32 times 4 numbers. This allows things to be a bit more stable and less wiggly inside those estimates of those statistics.

## The Wavenet Paper

We implemented this architecture from the Wavenet paper, but we did not implement the specific forward pass of it where you have a more complicated linear layer. 

In conclusion, we have made some improvements in our neural network performance, but there is still a lot of work to be done. We need a more systematic approach to tuning the architecture and setting the hyperparameters. We also need to consider implementing more complex layers as suggested in the Wavenet paper.
# Gated Linear Layers and Convolutional Neural Networks

In our previous discussions, we've touched on the concept of gated linear layers, residual connections, and skip connections. However, we've only implemented the basic structure so far. In this article, we'll delve into how this structure relates to convolutional neural networks, as used in the WaveNet paper.

The use of convolutions in this context is strictly for efficiency. It doesn't actually change the model we've implemented. To illustrate this, let's look at a specific example.

Consider a name in our training set, "DeAndre". This name has seven letters, which translates to eight independent examples in our model. Each of these rows are independent examples of "DeAndre". 

We can forward any one of these rows independently. However, there's a slight trick here. The reason for this is that `extra[7]` is a one-dimensional array of eight. Therefore, we can't actually call the model on it as it would result in an error due to the lack of a batch dimension. 

To overcome this, when we do `extra[[7]]`, the shape becomes `1x8`, giving us an extra batch dimension of one. Then, we can forward the model. This forwards a single example. 

You might imagine that you may want to forward all of these eight examples at the same time. Pre-allocating some memory and then doing a for loop eight times, forwarding all of these eight examples, will give us all the logits in all these different cases. 

For us, with the model as we've implemented it right now, this is eight independent calls to our model. But what convolutions allow you to do is to slide this model efficiently over the input sequence. This for loop can be done not outside in Python, but inside of kernels in CUDA. This for loop gets hidden into the convolution.

The convolution is essentially a for loop applying a little linear filter over space of some input sequence. In our case, the space we're interested in is one-dimensional, and we're interested in sliding these filters over the input data.

This diagram is a good illustration of this concept. They are highlighting in black one single tree of this calculation, just calculating a single output example. This is what we've implemented. We've implemented a single structure and calculated a single output, a single example.

But what convolutions allow you to do is to take this structure and slide it over the input sequence, calculating all of these outputs at the same time. This corresponds to calculating all of these outputs at all the positions of "DeAndre" at the same time.

This is much more efficient for two reasons. Firstly, as mentioned, the for loop is inside the CUDA kernels in the sliding, which makes it efficient. Secondly, there's variable reuse. For example, a node is the right child of one node but is also the left child of another node. This node and its value are used twice. In a naive way, we'd have to recalculate it, but with convolutions, we are allowed to reuse it.
# Understanding Convolutional Neural Networks

In the convolutional neural network, we think of these linear layers as filters. We take these filters, which are linear, and slide them over the input sequence. We calculate the first layer, then the second layer, then the third layer, and finally the output layer of the sandwich. All of this is done very efficiently using convolutions. We will cover this in more detail in a future video.

The second thing I hope you took away from this video is that you've seen me implement all of these layer Lego building blocks or module building blocks. We've implemented a number of layers together and also implemented these containers. Overall, we've made our code much more PyTorch-friendly.

Essentially, what we're doing here is re-implementing `torch.nn`, which is the neural networks library on top of `torch.tensor`. It looks very much like this, except it is much better because it's in PyTorch instead of jingling my Jupyter notebook. Going forward, I will probably consider us having unlocked `torch.nn`. We understand roughly what's in there, how these modules work, how they're nested, and what they're doing on top of `torch.tensor`. Hopefully, we'll just switch over and start using `torch.nn` directly.

The next thing I hope you got a sense of is what the development process of building deep neural networks looks like. We spend a lot of time in the documentation page of PyTorch, reading through all the layers, looking at documentations, understanding the shapes of the inputs, what they can be, and what the layer does.

Unfortunately, I have to say the PyTorch's documentation is not very good. They spend a ton of time on hardcore engineering of all kinds of distributed primitives, etc., but as far as I can tell, no one is maintaining any documentation. It will lie to you, it will be wrong, it will be incomplete, it will be unclear. So unfortunately, it is what it is, and you just kind of do your best with what they've given us.

There's a ton of trying to make the shapes work and a lot of gymnastics around these multi-dimensional arrays. Are they two-dimensional, three-dimensional, four-dimensional? What layers take what shapes? Is it NCL or NLC? You're promoting and viewing, and it can get pretty messy.

I very often prototype these layers and implementations in Jupyter notebooks and make sure that all the shapes work out. I spend a lot of time babysitting the shapes and making sure everything is correct. Once I'm satisfied with the functionality in the Jupyter notebook, I take that code and copy-paste it into my repository of actual code that I'm training with. Then I'm working with VS Code on the side. I usually have Jupyter notebook and VS Code. I develop in Jupyter notebook, paste into VS Code, and then kick off experiments from the code repository.

This lecture unlocks a lot of potential further lectures because we have to convert our neural network to actually use these dilated causal convolutional layers, so implementing the ConvNet. Potentially starting to get into what this means, what are residual connections and skip connections, and why are they useful. As I mentioned, we don't have any experimental harness, so right now, we're just building the components.
# Deep Learning Workflows and Recurring Neural Networks

Now, I'm just guessing and checking everything. This is not representative of typical deep learning workflows. You have to set up your evaluation harness. You can kick off experiments. You have lots of arguments that your script can take. You're kicking off a lot of experimentation. You're looking at a lot of plots of training and validation losses, and you're looking at what is working and what is not working. You're working on this like population level and you're doing all these hyperparameter searches. So far, we've done none of that. 

Setting that up and making it good is a whole another topic. We should probably cover recurring neural networks (RNNs), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), and of course, Transformers. There are so many places to go and we'll cover that in the future. 

If you are interested, I think it is kind of interesting to try to beat this number 1.993. I really haven't tried a lot of experimentation here and there's quite a bit of fruit potentially to still purchase further. I haven't tried any other ways of allocating these channels in this neural net. Maybe the number of dimensions for the embedding is all wrong. Maybe it's possible to actually take the original network with just one hidden layer and make it big enough and actually beat my fancy hierarchical network. It's not obvious. That would be kind of embarrassing if this did not do better even once you torture it a little bit. 

Maybe you can read the Weight Net paper and try to figure out how some of these layers work and implement them yourselves using what we have. Of course, you can always tune some of the initialization or some of the optimization and see if you can improve it that way. I'd be curious if people can come up with some ways to beat this.