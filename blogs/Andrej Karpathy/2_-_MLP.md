---
layout: default
title: 2 - MLP
parent: Andrej Karpathy
has_children: false
nav_order: 2
---

# Implementing Makemore: A Multi-Layer Perceptron Model

Hello everyone! Today, we are continuing our implementation of Makemore. In the last lecture, we implemented the bigram language model using both counts and a super simple neural network with a single linear layer. 

We built out a Jupyter notebook in the last lecture, where we approached the problem by looking at only the single previous character and predicting the distribution for the character that would go next in the sequence. We did this by taking counts and normalizing them into probabilities, ensuring that each row sums to one. 

This approach works well if you only have one character of previous context. However, the problem with this model is that the predictions are not very good because you only take one character of context into account. As a result, the model didn't produce very name-like sounding things. 

The problem with this approach is that if we are to take more context into account when predicting the next character in a sequence, things quickly blow up. The size of this table grows exponentially with the length of the context. If we only take a single character at a time, that's 27 possibilities of context. But if we take two characters in the past and try to predict the third one, suddenly the number of rows in this matrix grows to 27 times 27, so there are 729 possibilities for what could have come in the context. If we take three characters as the context, suddenly we have 20,000 possibilities of context. There are just way too many rows in this matrix, way too few counts for each possibility, and the whole thing just kind of explodes and doesn't work very well. 

That's why today, we're going to move on to implement a multi-layer perceptron model to predict the next character in a sequence. This modeling approach that we're going to adopt follows the paper by Bengio et al., 2003. 

This paper isn't the very first one that proposed the use of multi-layer perceptrons or neural networks to predict the next character or token in a sequence, but it's definitely one that was very influential around that time. It is often cited to stand in for this idea, and it's a very nice write-up. 

The paper has 19 pages, so we don't have time to go into the full detail of this paper, but I invite you to read it. It's very readable, interesting, and has a lot of interesting ideas in it as well. In the introduction, they describe the exact same problem I just described. 

To address it, they propose the following model. Keep in mind that we are building a character-level language model, so we're working on the level of characters. In this paper, they have a vocabulary of 17,000 possible words and they instead build a word-level language model. But we're going to still stick with the characters and take the same modeling approach. 

What they do is basically propose to take every one of these words, all 17,000 words, and associate each word with a 30-dimensional feature vector. So, every word is now...
# Understanding Word Embeddings in Neural Networks

In this article, we will delve into the concept of word embeddings in neural networks. To start with, let's consider a vocabulary of 17,000 words. Each word is embedded into a thirty-dimensional space. You can think of it as having 17,000 points or vectors in a 30-dimensional space. 

You might imagine that this is very crowded - that's a lot of points for a very small space. In the beginning, these words are initialized completely randomly, so they're spread out at random. However, we're going to tune these embeddings of these words using back propagation. 

During the course of training of this neural network, these points or vectors are going to move around in this space. For example, words that have very similar meanings or that are indeed synonyms of each other might end up in a very similar part of the space. Conversely, words that mean very different things would go somewhere else in the space.

The modeling approach is identical to ours. They are using a multi-layer neural network to predict the next word given the previous words. To train the neural network, they are maximizing the log likelihood of the training data, just like we did. 

Let's consider a concrete example of this intuition. Suppose that you are trying to predict "A dog was running in a ____". Now suppose that the exact phrase "A dog was running in a ____" has never occurred in the training data. Here you are at test time, later when the model is deployed somewhere, and it's trying to make a sentence. Because it's never encountered this exact phrase in the training set, you're out of distribution. 

However, this approach allows you to get around that. Maybe you didn't see the exact phrase "A dog was running in a ____", but maybe you've seen similar phrases. Maybe you've seen the phrase "The dog was running in a ____". Your network may have learned that "a" and "the" are frequently interchangeable with each other. So, it took the embedding for "a" and the embedding for "the" and put them nearby each other in the space. This allows you to transfer knowledge through that embedding and generalize in that way. 

Similarly, the network could know that cats and dogs are animals and they co-occur in lots of very similar contexts. So, even though you haven't seen this exact phrase, or if you haven't seen exactly "walking" or "running", you can transfer knowledge through the embedding space and generalize to novel scenarios. 

Let's now look at the diagram of the neural network. In this example, we are taking three previous words and we are trying to predict the fourth word in a sequence. These three previous words are the index of the incoming word. Because there are 17,000 words, this is a lookup table that they call "c". This lookup table is a matrix that is 17,000 by 30. We're treating this as a lookup table, and so every index is a point in this space.
# Neural Network Embedding

The first step in our process is plucking out a row of the embedding matrix. This allows us to convert each index into a 30-dimensional vector that corresponds to the embedding vector for that word. For instance, if we have an input layer of 30 neurons for three words, we end up with 90 neurons in total. 

This matrix, denoted as 'C', is shared across all the words. This means we're always indexing into the same matrix 'C' over and over for each one of these words. 

## Hidden Layer

Next up is the hidden layer of this neural network. The size of this hidden neural layer is a hyperparameter, a design choice up to the designer of the neural net. This can be as large or as small as you'd like. For example, the size could be a hundred. We will go over multiple choices of the size of this hidden layer and evaluate how well they work.

Assuming there were 100 neurons here, all of them would be fully connected to the 90 words or 90 numbers that make up these three words. This is a fully connected layer. Then there's a tangent hyperbolic linearity, followed by an output layer. 

## Output Layer

Because there are 17,000 possible words that could come next, this layer has 17,000 neurons. All of them are fully connected to all of these neurons in the hidden layer. There are a lot of parameters here because there are a lot of words. Most computation is here, making this the expensive layer.

There are 17,000 logits here, so on top of there, we have the softmax layer. Every one of these logits is exponentiated, and then everything is normalized to sum to 1. This gives us a nice probability distribution for the next word in the sequence.

During training, we actually have the label. We have the identity of the next word in a sequence. That word, or its index, is used to pluck out the probability of that word. We then maximize the probability of that word with respect to the parameters of this neural net. 

The parameters are the weights and biases of this output layer, the weights and biases of the hidden layer, and the embedding lookup table 'C'. All of that is optimized using backpropagation. 

## Implementation

Now, let's implement it. I started a brand new notebook for this lecture. We are importing PyTorch and Matplotlib so we can create figures. Then, I am reading all the names into a list of words and showing the first eight. Keep in mind that we have 32,000 in total; these are just the first eight. 

Next, I'm building out the vocabulary of characters and all the mappings from the characters as strings to integers and vice versa. 

The first thing we want to do is compile the dataset for the neural network. I had to rewrite this code, and I'll briefly explain how this works. 

First, we're going to define something called block size. This is basically the context length of how many characters do we take to predict the next one.
# Building a Neural Network for Text Prediction

In this tutorial, we will be building a neural network that predicts the next character in a sequence based on the previous three characters. This is a simple example of a text prediction model, which can be used in a variety of applications such as autocomplete, spell check, and language translation.

## Step 1: Preparing the Data

The first step in building our model is to prepare the data. We will be using a block size of three, meaning that we will be taking three characters to predict the fourth one. 

To do this, we will build out the X and Y arrays. The X array will be the input to the neural network, and the Y array will be the labels for each example inside X. 

For efficiency, we will initially iterate over the first five words in our dataset. However, once we have developed all the code, we will erase this limitation so that we use the entire training set.

Let's take the word "Emma" as an example. From this single word, we can generate five examples. When we are given the context of just ".", the first character in a sequence is "E". In this context, the label is "M". When the context is ".E", the label is "M", and so forth. 

To build this out, we start with a padded context of just zero tokens. Then, we iterate over all the characters, get the character in the sequence, and build out the Y array of the current character and the X array, which stores the current running context. 

We can change the block size to four, five, or even ten, depending on how many characters we want to use to predict the next one. We always pad with dots to maintain consistency.

## Step 2: Building the Dataset

From these five words, we have created a dataset of 32 examples. Each input of the neural network is three integers, and we have a label that is also an integer. 

The X array looks like this:

```
[...]
```

And the Y array (the labels) looks like this:

```
[...]
```

## Step 3: Building the Neural Network

Now that we have our dataset, let's write a neural network that takes these Xs and predicts the Ys. 

First, let's build the embedding lookup table C. We have 27 possible characters, and we're going to embed them in a lower-dimensional space. In the paper we're referencing, they have 17,000 words and they embed them in spaces as small as 30 dimensions. In our case, we have only 27 possible characters, so let's embed them in a two-dimensional space.

This lookup table will be initialized with random numbers. We'll have 27 rows (one for each character) and two columns (for the two dimensions). 

To embed an individual integer, like say five, we can simply index into the fifth row of the lookup table C. This gives us a two-dimensional vector representing the character associated with the integer five.

And that's it! We have now built a simple neural network for text prediction. In the next steps, we would train this network on our dataset and then use it to predict the next character in a sequence.
# Embedding Integers in PyTorch

In this tutorial, we will explore two different ways to embed integers in PyTorch. The first method is the one we discussed in the previous lecture, and the second method, which seems different, is actually identical.

## Method 1: One-Hot Encoding

In the previous lecture, we took integers and used one-hot encoding to encode them. Let's say we want to encode the integer 5, and we want to specify that the number of classes is 27. This will result in a 26-dimensional vector of all zeros, except the fifth bit is turned on.

However, this method doesn't work as expected because the input must be a tensor, not an integer. This is a common error that can be fixed easily. After fixing this, we get a one-hot vector where the fifth dimension is one, and the shape of this is 27.

## Method 2: Matrix Multiplication

Now, let's take this one-hot vector and multiply it with a matrix. Initially, you might expect an error because PyTorch doesn't know how to multiply an integer with a float. The problem here is that the data type of the one-hot vector is long (a 64-bit integer), but the matrix is a float tensor. To fix this, we need to explicitly cast the one-hot vector to a float so that we can multiply it.

The output here is identical to the first method. This is because of the way matrix multiplication works. We have the one-hot vector multiplying columns of the matrix, and because of all the zeros, they end up masking out everything in the matrix except for the fifth row, which is plucked out.

## Interpretation

We can interpret the embedding of the integer in two ways. We can either think of it as the integer indexing into a lookup table, or equivalently, we can think of it as the first layer of a larger neural network. This layer has neurons that have no non-linearity (there's no tanh), they're just linear neurons, and their weight matrix is the matrix we used for multiplication.

We can encode integers into one-hot vectors and feed those into a neural network. This first layer basically embeds them. However, indexing is much faster, so we will discard the interpretation of one-hot inputs into neural networks and just index integers and use embedding tables.

## Embedding Multiple Integers

Embedding a single integer like 5 is easy enough; we can simply ask PyTorch to retrieve the fifth row of the matrix. But how do we simultaneously embed all of the integers stored in an array?

Luckily, PyTorch indexing is quite powerful and flexible. It doesn't just work to ask for a single element like this; we can actually index using lists. For example, we can get the rows five, six, and seven. It doesn't just have to be a list; it can also be a tensor of integers, and we can index with that.
# PyTorch Indexing and Embedding

In this tutorial, we will be working with an integer tensor, which we will denote as `567`. This tensor will work just as well as any other tensor in PyTorch. For instance, we can repeat row 7 and retrieve it multiple times. The same index will get embedded multiple times. 

In this example, we are indexing with a one-dimensional tensor of integers. However, PyTorch also allows us to index with multi-dimensional tensors of integers. Let's consider a two-dimensional integer tensor. We can simply index it as `c[x]` and it works perfectly. 

The shape of this tensor is 32 by 3, which is the original shape. Now, for every one of those 32 by 3 integers, we've retrieved the embedding vector. As an example, let's consider the 13th index. The second dimension is the integer 1. If we do `c[x]`, which gives us that array, and then we index into 13 by 2 of that array, we get the embedding. You can verify that `c[1]`, which is the integer at that location, is indeed equal to this. 

In summary, PyTorch indexing is awesome and to embed all of the integers in `x` simultaneously, we can simply do `c[x]`. That is our embedding and it just works.

## Constructing the Hidden Layer

Now, let's construct the hidden layer. We have the weights, denoted as `w1`, which we will initialize randomly. The number of inputs to this layer is going to be six because we have two-dimensional embeddings and we have three of them. The number of neurons in this layer is a variable up to us. Let's use 100 neurons as an example. The biases will also be initialized randomly and we just need 100 of them.

However, we can't simply take the input, in this case, the embedding, multiply it with these weights, and then add the bias. This is roughly what we want to do, but the problem here is that these embeddings are stacked up in the dimensions of this input tensor. This matrix multiplication will not work because this is a shape 32 by 3 by 2 and we can't multiply that by 6 by 100. 

We need to concatenate these inputs together so that we can perform a matrix multiplication. So, how do we transform this 32 by 3 by 2 into a 32 by 6 so that we can actually perform this multiplication? 

There are usually many ways of implementing what you'd like to do in PyTorch. Some of them will be faster, better, shorter, etc. This is because PyTorch is a very large library and it's got lots and lots of functions. If you just go to the documentation and click on torch, you'll see that there are so many functions that you can call on these tensors to transform them, create them, multiply them, add them, and perform all kinds of different operations on them.
# Exploring the Space of Possibility with PyTorch

One of the things you can do with PyTorch is to manipulate tensors in various ways. For instance, you can use the `torch.cat` function, short for concatenate, to concatenate a given sequence of tensors in a specific dimension. However, these tensors must have the same shape. 

Let's consider an example where we have three embeddings for each input. We want to concatenate these three parts. To do this, we first need to retrieve these three parts. We can do this by indexing the tensor. For instance, if we want to retrieve the embeddings of the first word, we can index the tensor as follows:

```python
m[0, :, :]
```

This will give us the 32 by 2 embeddings of the first word. We can do the same for the second and third words. Once we have these three pieces, we can treat them as a sequence and use `torch.cat` to concatenate them:

```python
torch.cat([m[0, :, :], m[1, :, :], m[2, :, :]], dim=1)
```

This will give us a tensor of shape 32 by 6, which is exactly what we want. 

However, this approach is not ideal because it does not generalize well. If we want to change the block size, we would have to change the code because we are directly indexing the tensor. 

Fortunately, PyTorch provides a function called `unbind` that can help us. The `unbind` function removes a tensor dimension and returns a tuple of all slices along a given dimension. We can use this function to get a list of tensors:

```python
torch.unbind(m, dim=1)
```

This will give us a list of tensors equivalent to the one we got by indexing. We can then call `torch.cat` on this list to concatenate the tensors:

```python
torch.cat(torch.unbind(m, dim=1), dim=1)
```

This will give us the same result as before, but now the code will work regardless of the block size. 

However, there is an even better and more efficient way to do this. PyTorch allows us to quickly re-represent a tensor as a different sized and dimensional tensor using the `view` function. For instance, we can create a tensor with elements from 0 to 17:

```python
a = torch.arange(18)
```

This will give us a tensor of shape 18. We can then use the `view` function to reshape this tensor:

```python
a.view(2, 9)
```

This will give us a tensor of shape 2 by 9. We can also reshape it into a tensor of a different shape:

```python
a.view(3, 6)
```

This will give us a tensor of shape 3 by 6. The `view` function provides a flexible and efficient way to manipulate tensors in PyTorch.
# Understanding PyTorch Tensors

In this tutorial, we will be discussing the concept of tensors in PyTorch, specifically focusing on the `view` operation and how it works. 

Let's start with a simple example. Suppose we have a tensor, `m`, with a shape of 32 by 3 by 2. This tensor can also be represented as a nine by two tensor or a three by three by two tensor. As long as the total number of elements multiply to be the same, this operation will work. 

In PyTorch, this operation is called `view` and it is extremely efficient. The reason for its efficiency lies in the underlying storage of each tensor. The storage is just the numbers, always represented as a one-dimensional vector. This is how the tensor is represented in the computer memory. 

When we call `view`, we are manipulating some attributes of the tensor that dictate how this one-dimensional sequence is interpreted to be an n-dimensional tensor. No memory is being changed, copied, moved, or created when we call `view`. The storage is identical, but some of the internal attributes of the tensor are being manipulated and changed. 

In particular, there's something called a storage offset, strides, and shapes. These are manipulated so that this one-dimensional sequence of bytes is seen as different n-dimensional arrays. 

There's a blog post from Eric called "PyTorch Internals" where he goes into some of this with respect to tensor and how the view of the tensor is represented. This is really just like a logical construct of representing the physical memory. 

Now, let's get back to our example. We see that the shape of our tensor `m` is 32 by 3 by 2, but we can simply ask PyTorch to view this instead as a 32 by 6. The way this gets flattened into a 32 by 6 array just happens that these two get stacked up in a single row. This is basically the concatenation operation that we're after. 

You can verify that this actually gives the exact same result as what we had before. So, if this is `h`, then `h` shape is now the 100-dimensional activations for every one of our 32 examples. This gives the desired result. 

To make this operation more flexible, we can replace the hard-coded number 32 with `m.shape[0]`. This will work for any size of `m`. Alternatively, we can also use `-1`. When we use `-1`, PyTorch will infer what this should be because the number of elements must be the same and we're saying that this is 6. PyTorch will derive that this must be 32 or whatever else it is if `m` is of a different size. 

One more thing to note is that when we do the concatenation, this operation is much less efficient because this concatenation would require additional memory. 

In conclusion, understanding how tensors work in PyTorch and how to manipulate them efficiently is crucial for working with neural networks. The `view` operation is a powerful tool that allows us to reshape our tensors without changing the underlying data.
# Creating a New Tensor with New Storage

In this tutorial, we will create a new tensor with new storage. This process involves creating new memory because there's no way to concatenate tensors just by manipulating the view attributes. This method can be inefficient as it creates all kinds of new memory.

Let's start by deleting any unnecessary data. We don't need to keep everything, especially if it's not going to be used in our calculations.

Next, we will calculate `h` and also apply the `tanh` function to it. These are now numbers between negative one and one because of the `tanh` function. We can observe that the shape is 32 by 100, which represents this hidden layer of activations for every one of our 32 examples.

There's one more thing to be cautious about, and that's the addition operation. We want to ensure that the broadcasting will do what we expect. The shape of our tensor is 32 by 100 and the ones shape is 100. The addition here will broadcast these two, and in particular, we have 32 by 100 broadcasting to 100. Broadcasting will align on the right, create a fake dimension here so this will become a 1 by 100 row vector, and then it will copy vertically for every one of these rows of 32 and do an element-wise addition.

In this case, the correct thing will be happening because the same bias vector will be added to all the rows of this matrix. It's always good practice to make sure so that you don't make a mistake.

Finally, let's create the final layer. We'll create `w2` and `v2`. The input now is 100, and the output number of neurons will be 27 because we have 27 possible characters that come next. The biases will be 27 as well.

Therefore, the logits, which are the outputs of this neural net, are going to be `h` multiplied by `w2` plus `b2`. The shape of the logits is 32 by 27.

Just as we saw in the previous video, we want to take these logits and first exponentiate them to get our fake counts, and then we want to normalize them into a probability. The shape of the probability is 32 by 27, and you'll see that every row of the probability sums to one, so it's normalized.

We have the actual letter that comes next from the array `y`, which we created during the dataset creation. `y` is the identity of the next character in the sequence that we'd like to now predict.

What we'd like to do now is index into the rows of the probability and in each row, we'd like to pluck out the probability assigned to the correct character. We can do this by using `torch.range` of 32, which iterates over numbers from 0 to 31, and then we can index into the probability in the following way: `prob[torch.range(32), y]`. This will iterate the rows and in each row, we'd like to grab the probability of the correct character.
# Understanding Neural Networks

In this article, we will be discussing the current probabilities as assigned by a neural network with a specific setting of its weights to the correct character in the sequence. 

As you can see, the probabilities look okay for some characters, like 0.2, but not very good at all for many other characters, like 0.0701 probability. The network thinks that some of these are extremely unlikely. However, we haven't trained the neural network yet, so this will improve. Ideally, all of these numbers should be one because then we are correctly predicting the next character.

Just as in the previous video, we want to take these probabilities, look at the log probability, and then look at the average probability. We then take the negative of it to create the negative log likelihood loss. The loss here is 17, and this is the loss that we'd like to minimize to get the network to predict the correct character in the sequence.

I have rewritten everything here and made it a bit more respectable. Our dataset and all the parameters that we defined are now using a generator to make it reproducible. I clustered all the parameters into a single list of parameters so that it's easy to count them and see that in total we currently have about 3400 parameters. This is the forward pass as we developed it, and we arrive at a single number here, the loss that is currently expressing how well this neural network works with the current setting of parameters.

Now, I would like to make it even more respectable. In particular, see these lines here where we take the logits and we calculate the loss. We're not actually reinventing the wheel here. This is just classification, and many people use classification. That's why there is a `functional.cross_entropy` function in PyTorch to calculate this much more efficiently. 

We can just simply call `f.cross_entropy`, and we can pass in the logits and the array of targets `y`. So in fact, we can simply put this here and erase these three lines, and we're going to get the exact same result.

There are actually many good reasons to prefer `f.cross_entropy` over rolling your own implementation. I did this for educational reasons, but you'd never use this in practice. Why is that?

1. When you use `f.cross_entropy`, PyTorch will not actually create all these intermediate tensors because these are all new tensors in memory, and all this is fairly inefficient to run like this. Instead, PyTorch will cluster up all these operations and very often create fused kernels that very efficiently evaluate these expressions that are sort of like clustered mathematical operations.

2. The backward pass can be made much more efficient, and not just because it's a fused kernel but also analytically and mathematically. It's often a very much simpler backward pass to implement. 

We actually saw this with micrograd. You see here when we implemented `tanh`, the forward pass of this operation to calculate the `tanh` was actually a fairly complicated mathematical expression. But because it's a clustered mathematical expression, when we did the backward pass, we didn't individually backward through the `x` and the `2` times and the `-1` in division, etc. We just said it's `1 - t^2`, and that's a much simpler mathematical expression.
# Understanding the Efficiency of Cross-Entropy in Neural Networks

In our exploration of neural networks, we've discovered that we can significantly improve efficiency by reusing calculations and mathematically deriving the derivative. Often, this expression simplifies mathematically, making it much easier to implement. 

Not only can it be made more efficient because it runs in a fused kernel, but also because the expressions can take a much simpler form mathematically. 

## Numerical Stability in Cross-Entropy

Under the hood, the cross-entropy function can also be significantly more numerically well-behaved. Let's illustrate this with an example. Suppose we have logits of -2, 3, -3, 0, and 5. We then take the exponent of these logits and normalize them to sum to 1. When logits take on these values, everything works well, and we get a nice probability distribution.

However, consider what happens when some of these logits take on more extreme values, which can happen during the optimization of the neural network. Suppose that some of these numbers grow very negative, like -100. In this case, everything will come out fine. We still get the probabilities that are well-behaved, and they sum to one. 

But, if you have very positive logits, let's say positive 100, you start to run into trouble, and we get a 'not a number' error. The reason for this is that these counts have an 'if' condition. If you pass in a very negative number to the exponent, you get a very small number, very near zero, and that's fine. But if you pass in a very positive number, we run out of range in our floating-point number that represents these counts. 

## PyTorch's Solution

PyTorch has a solution for this. It turns out that because of the normalization, you can offset logits by any arbitrary constant value, and you will get the exact same probabilities. So, because negative numbers are okay but positive numbers can overflow the exponent, what PyTorch does is it internally calculates the maximum value that occurs in the logits and subtracts it. Therefore, the greatest number in logits will become zero, and all the other numbers will become some negative numbers. The result of this is always well-behaved. 

So, even if we have 100 here, which was previously not good, because PyTorch will subtract 100, this will work. 

In conclusion, there are many good reasons to use the cross-entropy function. The forward pass can be much more efficient, the backward pass can be much more efficient, and things can be much more numerically well-behaved. 

Now, let's set up the training of this neural network. We don't need these losses equal to the...
# Implementing a Neural Network with PyTorch

In this tutorial, we will be implementing a neural network using PyTorch. We will start with the forward pass using cross entropy, then move on to the backward pass. 

First, we need to set the gradients to zero. This can be done using the following code:

```python
for p in parameters:
    p.grad = None
```

This is equivalent to setting it to zero in PyTorch. 

Next, we use `loss.backward()` to populate the gradients. Once we have the gradients, we can update the parameters. This can be done with the following code:

```python
for p in parameters:
    p.data -= p.grad
```

This code takes all the data and nudges it. We then repeat this process. However, this alone will not suffice and will create an error. We also need to ensure that `p.requires_grad` is set to `True` in PyTorch. 

```python
for p in parameters:
    p.requires_grad = True
```

With this, our code should work. 

We start off with a loss of 17 and we aim to decrease it. If we run this for a thousand times, we get a very low loss, which means that we're making very good predictions. 

However, this is straightforward because we're only overfitting 32 examples. We have 3,400 parameters and only 32 examples, so we're overfitting a single batch of the data. This results in a very low loss and good predictions. 

We're not able to achieve exactly zero loss because there are cases where multiple outcomes are possible for the same input in our training set. Therefore, we're not able to completely overfit and make the loss be exactly zero. But we're getting very close in the cases where there's a unique input for a unique output. In those cases, we do overfit and get the exact correct result. 

Now, all we have to do is read in the full dataset and optimize the neural network. 

```python
# Erase the first five words limitation and the print statements
```

When we process the full dataset of all the words, we now have 228,000 examples instead of just 32. 

```python
# Reinitialize the weights
# Ensure all parameters require gradients
```

And that's it! We have successfully implemented a neural network using PyTorch.
# Optimizing Neural Networks with Mini-Batches

In this tutorial, we will discuss how to optimize neural networks using mini-batches. We will start with a high loss and then optimize it. However, you'll notice that it takes quite a bit of time for every single iteration. This is because we're doing way too much work forwarding and backwarding 220,000 examples. 

In practice, what people usually do is perform a forward and backward pass and update on many batches of the data. We want to randomly select some portion of the data set, which is a mini-batch, and then only forward, backward, and update on that little mini-batch. We then iterate on those mini-batches.

In PyTorch, we can use `storage.randint` to generate numbers between 0 and 5. The size has to be a tuple in PyTorch, so we can have a tuple of 32 numbers between zero and five. We want `x.shape[0]` here, which creates integers that index into our dataset, and there are 32 of them. 

If our mini-batch size is 32, then we can construct a mini-batch. The integers that we want to optimize in this single iteration are in the `ix`. We then want to index into `x` with `ix` to only grab those rows. We're only getting 32 rows of `x`, and therefore embeddings will again be 32 by 3 by 2, not 200,000 by 3 by 2. This `ix` has to be used not just to index into `x`, but also to index into `y`. 

Now, this should be many batches and this should be much faster. This way, we can run many examples nearly instantly and decrease the loss much faster. However, because we're only dealing with mini-batches, the quality of our gradient is lower, so the direction is not as reliable. It's not the actual gradient direction, but the gradient direction is good enough even when it's estimated on only 32 examples. It is useful and so it's much better to have an approximate gradient and just make more steps than it is to evaluate the exact gradient and take fewer steps. That's why in practice, this works quite well.

We're hovering around 2.5 or so, however, this is only the loss for that mini-batch. Let's actually evaluate the loss for all of `x` and for all of `y` just so we have a full sense of exactly how the model is doing right now. Right now, we're at about 2.7 on the entire training set. Let's run the optimization for a while. 

One issue, of course, is we don't know if we're stepping too slow or too fast. This point one I just guessed it. So one question is how do you determine this learning rate? And how do we gain confidence that we're stepping at the right speed? I'll show you one way to determine a reasonable learning rate. It works as follows: let's reset our parameters to the initial settings.
# Finding the Optimal Learning Rate

Let's start by printing every step. We'll only do about 10 steps, or maybe even 100 steps. The goal is to find a reasonable set search range. For example, if the learning rate is very low, we'll see that the loss is barely decreasing. That's too low. 

Let's try a different learning rate. We're decreasing the loss, but not very quickly. That's a pretty good low range. Now, let's reset it again and try to find the place at which the loss kind of explodes. 

We see that we're minimizing the loss, but it's kind of unstable. It goes up and down quite a bit. So, a learning rate of -1 is probably too fast. Let's try -10. This isn't optimizing well, so -10 is way too big. Even -1 was already kind of big. 

Therefore, -1 was somewhat reasonable. If I reset, I'm thinking that the right learning rate is somewhere between -0.001 and -1. 

We can use `torch.linspace` to do this. We want to do something like this between 0 and 1, but we need to specify the number of steps. Let's do a thousand steps. This creates 1000 numbers between 0.01 and 1. 

However, it doesn't really make sense to step between these linearly. Instead, let's create a learning rate exponent. Instead of 0.001, this will be -3 and this will be 0. The actual learning rates that we want to search over are going to be 10 to the power of the learning rate exponent. 

Now, we're stepping linearly between the exponents of these learning rates. This is 0.001 and this is 1 because 10 to the power of 0 is 1. Therefore, we are spaced exponentially in this interval. These are the candidate learning rates that we want to search over. 

Next, we're going to run the optimization for 1000 steps. Instead of using a fixed number, we're going to use a learning rate that's in the beginning very low. In the beginning, it's going to be 0.001, but by the end, it's going to be 1. Then, we're going to step with that learning rate. 

We want to track the learning rates that we used and look at the losses that resulted. Let's track these stats. We'll append the learning rate and the loss to their respective lists. 

To summarize, we reset everything and then run the optimization. We track the learning rates and the losses. This allows us to find the optimal learning rate for our model.
# Finding the Optimal Learning Rate for Neural Networks

In this article, we will discuss how to find the optimal learning rate for training neural networks. We will start with a very low learning rate and gradually increase it to a learning rate of negative one. 

To visualize this process, we can plot the learning rates on the x-axis and the losses on the y-axis. Often, you will find that your plot looks something like this: in the beginning, you had very low learning rates, so barely anything happened. Then we got to a nice spot, and as we increased the learning rate enough, we started to be kind of unstable. 

A good learning rate turns out to be somewhere around here. Because we have `lri` here, we may want to log not the learning rate but the exponent, so that would be `lre` at `i`. Let's reset this and redo that calculation.

Now, on the x-axis, we have the exponent of the learning rate. We can see that the exponent of the learning rate that is good to use would be roughly in the valley here. Here, the learning rates are just way too low, and then here, where we expect relatively good learning rates, somewhere here, and then here, things are starting to explode. 

Somewhere around negative one x the exponent of the learning rate is a pretty good setting, and 10 to the negative one is 0.1. So, 0.1 is actually a fairly good learning rate around here, and that's what we had in the initial setting. 

That's roughly how you would determine it. Now, we can take out the tracking of these and simply set `lr` to be 10 to the negative one or, basically, otherwise 0.1 as it was before. Now we have some confidence that this is actually a fairly good learning rate.

Now, we can crank up the iterations, reset our optimization, and run for a pretty long time using this learning rate. 

Once we're at the late stages of training, we may want to go a bit slower. We can do this by implementing a learning rate decay, where we take our learning rate and lower it by a factor of 10. 

We continue this process until we get a sense that the loss is starting to plateau off. At this point, we can do a few more steps of learning rate decay to further optimize the network. 

In this example, we achieved a loss of about 2.3. Obviously, this is not exactly how you would train it in production, but this is roughly what you're going through. 

You first find a decent learning rate using the approach that I showed you. Then you start with that learning rate and train for a while. At the end, you implement a learning rate decay and do a few more steps. This process will give you a trained network, roughly speaking.
# Improving the Bi-gram Language Model with a Simple Neural Net

We've achieved a loss of 2.3, dramatically improving on the bi-gram language model using a simple neural net as described here. This was achieved using 3,400 parameters. However, there's something we have to be careful with. 

I mentioned that we have a better model because we are achieving a lower loss, 2.3, which is much lower than the 2.45 we had with the bi-gram model previously. But that's not exactly true. This is actually a fairly small model, but these models can get larger and larger if you keep adding neurons and parameters. 

You can imagine that we don't potentially have a thousand parameters, we could have 10,000 or 100,000 or even millions of parameters. As the capacity of the neural network grows, it becomes more and more capable of overfitting your training set. 

What that means is that the loss on the training set, on the data that you're training on, will become very, very low, as low as zero. But all that the model is doing is memorizing your training set verbatim. So, if you take that model and it looks like it's working really well, but you try to sample from it, you will basically only get examples exactly as they are in the training set. You won't get any new data. 

In addition to that, if you try to evaluate the loss on some withheld names or other words, you will actually see that the loss on those can be very high. So basically, it's not a good model. 

The standard in the field is to split up your data set into three splits, as we call them. We have the training split, the dev split or the validation split, and the test split. Typically, this would be say 80% of your data set for training, 10% for validation, and 10% for testing. 

These 80% of the data set, the training set, is used to optimize the parameters of the model, just like we're doing here using gradient descent. The 10% of the examples, the dev or validation split, they're used for development over all the hyperparameters of your model. 

Hyperparameters are, for example, the size of this hidden layer, the size of the embedding (which is a hundred or two for us, but we could try different things), the strength of the regularization (which we aren't using yet so far). So there's lots of different hyperparameters and settings that go into defining your neural net, and you can try many different variations of them and see whichever one works best on your validation split. 

The test split is used to evaluate the performance of the model at the end. We're only evaluating the loss on the test split very sparingly and very few times because every single time you evaluate your test loss and you learn something from it, you are basically starting to also train on the test split. 

So you are only allowed to test the loss on a test set very few times, otherwise, you risk overfitting to it as well as you experiment on your model. 

Let's also split up our training data into train, dev, and test. Then we are going to train on train, and only evaluate on tests very sparingly. 

Here is where we took all the words and put them into X and Y tensors.
# Building a Neural Network: A Step-by-Step Guide

In this tutorial, we will be building a neural network from scratch. Let's start by creating a new cell and copying some code into it. This code might seem complex at first, but we will break it down to make it easier to understand.

The function we are creating takes a list of words as input and builds two arrays, X and Y, based on these words. We then shuffle all the words randomly. This is done to ensure that our model doesn't learn any inherent order in the data and can generalize better to unseen data.

We then set `n1` to be 80% of the total number of words and `n2` to be 90% of the total number of words. For instance, if the length of words is 32,000, `n1` would be 25,000 and `n2` would be 28,000.

Next, we call the `build_dataset` function to build the training set X and Y. We do this by indexing up to `n1`, so we only have 25,000 training words. The difference between `n2` and `n1` gives us around 3,000 validation examples. The remaining words, i.e., the length of words minus `n2`, gives us around 3,204 examples for the test set.

Now that we have our X's and Y's, let's take a look at their size. These are not words anymore, but individual examples made from those words.

When we train our model, we will only be using `x_train` and `y_train`. Training neural networks can take a while. Usually, you don't do it inline. Instead, you launch a bunch of jobs and wait for them to finish, which can take multiple days.

After training, we evaluate the model using the validation set (`x_dev` and `y_dev`) to calculate the loss. We then decay the learning rate and evaluate the dev loss again. We are getting about 2.3 on dev, which is pretty decent considering that the neural network did not see these dev examples during training.

We can also calculate the loss on the entire training set. We find that the training and dev loss are about equal, indicating that we are not overfitting. This means that our model is not powerful enough to memorize the data. We are underfitting, which typically means that our network is very small.

To improve performance, we can scale up the size of the neural network. The easiest way to do this is to increase the size of the hidden layer, which currently has 100 neurons. Let's bump this up to 300 neurons and see how our model performs.
# Initializing and Optimizing a Neural Network

In this tutorial, we will initialize a neural network with 10,000 parameters instead of the usual 3,000. This is achieved by having 300 biases and 300 inputs into the final layer. 

```python
# Initialize the neural network
neural_net = NeuralNet(10000)
```

We will not be using the previous network for this tutorial. Instead, we will keep track of the loss and steps during the training process. 

```python
# Keep track of stats
stats = {'loss': [], 'steps': []}
```

We will train the network on 30,000 steps with a learning rate of 0.1. 

```python
# Train the network
neural_net.train(30000, learning_rate=0.1)
```

Once the network is trained, we can plot the steps and loss function to visualize how the network is being optimized. 

```python
# Plot the steps and loss function
plt.plot(stats['steps'], stats['loss'])
```

You will notice that there is quite a bit of thickness to the plot. This is because we are optimizing over mini-batches, which create a bit of noise. 

At this point, we are at a loss of 2.5, which means we haven't optimized the neural network very well. This could be because we made the network bigger, and it might take longer for this network to converge. 

One possibility is that the batch size is so low that we have too much noise in the training. We may want to increase the batch size to have a more accurate gradient and reduce thrashing. 

```python
# Increase the batch size
neural_net.batch_size = 500
```

After reinitializing, the plot may not look pleasing, but there is likely a tiny improvement. 

We can try to decrease the learning rate to improve the loss. 

```python
# Decrease the learning rate
neural_net.learning_rate = 0.05
```

We expect to see a lower loss than before because we have a much bigger model and we were underfitting. So, increasing the size of the model should help the neural network. 

However, one concern is that even though we've made the hidden layer much bigger, the bottleneck of the network could be the embeddings that are two-dimensional. We might be cramming too many characters into just two dimensions, and the neural network may not be able to use that space effectively. This could be the bottleneck to our network's performance. 

To potentially eliminate this bottleneck, we can increase the embedding size from two. But before we do that, we should visualize the embedding vectors for these characters. 

```python
# Visualize the embedding vectors
neural_net.visualize_embeddings()
```

Once we increase the embedding size, we won't be able to visualize them. So, it's important to do this before scaling up. 

In the next tutorial, we will explore how to increase the embedding size and further optimize the neural network.
# Visualizing Character Embeddings in Neural Networks

We're currently at 2.23 and 2.24, so we're not improving much more. The bottleneck now might be the character embedding size, which is two. 

I have a bunch of code that will create a figure. We're going to visualize the embeddings that were trained by the neural net on these characters. Since the embedding has just two, we can visualize all the characters with the x and y coordinates as the two embedding locations for each of these characters. 

Here are the x coordinates and the y coordinates, which are the columns of 'c'. For each one, I also include the text of the character. 

What we see is actually kind of interesting. The network has basically learned to separate out the characters and cluster them a bit. For example, you see how the vowels (a, e, i, o, u) are clustered up here. That's telling us that the neural net treats these as very similar. When they feed into the neural net, the embedding for all these characters is very similar, so the neural net thinks that they're interchangeable. 

The points that are really far away are, for example, 'q'. 'Q' is kind of treated as an exception and has a very special embedding vector. Similarly, '.' (dot), which is a special character, is all the way out here. A lot of the other letters are clustered up here. It's interesting that there's a bit of structure here after the training. It's definitely not random, and these embeddings make sense. 

We're now going to scale up the embedding size. We won't be able to visualize it directly, but we expect that because we're underfitting and we made this layer much bigger and did not sufficiently improve the loss, the constraint to better performance right now could be these embedding vectors. So, let's make them bigger. 

We don't have two-dimensional embeddings anymore. We're going to have 10-dimensional embeddings for each word. This layer will receive 3 times 10, so 30 inputs will go into the hidden layer. Let's also make the hidden layer a bit smaller. Instead of 300, let's just do 200 neurons in that hidden layer. 

Now, the total number of elements will be slightly bigger at 11,000. We have to be careful here because the learning rate we set to 0.1. Here we hardcoded in six. Obviously, if you're working in production, you don't want to be hard-coding magic numbers. Instead of six, this should now be thirty. 

Let's run for fifty thousand iterations. I'll split out the initialization here outside so that when we run this cell multiple times, it's not going to wipe out. 

In addition to that, instead of logging `loss.item`, let's log the `log10` of the loss. I'll show you why in a second. Basically, I'd like to plot the log loss.
# Optimizing Neural Networks: A Practical Approach

Instead of the loss, when you plot it, many times it can have this hockey stick appearance. Using a logarithm squashes it in, making it look nicer. The x-axis represents the step, and in this case, it's 30. 

Let's look at the loss. It's very thick because the mini-batch size is very small, but the total loss over the training set is 2.3, and the test or the dev set is 2.38. So far, so good. 

Now, let's try to decrease the learning rate. We'd hope that we would be able to beat the previous loss, but we're just kind of doing this very haphazardly. I don't actually have confidence that our learning rate is set very well, or that our learning rate decay, which we just do at random, is set very well. 

The optimization here is kind of suspect, to be honest. This is not how you would do it typically in production. In production, you would create parameters or hyperparameters out of all these settings and then you would run lots of experiments and see whichever ones are working well for you.

After decreasing the learning rate, we have 2.17 for the training set and 2.2 for the validation set. You can see how the training and the validation performance are starting to slightly slowly depart. Maybe we're getting the sense that the neural net is getting good enough, or that the number of parameters is large enough that we are slowly starting to overfit. 

Let's maybe run one more iteration of this. But yeah, basically you would be running lots of experiments and then you are slowly scrutinizing whichever ones give you the best dev performance. Once you find all the hyperparameters that make your dev performance good, you take that model and you evaluate the test set performance a single time. That's the number that you report in your paper or wherever else you want to talk about and brag.

After rerunning the plot and rerun, we're getting lower loss now. It is the case that the embedding size is 2.16 for the training set and 2.19 for the validation set. 

There are many ways to go from here. We can continue tuning the optimization. We can continue, for example, playing with the sizes of the neural net. Or we can increase the number of words or characters in our case that we are taking as an input. Instead of just three characters, we could be taking more characters as an input and that could further improve the loss.

I changed the code slightly so we have here 200,000 steps of the optimization. In the first 100,000, we're using a learning rate of 0.1 and then in the next 100,000, we're using a learning rate of 0.01. This is the loss that I achieved. The performance on the training and validation loss is as follows: the best validation loss I've been able to obtain in the last 30 minutes or so is 2.17.

Now, I invite you to beat this number. You have quite a few knobs available to you to surpass this number. You can change the number of neurons in the hidden layer of this model. You can change the dimensionality of the embedding lookup table. You can change the number of characters that are feeding in as an input as the context into this model. And then, of course, you can change the details of the optimization: how long are we running, what is the learning rate, how does it change over time, how does it decay. You can change the batch size and you may be able to actually achieve a much better result.
# Improving Model Convergence Speed and Sampling

In terms of model training, one of the key aspects to consider is the convergence speed. This refers to how many seconds or minutes it takes to train the model and get your result with a really good loss. 

I highly recommend reading this paper, which is 19 pages long. At this point, you should be able to understand a good chunk of it. The paper also presents quite a few ideas for improvements that you can experiment with. All of these are available to you, and you should be able to beat this number. I'm leaving that as an exercise to the reader.

Before we wrap up, I also wanted to demonstrate how you would sample from the model. We're going to generate 20 samples. We begin with all dots as the context. Until we generate the zeroth character again, we're going to embed the current context using the embedding table C. 

Usually, the first dimension was the size of the training set, but here we're only working with a single example that we're generating. This embedding then gets projected into the end state, and you get the logits. 

We calculate the probabilities for that using `f.softmax` of logits. This function exponentiates the logits and makes them sum to one. Similar to cross entropy, it is careful to ensure there are no overflows. Once we have the probabilities, we sample from them using torch multinomial to get our next index. We then shift the context window to append the index and record it. 

We can then decode all the integers to strings and print them out. The model now works much better. The words here are much more word-like or name-like. We have things like 'ham', 'joes', and it's starting to sound a little bit more name-like. We're definitely making progress, but we can still improve on this model quite a lot.

## Bonus Content: Google Colab

I want to make these notebooks more accessible. I don't want you to have to install Jupyter notebooks, torch, and everything else. I will be sharing a link to a Google Colab. Google Colab will look like a notebook in your browser. You can just go to the URL and you'll be able to execute all of the code that you saw in the Google Colab. 

This is me executing the code in this lecture. I shortened it a little bit, but basically, you're able to train the exact same network, plot, and sample from the model. Everything is ready for you to tinker with the numbers right there in your browser, no installation necessary. The link to this will be in the video description.