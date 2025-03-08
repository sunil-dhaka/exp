---
layout: default
title: 1 - Bigram
parent: Andrej Karpathy
has_children: false
nav_order: 1
---

# Building a Character Level Language Model with Make More

Hello everyone, I hope you're all doing well. Today, I'd like to discuss a project I've been working on called Make More. This is a repository that I have on my GitHub page. Just like with my previous project, Micrograd, I'm going to build it out step by step, spelling everything out so we can build it together, slowly but surely.

## What is Make More?

As the name suggests, Make More is designed to create more of whatever you give it. For instance, I have a dataset called `names.txt` that I use with Make More. This dataset is a large collection of names, with lots of different types and variations. In fact, there are approximately 32,000 names that I've randomly found on a government website.

When you train Make More on this dataset, it learns to create more of the same type of data. In this case, it will generate more things that sound like names. These are unique names, so if you're expecting a baby and looking for a unique name, Make More might be able to help you.

Here are some examples of unique names that the neural network generated once we trained it on our dataset:

- Dontel
- Irot
- Zhendi

These names sound like they could be real, but they're not actual names.

## How Does Make More Work?

Under the hood, Make More is a character-level language model. This means that it treats every single line in the dataset as an example, and within each example, it treats them all as sequences of individual characters. For instance, "Reese" would be treated as the sequence of characters "R", "e", "e", "s", "e". 

The model then learns to predict the next character in the sequence. We're going to implement a variety of character-level language models, from simple bi-gram and back of work models to multilingual perceptrons, recurrent neural networks, and even modern transformers. In fact, the transformer we will build will be equivalent to GPT-2, a modern network.

## Future Extensions

After we've mastered character-level language modeling, we'll move on to word-level modeling so we can generate entire documents, not just small segments of characters. We'll also explore image and image-text networks such as Dolly Stable Diffusion. But for now, we're starting with character-level language modeling.

To begin, we'll start with a completely blank Jupyter notebook page. The first step is to load up the `names.txt` dataset. We'll open up `names.txt` for reading and read everything into a massive string. 

Stay tuned for more updates on this exciting project!
# Analyzing and Modeling Character Sequences in Python

In this article, we will be working with a dataset of words, specifically names, and we will be exploring how to analyze and model the character sequences in these words using Python. 

Firstly, we need to extract the individual words from our dataset and put them in a list. We can do this by calling the `splitlines` function on our string of names. This will give us all of our words as a Python list of strings. 

For instance, if we look at the first 10 words in our list, we will see that it's a list of names like Emma, Olivia, Eva, and so on. If we cross-reference this with the top of our dataset, we can confirm that our list is accurate. 

Upon examining this list, it seems like the names are probably sorted by frequency. Now that we have our words, we'd like to learn a little bit more about this dataset. 

Let's start by looking at the total number of words. We expect this to be roughly 32,000. We can also examine the shortest and longest words in our list. By using the `min` and `max` functions on the length of each word, we find that the shortest word has a length of two characters, and the longest word has 15 characters. 

Now, let's think through our very first language model. A character-level language model predicts the next character in a sequence given a specific sequence of characters before it. 

Every single word in our dataset, like 'Isabella', is actually quite a few examples packed into that single word. The existence of a word like 'Isabella' in the dataset tells us that the character 'I' is a very likely character to come first in the sequence of a name. The character 'S' is likely to come after 'I', the character 'A' is likely to come after 'IS', the character 'B' is very likely to come after 'ISA', and so on all the way to 'A' following 'ISABEL'. 

There's one more example packed in here, and that is that after 'ISABELLA', the word is very likely to end. This is one more piece of explicit information that we have to be careful with. There's a lot packed into a single individual word in terms of the statistical structure of what's likely to follow in these character sequences. 

Of course, we don't have just an individual word. We actually have 32,000 of these, so there's a lot of structure here to model. 

In the beginning, I'd like to start with building a bi-gram language model. In the bi-gram language model, we're always working with just two characters at a time. We're only looking at one character that we are given, and we're trying to predict the next character in the sequence. 

For example, we're modeling what characters are likely to follow 'A' and so on. We're just modeling that kind of local structure and forgetting the fact that we may have a lot more information. We're always just looking at the previous character to predict the next one. 

While it's a very simple and weak language model, I think it's a great place to start. Let's begin by looking at these bi-grams in our dataset and what they look like. These bi-grams are just two characters in a row. For each word in our list of words, each word is an individual word.
# Python: Iterating Through Strings and Counting Bigrams

In this tutorial, we will be iterating through a string, specifically a word, with consecutive characters, two characters at a time. This is a neat trick in Python that can be achieved using the `zip` function. 

Let's start with an example. We will iterate through the word 'Emma'. Here's how we can do it:

```python
w = 'Emma'
for character_one, character_two in zip(w, w[1:]):
    print(character_one, character_two)
```

In this example, `w` is the string 'Emma' and `w[1:]` is the string 'mma'. The `zip` function takes two iterators and pairs them up, creating an iterator over the tuples of their consecutive entries. If one of these lists is shorter than the other, it will halt and return. This is why we get the tuples 'e', 'm', 'm', 'a'. 

However, we need to be careful as we have more information here than just these examples. We know that 'e' is very likely to come first and 'a' is coming last. 

To account for this, we can create a special array with a special start and end token. Here's how we can do it:

```python
special_start = ['<s>']
special_end = ['</s>']
all_characters = special_start + list(w) + special_end

for character_one, character_two in zip(all_characters, all_characters[1:]):
    print(character_one, character_two)
```

In this example, the bigram of the start character and 'e' and the bigram of 'a' and the special end character are included. 

Now, let's learn the statistics about which characters are likely to follow other characters. The simplest way to do this in bigram language models is by counting. We will count how often any one of these combinations occurs in the training set. 

To achieve this, we will need a dictionary that will maintain counts for every one of these bigrams. Here's how we can do it:

```python
b = {}
for character_one, character_two in zip(all_characters, all_characters[1:]):
    bigram = (character_one, character_two)
    b[bigram] = b.get(bigram, 0) + 1
```

In this example, `b.get(bigram, 0)` is the same as `b[bigram]`, but in the case that bigram is not in `b`, it will return `0`. 

And that's it! We have successfully iterated through a string and counted the bigrams.
# Analyzing Bigrams in Python

In this tutorial, we will be analyzing bigrams in Python. A bigram is a sequence of two adjacent elements from a string of tokens, which are typically letters, syllables, or words. We will be using a dictionary `b` to count the occurrence of each bigram. By default, we will return to zero plus one. This will add up all the bigrams and count how often they occur.

Let's inspect what `b` is in this case. We see that many bigrams occur just a single time. For example, the bigram 'a' occurred three times, meaning 'a' was an ending character three times. This is true for all of these words: 'Emma', 'Olivia', and 'Eva'. 

Now, `b` will have the statistics of the entire data set. These are the counts across all the words of the individual bigrams. We could, for example, look at some of the most common and least common ones. 

In Python, we can do this by using `b.items()`. `b.items()` returns the tuples of key-value pairs, where the keys are the character bigrams and the values are the counts. We then want to sort this by the values, which are the second element of a tuple. We can do this by using the key `equals lambda` that takes the key-value and returns the key-value at one, not at zero, which is the count. We want to sort by the count of these elements. 

For example, the bigram 'qr' occurs only a single time, and 'dz' occurred only a single time. When we sort this the other way around, we see the most likely bigrams. For instance, 'n' was very often an ending character, and apparently 'n' almost always follows an 'a', making it a very likely combination. 

This is the individual counts that we achieve over the entire data set. However, it's actually going to be significantly more convenient for us to keep this information in a two-dimensional array instead of a Python dictionary. 

We're going to store this information in a 2D array. The rows will be the first character of the bigram and the columns will be the second character. Each entry in this two-dimensional array will tell us how often that first character follows the second character in the data set. 

In particular, the array representation that we're going to use is from PyTorch. PyTorch is a deep learning neural network framework, but part of it is also `torch.tensor`, which allows us to create multi-dimensional arrays and manipulate them very efficiently. 

Let's import PyTorch, which you can do by `import torch`. Then, we can create arrays. Let's create an array of zeros and give it a size of this array. Let's create a 3x5 array as an example.
# Manipulating Tensors in PyTorch

In this tutorial, we will be working with a three by five array of zeros. By default, you'll notice `a.dtype`, which is short for data type, is `float32`. These are single precision floating point numbers. However, because we are going to represent counts, let's actually use `dtype` as `torch.int32`. These are 32-bit integers. Now, you can see that we have integer data inside this tensor.

Tensors allow us to manipulate all the individual entries very efficiently. For example, if we want to change a specific bit, we have to index into the tensor. Here, this is the first row and the fourth column (because it's zero indexed). So, `a[1, 3]` can be set to one. Then, `a` will have a 1 in that position. We can also do things like `a[1, 3] = 2` or `a[1, 3] = 3`. Also, we can say `a[0, 0] = 5`, and then `a` will have a 5 in the top left corner. That's how we can index into the arrays.

Now, the array that we are interested in is much larger. For our purposes, we have 26 letters of the alphabet and then we have two special characters, 's' and 'e'. So, we want a 28 by 28 array. Let's call it `N` because it's going to represent the counts.

```python
N = torch.zeros(28, 28, dtype=torch.int32)
```

Now, instead of having a dictionary `b`, we now have `N`. The problem here is that we have these characters which are strings, but we have to index into an array using integers. So, we need a lookup table from characters to integers.

Let's construct such a character array. We're going to take all the words, which is a list of strings, and concatenate all of it into a massive string. This is simply the entire dataset as a single string. We're going to pass this to the `set` constructor which takes this massive string and throws out duplicates because sets do not allow duplicates. So, `set` of this will just be the set of all the lowercase characters, and there should be a total of 26 of them.

Now, we actually don't want a set, we want a list. But we don't want a list sorted in some arbitrary way, we want it to be sorted from 'a' to 'z'. So, we use `sorted(list(set(words)))`.

Now, what we want is this lookup table. Let's create a special `s2i` (string to integer) mapping.

```python
s2i = {s: i for i, s in enumerate(chars)}
```

`enumerate` basically gives us this iterator over the integer index and the actual element of the list, and then we are mapping the character to the integer. So, `s2i` is a mapping from 'a' to 0, 'b' to 1, etc. all the way to 'z'. That's going to be useful here, but we actually also have to specifically set that 's' will be 26.
# Visualizing Character Bi-grams with Matplotlib

In this tutorial, we will be working with character bi-grams and visualizing them using the Matplotlib library in Python. 

Let's start by mapping characters to integers. For instance, 'a' will map to 0, 'b' to 1, and so on. 'z' will map to 25. We will also map ' ' (space) to 26 and 's' to 27. 

Now, we can map both character 1 and character 2 to their respective integers. This will be done using the `s2i` function at character 1 and `ix2` will be `s2i` of character 2. 

Next, we will create a two-dimensional array using the mapped integers. We will increment the count in the array for each occurrence of a bi-gram. Since everything starts at zero, we should be able to get a large 28 by 28 array of all these counts. 

```python
n[x1, ix2] += 1
```

If we print `n`, we will see the array. However, it might not look very appealing. So, let's try to visualize it in a more appealing way using Matplotlib. 

Matplotlib allows us to create figures. We can visualize the count array using the `imshow` function. 

```python
plt.imshow(n)
```

This will give us a 28x28 array structure. However, it might still look a bit ugly. So, let's try to create a much nicer visualization. 

First, we will need to invert the `s2i` dictionary to get `i2s`, which maps inversely from 0 to 'a', 1 to 'b', etc. 

```python
i2s = {i: s for s, i in s2i.items()}
```

Next, we will create a figure and plot `n`. Then, we will iterate over all the individual cells and create a character string, which is the inverse mapping `i2s` of the integer `i` and the integer `j`. These are the bi-grams in a character representation. 

```python
fig, ax = plt.subplots()
ax.imshow(n)
for i in range(28):
    for j in range(28):
        c = n[j,i]
        ax.text(i, j, f'{i2s[i]}{i2s[j]}:{c.item()}', va='center', ha='center')
```

Here, we are using the `item()` function to get the actual integer value from the tensor. 

This will give us a nice visualization of the counts of each bi-gram. Some of them occur often and some of them do not occur often. 

However, if you scrutinize this carefully, you will notice that we are not being very clever. For example, we have an entire row of completely zeros. That's because the 'end' character is never possibly going to be the first character of a bi-gram. We are always placing these 'end' tokens all at the end of the bi-gram. 

In conclusion, this tutorial showed you how to visualize character bi-grams using Matplotlib. This can be a useful tool for understanding the distribution of bi-grams in a text dataset.
# Cleaning Up a Bigram Language Model

In our bigram language model, we have entire columns of zeros. This is because the 's' character will never possibly be the second element of a bigram. We always start with 's' and we end with 'e', and we only have the words in between. So, we have an entire column of zeros, an entire row of zeros, and in this little two by two matrix here as well. The only one that can possibly happen is if 's' directly follows 'e'. 

That can be non-zero if we have a word that has no letters. In that case, there are no letters in the word, it's an empty word, and we just have 's' follows 'e'. But the other ones are just not possible. So, we're basically wasting space and not only that, but the 's' and the 'e' are getting very crowded here. 

I was using these brackets because there's a convention in natural language processing to use these kinds of brackets to denote special tokens. But we're going to use something else. Let's fix all this and make it prettier. We're not actually going to have two special tokens, we're only going to have one special token. 

So, we're going to have an n by n array of 27 by 27 instead of having two. We will just have one and I will call it a dot. 

Now, one more thing that I would like to do is I would actually like to make this special character hold position zero, and I would like to offset all the other letters off. I find that a little bit more pleasing. So, we need a plus one here so that the first character, which is 'a', will start at one. 

So, 's2i' will now be 'a' starts at one and dot is 0. And 'i2s' of course, we're not changing this because 'i2s' just creates a reverse mapping and this will work fine. So, 1 is 'a', 2 is 'b', 0 is dot. 

So, we've reversed that here. We have a dot and a dot. This should work fine. Make sure I start at zeros count, and then here we don't go up to 28, we go up to 27. 

So, we see that dot never happened, it's at zero because we don't have empty words. Then this row here now is just very simply the counts for all the first letters. For example, 'j' starts a word, 'h' starts a word, 'i' starts a word, etc. And then these are all the ending characters. 

In between, we have the structure of what characters follow each other. So, this is the counts array of our entire data set. This array actually has all the information necessary for us to actually sample from this bigram character-level language model. 

Roughly speaking, what we're going to do is we're just going to start following these probabilities and these counts, and we're going to start sampling from the model. So, in the beginning, of course, we start with the dot, the start token. 

To sample the first character of a name, we're looking at this row here. We see that we have the counts and those counts terminally are telling us how often any one of these characters is to start a word. So, if we take this 'n' and we grab the first row, we can do that by using just indexing as such.
# Sampling from a Multinomial Distribution in PyTorch

In this tutorial, we will be discussing how to sample from a multinomial distribution using PyTorch. We will be using the notation column for the rest of the row. 

Let's start by indexing into the zeroth row and grabbing all the columns. This will give us a one-dimensional array of the first row. The shape of this is 27, which is just the row of 27. 

```python
n[0, :]
```

Another way to achieve this is by simply grabbing the zeroth row like this:

```python
n[0]
```

These are the counts. Now, we'd like to convert these raw counts to probabilities to create a probability vector. 

```python
p = n[0].float()
```

Here, we are converting the integers to floating point numbers. We're creating floats because we're about to normalize these counts. To create a probability distribution, we want to divide by the sum of `p`. 

```python
p /= p.sum()
```

Now, we get a vector of smaller numbers which are now probabilities. Since we divided by the sum, the sum of `p` now is 1. This is a proper probability distribution that sums to 1 and gives us the probability for any single character to be the first character of a word.

Next, we can sample from this distribution. To do this, we're going to use `torch.multinomial`, which samples from the multinomial probability distribution. In simpler terms, you provide probabilities and it will give you integers which are sampled according to the probability distribution.

To make everything deterministic, we're going to use a generator object in PyTorch. This makes everything deterministic so that when you run this on your computer, you're going to get the exact same results that I'm getting here on my computer.

Here's the deterministic way of creating a torch generator object:

```python
g = torch.Generator()
g.manual_seed(1111)
```

We seed the generator with some number that we can agree on, which gives us an object `g`. Then we can pass that `g` to a function that creates random numbers.

```python
torch.rand(3, generator=g)
```

This creates random numbers between 0 and 1. Whenever we run it again, we're always going to get the same result because we keep using the same generator object which we're seeding here.

To normalize, we can divide to get a nice probability distribution of just three elements.

Finally, we can use `torch.multinomial` to draw samples from it:

```python
torch.multinomial(torch.tensor([0.1, 0.2, 0.7], generator=g), 20)
```

This takes the torch tensor of probability distributions, then we can ask for a number of samples, let's say 20. 

And that's it! We have successfully sampled from a multinomial distribution using PyTorch.
# Understanding Replacement Equals True and Torsion Multinomial

Replacement equals true means that when we draw an element, we can draw it and then put it back into the list of eligible indices to draw again. We have to specify replacement as true because, by default, for some reason, it's false. It's just something to be careful with.

The generator is passed in here so we're going to always get deterministic results, the same results. So if I run these two, we're going to get a bunch of samples from this distribution.

Now, you'll notice here that the probability for the first element in this tensor is 60. So in these 20 samples, we'd expect 60% of them to be zero. We'd expect 30% of them to be one. And because the element index two has only a 10% probability, very few of these samples should be two. Indeed, we only have a small number of twos.

We can sample as many as we'd like, and the more we sample, the more these numbers should roughly have the distribution here. So we should have lots of zeros, half as many ones, and three times as few twos. So you see that we have very few twos, we have some ones, and most of them are zero. That's what Torsion Multinomial is doing for us here.

We are interested in this row we've prepared here, and now we can sample from it. So if we use the same seed and then we sample from this distribution, let's just get one sample. Then we see that the sample is, say, 13. So this will be the index. And let's see how it's a tensor that wraps 13, we again have to use that item to pop out that integer, and now the index would be just the number 13.

Of course, we can map the i2s of ix to figure out exactly which character we're sampling here. We're sampling 'm'. So we're saying that the first character in our generation, and just looking at the row here, 'm' was drawn and we can see that 'm' actually starts a large number of words. 'M' started 2,500 words out of 32,000 words, so almost a bit less than 10% of the words start with 'm'. So this was actually a fairly likely character to draw.

That would be the first character of our word and now we can continue to sample more characters because now we know that 'm' is already sampled. So now to draw the next character, we will come back here and we will look for the row that starts with 'm'. So you see 'm' and we have a row here. So we see that 'm.' is 516, 'ma' is this many, and 'mb' is this many, etc. So these are the counts for the next row and that's the next character that we are going to now generate.

I think we are ready to actually just write out the loop because I think you're starting to get a sense of how this is going to go. We always begin at index 0 because that's the start token, and then we continue while true.
# Language Model: Bigram Implementation

In this tutorial, we're going to implement a bigram language model. Let's start by grabbing the row corresponding to the index that we're currently on. In our case, this is `p`, which is `n_array` at `ix`. 

Next, we normalize `p` to sum to one. If you accidentally run an infinite loop, don't worry, just normalize `p` again. 

We then need a generator object. We're going to initialize this up here, and we're going to draw a single sample. This will tell us what index is going to be next. If the index sampled is `0`, then that's now the end token, so we will break. Otherwise, we are going to print `i2s`.

That's pretty much it. This should work okay. The name that we've sampled is what we started with, `m`, the next step, and this dot `we` it here as well. 

Let's actually create an array and instead of printing, we're going to append the results. At the end, let's just join up all the `outs` and print them. 

Now, we're always getting the same result because of the generator. If we want to do this a few times, we can go for `i` in range `10` to sample 10 names and do that 10 times. These are the names that we're getting out. 

I'll be honest with you, this doesn't look right. I spent a few minutes convincing myself that it actually is right. The reason these samples are so terrible is that the bigram language model is actually just really terrible. 

We can generate a few more here, and you can see that they're kind of like their name, like a little bit like "Yanu O'Reilly" etc., but they're just totally messed up. 

The reason that this is so bad is that we're generating `h` as a name. But you have to think through it from the model's eyes. It doesn't know that this `h` is the very first `h`. All it knows is that `h` was previously and now how likely is `h` the last character? Well, it's somewhat likely, and so it just makes it the last character. It doesn't know that there were other things before it or there were not other things before it. That's why it's generating all these nonsense names. 

To convince yourself that this is actually doing something reasonable, even though it's so terrible, is to check if these little `p` here are `27`. So, how about if we did something like this: instead of `p` having any structure whatsoever, what if `p` was just `np.ones(27)` divided by `27`? 

What I'm doing here is creating a uniform distribution which will make everything equally likely. We can sample from that to see if that does any better. 

This is what you get from a model that is completely untrained where everything is equally likely, so it's obviously garbage. But if we have a trained model which is trained on just bi-grams, this is what we get. You can see that it is more name-like. It is actually working, it's just that bigram is so terrible and we have to do better. 

Next, I would like to fix an inefficiency that we have going on here. What we're doing here is we're always fetching a row of `n` from the counts matrix up ahead, and then we're always doing the same normalization.
# Efficient Tensor Manipulation in PyTorch

In this article, we will discuss how to efficiently manipulate tensors in PyTorch. We will focus on the process of converting a matrix of counts into a matrix of probabilities, which is a common task in machine learning and data science.

Let's start with a matrix, `N`, which contains counts of occurrences. In our case, we are working with a 27x27 matrix where each row represents a character and each column represents the next character in a sequence. The value at a specific row and column represents the number of times the corresponding character sequence has occurred.

In our current implementation, we are converting the counts to float and dividing by the sum of the row in every single iteration of a loop. This process of renormalizing the rows is repeated over and over again, which is extremely inefficient and wasteful.

To improve this, we would like to prepare a matrix, `P`, that will contain the probabilities. In other words, it's going to be the same as the `N` matrix, but every single row will be normalized to 1, indicating the probability distribution for the next character given the character before it.

```python
P = N.float().clone()
P /= P.sum(axis=1)
```

However, we need to be careful here. The `sum()` function, by default, sums up all of the counts of the entire matrix and gives us a single number. That's not what we want. We want to simultaneously and in parallel divide all the rows by their respective sums.

To achieve this, we need to use the `sum()` function with a specific dimension. In PyTorch, we can specify the dimension along which we want to sum. In our case, we want to sum up over rows. 

```python
P = N.float().clone()
P /= P.sum(axis=0, keepdim=True)
```

The `keepdim` argument is also important here. If `keepdim` is `True`, then the output tensor is of the same size as the input, except the dimension along which is summed, which will become just one. If you pass in `keepdim` as `False`, then this dimension is squeezed out.

In conclusion, understanding and efficiently manipulating tensors is crucial in PyTorch. This is especially true when building complex models like transformers, where we need to perform some pretty complicated array operations for efficiency.
# Understanding Broadcasting Semantics in Torch

Let's start by examining the shape of a given array. In this case, we have a 1 by 27 row vector. The reason we get a row vector here is because we passed in zero dimension. This zero dimension becomes one, and we've done a sum, resulting in a row. Essentially, we've done the sum vertically and arrived at just a single 1 by 27 vector of counts.

If we remove the `keepdim` parameter, we just get 27. This operation squeezes out that dimension. However, we don't actually want a 1 by 27 row vector because that gives us the counts or the sums across the columns. 

Instead, we want to sum the other way, along dimension one. You'll see that the shape of this is 27 by 1, so it's a column vector. It's a 27 by 1 vector of counts. This is because we're going horizontally and this 27 by 27 matrix becomes a 27 by 1 array.

Interestingly, the actual numbers of these counts are identical. This is because this special array of counts comes from bi-gram statistics. It just so happens, by chance or because of the way this array is constructed, that the sums along the columns or along the rows horizontally or vertically are identical. 

However, what we want to do in this case is sum across the rows horizontally. So, we want a 27 by 1 column vector. 

Now, we have to be careful here. Is it possible to take a 27 by 27 array and divide it by a 27 by 1 array? Is that an operation that you can do? Whether or not you can perform this operation is determined by what's called broadcasting rules. 

If you search for broadcasting semantics in Torch, you'll find a special definition for broadcasting. This determines whether or not these two arrays can be combined in a binary operation like division. 

The first condition is that each tensor has at least one dimension, which is the case for us. Then, when iterating over the dimension sizes starting at the trailing dimension, the dimension sizes must either be equal, one of them is one, or one of them does not exist. 

So, let's align the two arrays and their shapes. Then, we iterate over from the right and going to the left. Each dimension must be either equal, one of them is a one, or one of them does not exist. In this case, they're not equal, but one of them is a one, so this is fine. And then, for the next dimension, they're both equal, so this is also fine. 

Therefore, all the dimensions are fine and this operation is broadcastable. That means that this operation is allowed. 

So, what happens when you divide a 27 by 27 array by a 27 by 1 array? It takes this dimension one and it stretches it out. It copies it to match 27 here in this case. So, in our case, it takes this column vector which is 27 by 1 and it copies it 27 times.
# Understanding Broadcasting Semantics in Python

To make both matrices 27 by 27 internally, you can think of it that way. So, it copies those counts and then does an element-wise division, which is what we want because we want to divide these counts by them on every single one of these columns in this matrix. 

This will normalize every single row. We can check that this is true by taking the first row, for example, and taking its sum. We expect this to be 1 because it's not normalized. Then, we expect this now because if we correctly normalize all the rows, we expect to get the exact same result. Let's run this. It's the exact same result. This is correct. 

Now, I would like to scare you a little bit. You actually have to read through broadcasting semantics. I encourage you to treat this with respect. It's not something to play fast and loose with. It's something to really respect, really understand, and look up maybe some tutorials for broadcasting and practice it. Be careful with it because you can very quickly run into bugs. Let me show you.

You see how here we have `p.sum(1, keepdims=True)`. The shape of this is 27 by 1. Let me take out this line just so we have the `n` and then we can see the counts. We can see that this is all the counts across all the rows. It's a 27 by 1 column vector, right?

Now, suppose that I tried to do the following, but I erase `keepdims=True` here. What does that do? If `keepdims` is not true, it's false. Then, according to documentation, it gets rid of this dimension 1. It squeezes it out. So, basically, we just get all the same counts, the same result, except the shape of it is not 27 by 1, it is just 27. The one disappears. But all the counts are the same.

So, you'd think that this divide would work. First of all, can we even write this, and will it run? Is it even expected to run? Is it broadcastable? Let's determine if this result is broadcastable. `p.sum(1)` is shape 27. Broadcasting into 27. 

So now, the rules of broadcasting are: number one, align all the dimensions on the right. Done. Now, iteration over all the dimensions, starting from the right going to the left. All the dimensions must either be equal, one of them must be one, or one that does not exist. So here, they are all equal. Here, the dimension does not exist. 

So internally, what broadcasting will do is it will create a one here, and then we see that one of them is a one, and this will get copied, and this will run. This will broadcast. Okay, so you'd expect this to broadcast and this we can divide. Now, if I run this, you'd expect it to work, but it doesn't. You actually get garbage. You get a wrong result because this is actually a bug.

In both cases, we are doing the correct counts. We are summing up across the rows, but `keepdims` is saving us and making it work. So in this case, I'd like to encourage you to potentially pause this video at this point and try to think about why this is buggy.
# Understanding Broadcasting in PyTorch

The reason for this discussion is to provide a hint on how broadcasting works in PyTorch. When we have a 27-dimensional vector, internally, during broadcasting, this becomes a 1 by 27 row vector. Now, when we divide 27 by 27 by 1 by 27, PyTorch replicates this dimension. 

Essentially, it takes this row vector and copies it vertically 27 times, so the 27 by 27 lies exactly and element-wise divides. What's happening here is that we're actually normalizing the columns instead of normalizing the rows. 

For instance, `p[0]`, which is the first row of `p.sum()`, is not one, it's seven. It is the first column that sums to one. 

## The Issue with Broadcasting

So, where does the issue come from? The issue arises from the silent addition of a dimension here because, in broadcasting rules, you align on the right and go from right to left. If a dimension doesn't exist, you create it. 

We still did the counts correctly. We did the counts across the rows and got the counts on the right here as a column vector. But because the `keepdims` was true, this dimension was discarded, and now we just have a vector of 27. 

Because of the way broadcasting works, this vector of 27 suddenly becomes a row vector. Then this row vector gets replicated vertically, and that every single point in the opposite direction. 

This thing just doesn't work. This needs to be `keepdims=True` in this case. Then we have that `p[0]` is normalized. Conversely, the first column you'd expect to potentially not be normalized, and this is what makes it work. 

## A Word of Caution

This is pretty subtle and hopefully, this helps to scare you that you should have a respect for broadcasting. Be careful, check your work, and understand how it works under the hood. Make sure that it's broadcasting in the direction that you like. Otherwise, you're going to introduce very subtle bugs, very hard to find bugs. 

One more note on efficiency: we don't want to be doing this here because this creates a completely new tensor that we store into `p`. We prefer to use in-place operations if possible. This would be an in-place operation. It has the potential to be faster, it doesn't create new memory under the hood. 

## Training a Bigram Language Model

We're actually in a pretty good spot now. We trained a bigram language model by counting how frequently any pairing occurs and then normalizing so that we get a nice property distribution. 

These elements of this array `p` are really the parameters of our bigram language model, giving us and summarizing the statistics of these bigrams. We trained the model and then we know how to sample from a model. We just iteratively sample the next character and feed it in each time to get the next character.
# Evaluating the Quality of a Model

In this article, we will discuss how to evaluate the quality of a model. We aim to summarize the quality of the model into a single number. Specifically, we want to understand how good the model is at predicting the training set.

As an example, we can evaluate the training loss in the training set. This training loss gives us an idea about the quality of the model in a single number, similar to what we saw in Micrograd.

To evaluate the quality of the model, we will use the same code we previously used for counting. Let's print these diagrams using f-strings. We will print character one followed by character two. These are the diagrams. For simplicity, we will only do this for the first three words. Here we have the bigrams for 'Emma', 'Olivia', and 'Ava'.

Next, we want to look at the probability that the model assigns to each of these diagrams. In other words, we can look at the probability summarized in the matrix B of i, x1, x2, and then print it as 'probability'. Since these probabilities are quite large, we will truncate them a bit.

What we have here are the probabilities that the model assigns to each of these bigrams in the dataset. Some of them are 4%, 3%, etc. To give you an idea, we have 27 possible characters or tokens. If everything was equally likely, then you'd expect all these probabilities to be around 4%. Anything above 4% means that we've learned something useful from these bigram statistics. Some of these are as high as 40%, 35%, and so on. This shows that the model actually assigns a pretty high probability to whatever's in the training set, which is a good thing.

Ideally, if you have a very good model, you'd expect these probabilities to be near one. This is because it means that your model is correctly predicting what's going to come next, especially on the training set where you trained your model.

Now, we need to think about how we can summarize these probabilities into a single number that measures the quality of this model. In the literature on maximum likelihood estimation and statistical modeling, what's typically used here is something called the 'likelihood'. The likelihood is the product of all these probabilities. It tells us about the probability of the entire dataset assigned by the model that we've trained. This is a measure of quality.

The product of these probabilities should be as high as possible when you are training the model. When you have a good model, the product of these probabilities should be very high. However, because the product of these probabilities is an unwieldy thing to work with, we will discuss a more manageable approach in the next article.
# Understanding Log Likelihood in Machine Learning

In machine learning, we often work with probabilities that are between zero and one. When we multiply these probabilities, we get a very tiny number. For convenience, we usually work with the log likelihood instead of the likelihood.

The product of these probabilities is the likelihood. To get the log likelihood, we just have to take the log of the probability. For example, if we have the log of x from zero to one, the log is a monotonic transformation of the probability. If you pass in one, you get zero. So, a probability of one gives you a log probability of zero. As you go lower and lower in probability, the log will grow more and more negative.

Let's take a look at an example using PyTorch:

```python
log_prob = torch.log(probability)
print(log_prob)
```

As you can see, when we plug in numbers that are very close to some of our higher numbers, we get closer and closer to zero. If we plug in very low probabilities, we get more and more negative numbers.

The reason we work with log likelihood is largely due to convenience. Mathematically, if you have a product of probabilities (a * b * c), the log of these probabilities is just the sum of the logs of the individual probabilities (log(a) + log(b) + log(c)). So, the log likelihood of the product probabilities is just the sum of the logs of the individual probabilities.

```python
log_likelihood = 0
log_likelihood = log(a) + log(b) + log(c)
```

The highest log likelihood can get is zero, when all the probabilities are one. When all the probabilities are lower, this will grow more and more negative.

However, we usually want a loss function where low is good because we're trying to minimize the loss. So, we need to invert the log likelihood. This gives us the negative log likelihood.

```python
negative_log_likelihood = -log_likelihood
```

The negative log likelihood is a very useful loss function. The lowest it can get is zero, and the higher it is, the worse off the predictions are that you're making.

Sometimes, for convenience, we like to normalize the negative log likelihood by making it an average instead of a sum. To do this, we can keep a count of the number of probabilities and divide the negative log likelihood by this count.

```python
n = 0
n += 1
normalized_negative_log_likelihood = negative_log_likelihood / n
```

In conclusion, understanding the concept of log likelihood and its transformation into a loss function is crucial in machine learning. It helps us to quantify the performance of our models and guide the optimization process.
# Maximizing Likelihood in Language Models

In language models, our goal is to maximize likelihood. This is the product of all the probabilities assigned by the model. We aim to maximize this likelihood with respect to the model parameters. In our case, the model parameters are defined in the table. These numbers, the probabilities, are the model parameters in our program language models. 

However, it's important to keep in mind that here we are storing everything in a table format. The probabilities are kept explicitly, but what's coming up is that these numbers will not be kept explicitly. Instead, these numbers will be calculated by a neural network. 

We want to change and tune the parameters of these neural networks. We want to change these parameters to maximize the likelihood, the product of the probabilities. 

Maximizing the likelihood is equivalent to maximizing the log likelihood because log is a monotonic function. Here's the graph of log. Essentially, all it is doing is scaling your loss function. So, the optimization problem here and here are actually equivalent because this is just scaling. 

These are two identical optimization problems. Maximizing the log-likelihood is equivalent to minimizing the negative log likelihood. In practice, people actually minimize the average negative log likelihood to get numbers like 2.4. This number summarizes the quality of your model and we'd like to minimize it and make it as small as possible. The lowest it can get is zero. The lower it is, the better off your model is because it's assigning high probabilities to your data. 

Let's estimate the probability over the entire training set just to make sure that we get something around 2.4. If we run this over the entire training set, we get 2.45. 

Now, what I'd like to show you is that you can actually evaluate the probability for any word that you want. For example, if we just test a single word 'Andre' and bring back the print statement, then you see that 'Andre' is actually kind of an unlikely word. On average, we take three log probability to represent it. Roughly, that's because 'ej' is very uncommon. 

When I take 'Andre' and I append 'q' and 'i', we actually get infinity. That's because 'jq' has a zero percent probability according to our model, so the log likelihood is infinity. 

In conclusion, the job of our training is to find the parameters that minimize the negative log likelihood loss. That would be a high-quality model. The lower the loss function, the better off we are, and the higher it is, the worse off we are.
# Understanding Language Modeling and Model Smoothing

The log of zero will be negative infinity, which results in infinite loss. This is undesirable because we could plug in a string that could be a somewhat reasonable name. However, what this is saying is that this model is exactly zero percent likely to predict this name, and our loss is infinity on this example. 

The reason for this is that 'J' is followed by 'Q' zero times. In other words, the likelihood of 'JQ' is zero percent. This is not ideal and people don't like this too much. To fix this, there's a very simple fix that people like to do to smooth out your model a little bit. It's called model smoothing. 

In model smoothing, we add some fake counts. Imagine adding a count of one to everything. We add a count of one like this and then we recalculate the probabilities. You can add as much as you like, you can add five and it will give you a smoother model. The more you add here, the more uniform model you're going to have and the less you add, the more peaked model you are going to have. One is a pretty decent count to add. This will ensure that there will be no zeros in our probability matrix P. 

This will of course change the generations a little bit. In this case, it didn't, but in principle, it could. What this is going to do now is that nothing will be infinity unlikely. Our model will predict some other probability and we see that 'JQ' now has a very small probability. The model still finds it very surprising that this was a word or a bigram but we don't get negative infinity. It's kind of like a nice fix that people like to apply sometimes and it's called model smoothing.

We've now trained a respectable bi-gram character level language model. We trained the model by looking at the counts of all the bigrams and normalizing the rows to get probability distributions. We can also use those parameters of this model to perform sampling of new words. We sample new names according to those distributions. We can also evaluate the quality of this model. The quality of this model is summarized in a single number which is the negative log likelihood. The lower this number is, the better the model is because it is giving high probabilities to the actual next characters in all the bi-grams in our training set.

We've arrived at this model explicitly by doing something that felt sensible. We were just performing counts and then we were normalizing those counts. Now, I would like to take an alternative approach. We will end up in a very similar position but the approach will look very different. I would like to cast the problem of bi-gram character level language modeling into the neural network framework. 

In the neural network framework, we're going to approach things slightly differently but again end up in a very similar spot. Our neural network is going to be a still a background character level language model so it receives a single character as an input.
# Neural Networks and Character Prediction

Let's discuss a neural network with some weights or parameters, denoted as 'w'. This network is designed to output the probability distribution over the next character in a sequence. Essentially, it's going to make educated guesses as to what is likely to follow a given character that was input into the model.

In addition to this, we're able to evaluate any setting of the parameters of the neural net because we have the loss function, the negative log likelihood. We're going to take a look at its probability distributions and we're going to use the labels, which are basically just the identity of the next character in that diagram, the second character.

Knowing what second character actually comes next in the bigram allows us to then look at how high of a probability the model assigns to that character. We, of course, want the probability to be very high, and that is another way of saying that the loss is low.

We're going to use gradient-based optimization then to tune the parameters of this network because we have the loss function and we're going to minimize it. We're going to tune the weights so that the neural net is correctly predicting the probabilities for the next character.

Let's get started. The first thing I want to do is compile the training set of this neural network. We're going to create the training set of all the bigrams. This code iterates over all the programs. We start with the words, iterate over all the bigrams and previously, as you recall, we did the counts but now we're not going to do counts, we're just creating a training set.

This training set will be made up of inputs and targets, the labels. These bi-grams will denote x, y - those are the characters. We're given the first character of the bi-gram and then we're trying to predict the next one. Both of these are going to be integers.

We'll take x's that append as just x1, y's that append as ix2. We actually don't want lists of integers, we will create tensors out of these. So, axis is torch.tensor of axis and wise a storage.tensor of ys.

We don't actually want to take all the words just yet because I want everything to be manageable. Let's just do the first word which is 'emma'. It's clear what these x's and y's would be.

Let me print character 1, character 2 just so you see what's going on here. The bigrams of these characters are dot e, e m, m m, m a, a dot. So this single word, as I mentioned, has five examples for our neural network. There are five separate examples in 'emma' and those examples are summarized here.

When the input to the neural network is integer 0, the desired label is integer 5 which corresponds to 'e'. When the input to the neural network is 5, we want its weights to be arranged so that 13 gets a very high probability. When 13 is put in, we want 13 to have a high probability.
# Understanding Neural Networks: A Deep Dive

When the number 13 is input, we also want 1 to have a high probability. Similarly, when 1 is input, we want 0 to have a very high probability. So, there are five separate input examples to a neural network.

I wanted to add a tangent of a node of caution to be careful with a lot of the APIs of some of these frameworks. You saw me silently use `torch.tensor` with a lowercase 't', and the output looked right. But you should be aware that there are actually two ways of constructing a tensor. There's a `torch.lowercase tensor` and there's also a `torch.capital tensor` class which you can also construct. 

You can actually call both. You can also do `torch.capital tensor` and you get a nexus and wise as well. So that's not confusing at all. There are threads on what is the difference between these two. Unfortunately, the docs are just not clear on the difference. When you look at the docs of lowercase tensor construct tensor with no autograd history by copying data, it's just like it doesn't make sense. 

The actual difference, as far as I can tell, is explained eventually in this random thread that you can Google. Really, it comes down to `torch.tensor` inferring the data type automatically while `torch.capital tensor` just returns a float tensor. I would recommend sticking to `torch.lowercase tensor`. 

Indeed, we see that when I construct this with a capital 'T', the data type here of 'xs' is float32. But with `torch.lowercase tensor`, you see how it's now `x.dtype` is now integer. So, it's advised that you use lowercase 't'. You can read more about it if you like in some of these threads. 

I'm pointing out some of these things because I want to caution you and I want you to get used to reading a lot of documentation and reading through a lot of Q and A's and threads like this. Some of the stuff is unfortunately not easy and not very well documented, and you have to be careful out there. What we want here are integers because that's what makes sense. And so, lowercase tensor is what we are using.

Now, we want to think through how we're going to feed in these examples into a neural network. It's not quite as straightforward as plugging it in because these examples right now are integers. So there's like a 0, 5, or 13 which gives us the index of the character, and you can't just plug an integer index into a neural net. 

These neural nets are made up of these neurons. These neurons have weights and as you saw in micrograd, these weights act multiplicatively on the inputs `wx + b` and so on. So it doesn't really make sense to make an input neuron take on integer values that you feed in and then multiply on with weights. 

Instead, a common way of encoding integers is what's called one-hot encoding. In one-hot encoding, we take an integer like 13 and we create a vector that is all zeros except for the 13th dimension which we turn to a one. Then, that vector can feed into a neural net.
# PyTorch Tutorial: Encoding Integers into Vectors

In this tutorial, we will be discussing how to encode integers into vectors using PyTorch. PyTorch conveniently has a function inside `torch.nn.functional` that allows us to do this. This function takes a tensor made up of integers and a number of classes, which is how large you want your tensor or vector to be.

Let's start by importing `torch.nn.functional`. This is a common way of importing it:

```python
import torch.nn.functional as F
```

Next, we use the `F.one_hot` function to encode the integers. We can feed in the entire array of integers and specify that the number of classes is 27. This way, it doesn't have to guess the number of classes, which could potentially lead to incorrect results.

```python
x_encoded = F.one_hot(x, num_classes=27)
```

We can then check the shape of our encoded tensor:

```python
print(x_encoded.shape)  # Output: torch.Size([5, 27])
```

This shows that we've successfully encoded all the five examples into vectors. We have five examples, so we have five rows, and each row is now an example that can be fed into a neural network. The appropriate bit is turned on as a one and everything else is zero.

However, we need to be careful with data types. When we're plugging numbers into neural networks, we don't want them to be integers. We want them to be floating point numbers that can take on various values. But the data type here is actually 64-bit integer. This is because `one_hot` received a 64-bit integer and returned the same data type. Unfortunately, `one_hot` doesn't take a desired data type of the output tensor. So, we need to cast this to float like this:

```python
x_encoded = x_encoded.float()
```

Now, everything looks the same, but the data type is `float32`, which can be fed into neural networks.

## Constructing Our First Neuron

Now, let's construct our first neuron. This neuron will look at these input vectors and perform a very simple function: `w * x + b`, where `w * x` is a dot product.

First, let's define the weights of this neuron. We can initialize them with `torch.randn`. `torch.randn` fills a tensor with random numbers drawn from a normal distribution. A normal distribution has a probability density function like this, so most of the numbers drawn from this distribution will be around 0.

```python
weights = torch.randn(27, dtype=torch.float32)
```

And that's it! We've successfully encoded integers into vectors and constructed our first neuron using PyTorch.
# Evaluating Neurons with PyTorch

In this article, we will discuss how to evaluate neurons using PyTorch. We will start with a column vector of 27 numbers, which we will refer to as `w`. These weights are then multiplied by the inputs. 

To perform this multiplication, we can take `x_encoding` and multiply it with `w`. This operation is a matrix multiplication operator in PyTorch. The output of this operation is a 5 by 1 matrix. 

The reason for this is as follows: we took `x_encoding`, which is a 5 by 27 matrix, and we multiplied it by a 27 by 1 matrix. In matrix multiplication, the output will become a 5 by 1 matrix because these 27 will multiply and add. 

So, what we're seeing here, out of this operation, is the five activations of this neuron on these five inputs. We've evaluated all of them in parallel. We didn't feed in just a single input to the single neuron; we fed in simultaneously all the five inputs into the same neuron. 

In parallel, PyTorch has evaluated the `wx + b` but here is just the `wx`, there's no bias. It has value `w` times `x` for all of them independently. 

Now, instead of a single neuron, I would like to have 27 neurons. I'll show you in a second why I want 27 neurons. So, instead of having just a 1 here, which is indicating the presence of one single neuron, we can use 27. 

When `w` is 27 by 27, this will in parallel evaluate all the 27 neurons on all the 5 inputs, giving us a much bigger result. So now, what we've done is 5 by 27 multiplied by 27 by 27, and the output of this is now 5 by 27. 

So, what is every element here telling us? It's telling us, for every one of 27, what is the firing rate of those neurons on every one of those five examples. 

For example, the element 3 comma 13 is giving us the firing rate of the 13th neuron looking at the third input. This was achieved by a dot product between the third input and the 13th column of this `w` matrix. 

Using matrix multiplication, we can very efficiently evaluate the dot product between lots of input examples in a batch and lots of neurons where all those neurons have weights in the columns of those `w's`. 

In matrix multiplication, we're just doing those dot products in parallel. Just to show you that this is the case, we can take `x` and we can take the third row and we can take the `w` and take its 13th column. Then we can do `x` and get three elementwise multiply with `w` at 13. 

Sum that up, that's `wx + b`. Well, there's no `+ b`, it's just `wx` dot product.
# Understanding Neural Networks: A Simple Linear Layer

In this discussion, we will delve into the workings of a neural network, specifically focusing on a simple linear layer. 

We start by feeding our 27-dimensional inputs into the first layer of a neural network that has 27 neurons. So, we have 27 inputs and now we have 27 neurons. These neurons perform the operation `W times X`. They don't have a bias and they don't have a non-linearity like `tanh`. We're going to leave them to be a linear layer. 

In addition to that, we're not going to have any other layers. This is going to be it. It's just going to be the simplest neural net, which is just a single linear layer. 

Now, I'd like to explain what I want those 27 outputs to be. Intuitively, what we're trying to produce here for every single input example is we're trying to produce some kind of a probability distribution for the next character in a sequence. And there are 27 of them. But we have to come up with precise semantics for exactly how we're going to interpret these 27 numbers that these neurons take on. 

You see here that these numbers are negative and some of them are positive, etc. That's because these are coming out of a neural net layer initialized with normal distribution parameters. But what we want is something like we had here. Each row here told us the counts and then we normalized the counts to get probabilities. We want something similar to come out of the neural net. But what we just have right now is just some negative and positive numbers. 

We want those numbers to somehow represent the probabilities for the next character. But you see that probabilities have a special structure. They're positive numbers and they sum to one. That doesn't just come out of a neural net. And then they can't be counts because these counts are positive and counts are integers. So counts are also not really a good thing to output from a neural net. 

So instead, what the neural net is going to output and how we are going to interpret the 27 numbers is that these 27 numbers are giving us log counts, basically. So instead of giving us counts directly like in this table, they're giving us log counts. And to get the counts, we're going to take the log counts and we're going to exponentiate them. 

Exponentiation takes the following form: it takes numbers that are negative or they are positive, it takes the entire real line. If you plug in negative numbers, you're going to get `e to the x`, which is always below one. So you're getting numbers lower than one. And if you plug in numbers greater than zero, you're getting numbers greater than one all the way growing to infinity. This here grows to zero. 

So basically, we're going to take these numbers here and instead of them being positive and negative and all over the place, we're going to interpret them as log counts. And then we're going to element-wise exponentiate these numbers.
# Understanding Neural Networks: A Deep Dive into Logits, Counts, and Probabilities

In this article, we will explore the concept of exponentiating numbers in the context of neural networks. This process gives us something interesting to observe. When we exponentiate numbers, all the negative numbers turn into numbers below 1, like 0.338, and all the positive numbers originally turn into even more positive numbers, greater than one. 

For instance, let's take the number seven. It's a positive number greater than zero. But when we exponentiate it, we get an output that we can use and interpret as the equivalent of counts. 

In the context of a neural network, these counts are positive numbers. They can never be below zero, which makes sense. They can now take on various values, depending on the settings of W (weights). 

Let's break this down further. We're going to interpret these to be the logits, often referred to as log counts. These logits, when exponentiated, give us something that we can interpret as counts. This is equivalent to the N matrix or the array of counts that we used previously. Each row here represents the counts for a specific example.

Now, the probabilities are just the counts normalized. We've already done this before. We want counts that sum along the first dimension and we want to keep them as true. This is how we normalize the rows of our counts matrix to get our probabilities. 

When we look at these probabilities, we see that every row sums to 1 because they're normalized. The shape of this is 5 by 27. What we've achieved is that for every one of our five examples, we now have a row that came out of a neural network. 

Because of the transformations here, we made sure that the output of this neural network now are probabilities or we can interpret them to be probabilities. Our WX (weighted inputs) gave us logits, and then we interpret those to be log counts. We exponentiate to get something that looks like counts, and then we normalize those counts to get a probability distribution. 

All of these are differentiable operations. So, what we've done now is we're taking inputs, we have differentiable operations that we can back propagate through, and we're getting out probability distributions. 

For example, for the zeroth example that we fed in, which was a one-hot vector of zero, it basically corresponded to feeding in a specific example. So, we're feeding in a specific example and getting out a probability distribution. 

In conclusion, understanding these concepts is crucial in the field of neural networks and machine learning. It helps us understand how we can manipulate and interpret the outputs of a neural network.
# Understanding Neural Networks

We start by feeding a 'dot' into a neural network. The process involves first getting its index, then one-hot encoding it. After this, it goes into the neural network, and out comes a distribution of probabilities. This distribution has a shape of 27, representing 27 numbers. We interpret this as the neural network's assignment for how likely each of the 27 characters is to come next. 

As we tune the weights (W), we will, of course, get different probabilities out for any character that you input. The question now is, can we optimize and find a good W such that the probabilities coming out are pretty good? We measure 'pretty good' by the loss function.

## A Summary of the Process

Let's organize everything into a single summary for clarity. It starts with an input dataset. We have some inputs to the neural network and some labels for the correct next character in a sequence. These are integers. 

Here, I'm using Torch generators, so you see the same numbers that I see. I'm generating 27 neurons weights. 

Next, we're going to plug in all the input examples (X's) into a neural network. This is a forward pass. First, we have to encode all of the inputs into one-hot representations. So, we have 27 classes. We pass in these integers, and X_inc becomes an array that is 5 by 27 zeros, except for a few ones. 

We then multiply this in the first layer of a neural network to get logits. We exponentiate the logits to get fake counts, sort of, and normalize these counts to get probabilities. 

The last two lines here are called the softmax. Softmax is a very often used layer in a neural network that takes these Z's, which are logits, exponentiates them, and divides and normalizes. It's a way of taking outputs of a neural network layer (these outputs can be positive or negative) and outputting probability distributions. It outputs something that always sums to one and are positive numbers, just like probabilities. 

It's kind of like a normalization function if you want to think of it that way. You can put it on top of any other linear layer inside a neural network, and it basically makes a neural network output probabilities. That's very often used, and we used it as well here. 

This is the forward pass, and that's how we made a neural network output probability. 

You'll notice that this entire forward pass is made up of differentiable layers. Everything here, we can backpropagate through. All that's happening here is just multiplication and addition. We know how to backpropagate through them. Exponentiation, we know how to backpropagate through. And here we are summing, and sum is easily backpropagable as well, and division as well. So everything here is a differentiable operation, and we can backpropagate through. 

Now, we achieve these probabilities which are 5 by 27. For every single example, we have a probability distribution.
# Understanding Neural Networks: A Deep Dive into Probabilities and Loss

In this article, we will break down the concept of neural networks using a simple example. We will use the word 'Emma' as our example, which is made up of five bigrams. 

A bigram is a sequence of two adjacent elements from a string of tokens. For instance, in our example, 'Emma', the first bigram is '.E' where 'E' is the beginning character right after the dot. The indexes for these are zero and five. 

This zero is then fed into the neural network as an input. The neural network then generates a vector of probabilities, which in this case, is 27 numbers. The label for this input is 5 because 'E' actually comes after the dot. This label is then used to index into the probability distribution. 

The index 5 here is 0, 1, 2, 3, 4, 5. It's this number here, which is the probability assigned by the neural network to the actual correct character. However, the network currently thinks that the next character, 'E', following the dot is only one percent likely. This is not very good because this is a training example and the network thinks this is currently very unlikely. This is just because we didn't get very lucky in generating a good setting of 'W'. 

The log likelihood then is very negative, and the negative log likelihood is very positive. This results in a high loss because the loss is just the average negative log likelihood. 

For the second character 'EM', the network thought that 'M' following 'E' is very unlikely. For 'M' following 'M', it thought it was two percent likely. For 'A' following 'M', it actually thought it was seven percent likely. So, just by chance, this one actually has a pretty good probability and therefore a pretty low negative log likelihood. 

Overall, our average negative log likelihood, which is the loss, is 3.76. This is a fairly high loss and indicates that this is not a very good setting of 'W'. 

However, we can change our 'W'. We can resample it and get a different 'W'. With this different setting of 'W', we now get a loss of 3.37. This is a much better 'W' because the probabilities just happen to come out higher for the characters that actually are next. 

You might think that we can just keep resampling this until we get a good result. However, this method of randomly assigning parameters and seeing if the network is good is not how you optimize a neural network. The way you optimize a neural network is you start with some random guess and work from there. 

In the next article, we will discuss how to optimize a neural network starting from a random guess. Stay tuned!
# Committing to a Loss Function

We are going to commit to this one, even though it's not very good. The big deal now is that we have a loss function. This loss is made up only of differentiable operations, and we can minimize the loss by tuning 'ws'. We compute the gradients of the loss with respect to these 'w' matrices. Then, we can tune 'w' to minimize the loss and find a good setting of 'w' using gradient-based optimization. 

Let's see how that will work. Now, things are actually going to look almost identical to what we had with Micrograd. I pulled up the lecture from Micrograd, the notebook is from this repository. When I scroll all the way to the end, where we left off with Micrograd, we had something very similar. 

We had a number of input examples. In this case, we had four input examples inside 'axis', and we had their targets. These are targets, just like here we have our 'axes' now, but we have five of them and they're now integers instead of vectors. We're going to convert our integers to vectors, except our vectors will be 27 large instead of three large. 

Then, we did a forward pass where we ran a neural net on all of the inputs to get predictions. Our neural net at the time, this 'nfx', was a multi-layer perceptron. Our neural net is going to look different because our neural net is just a single layer, a single linear layer followed by a softmax. That's our neural net. 

The loss here was the mean squared error. We simply subtracted the prediction from the ground truth, squared it, and summed it all up. That was the loss. The loss was a single number that summarized the quality of the neural net. When the loss is low, like almost zero, that means the neural net is predicting correctly. 

We had a single number that summarized the performance of the neural net. Everything here was differentiable and was stored in a massive compute graph. Then we iterated over all the parameters, we made sure that the gradients are set to zero, and we called 'loss.backward()'. 'Loss.backward()' initiated back propagation at the final output node of loss. 

Remember these expressions we had? Loss all the way at the end, we start back propagation and we went all the way back. We made sure that we populated all the 'parameters.grad'. That 'grad' started at zero, but back propagation filled it in. 

Then in the update, we iterated over all the parameters and we simply did a parameter update where every single element of our parameters was nudged in the opposite direction of the gradient. We're going to do the exact same thing here. 

I'm going to pull this up so that we have it available. We're actually going to do the exact same thing. This was the forward pass where we did this, and 'probs' is our 'y_pred'. Now we have to evaluate the loss, but we're not using the mean squared error, we're using the negative log likelihood because we are doing classification, not regression. 

Here, we want to calculate loss. The way we calculate it is just this average negative log likelihood. This 'probs' here has a shape of 5 by 27.
# Efficiently Accessing Probabilities with PyTorch

In this tutorial, we will be discussing how to efficiently access probabilities using PyTorch. We will be going through the process step by step, starting with defining the forward pass, forwarding the network and the loss, and finally doing the backward pass.

## Defining the Forward Pass

To begin with, we want to pluck out the probabilities at the correct indices. This is because the labels are stored array-wise. For instance, for the first example, we are looking at the probability of five at index five. For the second example, at the second row or row index one, we are interested in the probability assigned to index 13. Similarly, at the third row, we want one, and at the last row, which is four, we want zero. These are the probabilities we are interested in.

However, we want a more efficient way to access these probabilities, not just listing them out in a tuple. One of the ways to do this in PyTorch is to pass in all of these integers in the vectors. For example, we can create a range using `torch.range(5)`, which gives us 0, 1, 2, 3, 4. So, we can index here with `torch.range(5)` and index with `ys`. This plucks out the probabilities that the neural network assigns to the correct next character.

## Forwarding the Network and the Loss

Next, we take these probabilities and look at the log probability. We do this by using `.log()`, and then we take the mean of all of that. It's the negative average log likelihood that is the loss. So, the loss here is 3.76, which is exactly as we've obtained before, but this is a vectorized form of that expression. We get the same loss, and we can consider this part of the forward pass. We've now achieved the loss.

## Doing the Backward Pass

Now that we've defined the forward pass and forwarded the network and the loss, we're ready to do the backward pass. We want to first make sure that all the gradients are reset to zero. In PyTorch, you can set the gradients to be zero, but you can also just set it to `None`. Setting it to `None` is more efficient, and PyTorch will interpret `None` as a lack of a gradient, which is the same as zeros.

Before we do `loss.backward()`, we need to do one more thing. PyTorch actually requires that we pass in `requires_grad=True` so that we tell PyTorch that we are interested in calculating gradients for this leaf tensor. By default, this is `False`.

After setting the gradients to `None` and running `loss.backward()`, something magical happens. PyTorch, just like Micrograd, keeps track of all the operations when we do the forward pass. It builds a full computational graph under the hood, just like the graphs we've produced in Micrograd. These graphs exist inside PyTorch, and it knows all the dependencies and all the mathematical operations of everything. When you then calculate the loss, PyTorch uses this information to efficiently calculate the gradients.
# Neural Network Gradients and Loss Function

We can call a dot backward on our neural network, and that backward then fills in the gradients of all the intermediates, all the way back to the parameters of our neural network, which are represented by 'w'. 

Now, we can do `w.grad` and we see that it has gradients for every single element. So, `w.shape` is 27 by 27, `w.grad.shape` is the same 27 by 27, and every element of `w.grad` is telling us the influence of that weight on the loss function. 

For example, the zero-zero element of `w` has a positive gradient, which tells us that it has a positive influence on the loss. Slightly nudging `w` or adding a small 'h' to it would increase the loss mildly because this gradient is positive. Some of these gradients are also negative, which tells us about the gradient information. 

We can use this gradient information to update the weights of this neural network. The update is going to be very similar to what we had in micrograd. We need no loop over all the parameters because we only have one parameter tensor, and that is `w`. So, we simply do `w.data += -step_size * w.grad`, and that would be the update to the tensor. 

Because the tensor is updated, we would expect that now the loss should decrease. It was 3.76, but after updating `w` and recalculating the forward pass, the loss now should be slightly lower, at 3.74. 

We can again set the gradient to none, perform a backward pass, update, and now the parameters have changed again. So, if we recalculate the forward pass, we can see that the loss has decreased further. 

When we achieve a low loss, that will mean that the network is assigning high probabilities to the correct characters. 

I rearranged everything and put it all together from scratch. Here is where we construct our dataset of bigrams. Currently, we are just working with the word 'emma' and there are five bigrams there. 

I added a loop of exactly what we had before, so we had 10 iterations of gradient descent of forward pass, backward pass, and an update. Running these two cells, initialization and gradient descent, gives us some improvement on the loss function. 

But now, I want to use all the words. There are not just 5 but 228,000 bigrams now. However, this should require no modification whatsoever. Everything should just run because all the code we wrote doesn't care if there are five bigrams or 228,000 bigrams. With everything in place, it should just work. 

You can see that this will just run, but now we are optimizing over the entire dataset.
# Bigram Language Models: A Deep Dive

In this article, we will explore the entire training set of all the bigrams. As we progress, you will notice that we are decreasing very slightly. So, we can actually use a larger learning rate. Even 50 seems to work on this very simple example. 

Let's re-initialize and run 100 iterations. We seem to be coming up with some pretty good losses here, around 2.47. Let's run 100 more iterations. 

What is the number that we expect by the way in the loss? We expect to get something around what we had originally. So, all the way back, if you remember in the beginning of this video when we optimized just by counting, our loss was roughly 2.47 after we had it smoothing. But before smoothing, we had roughly 2.45 likelihood, or loss. 

That's actually roughly the vicinity of what we expect to achieve. But before, we achieved it by counting and here we are achieving the roughly the same result but with gradient-based optimization. So we come to about 2.4, 6 2.45 etc. 

This makes sense because fundamentally we're not taking any additional information. We're still just taking in the previous character and trying to predict the next one. But instead of doing it explicitly by counting and normalizing, we are doing it with gradient-based learning. 

It just so happens that the explicit approach happens to very well optimize the loss function without any need for a gradient-based optimization because the setup for bigram language models is so straightforward, so simple. We can just afford to estimate those probabilities directly and maintain them in a table. 

But the gradient-based approach is significantly more flexible. We've actually gained a lot because what we can do now is we can expand this approach and complexify the neural net. 

Currently, we're just taking a single character and feeding it into a neural net. The neural net is extremely simple, but we're about to iterate on this substantially. We're going to be taking multiple previous characters and we're going to be feeding them into increasingly more complex neural nets. 

Fundamentally, the output of the neural net will always just be logits. Those logits will go through the exact same transformation. We are going to take them through a softmax, calculate the loss function and the negative log likelihood, and do gradient-based optimization. 

As we complexify the neural nets and work all the way up to transformers, none of this will really fundamentally change. The only thing that will change is the way we do the forward pass where we take in some previous characters and calculate logits for the next character in the sequence. That will become more complex. 

But we'll use the same machinery to optimize it. It's not obvious how we would have extended this bigram approach into the case where there are many more characters at the input because eventually these tables would get way too large. There's way too many combinations of what previous characters could be. If you only have one previous character, we can just keep everything in a table.
# Neural Network Approach to Language Modeling

In this article, we will delve into the neural network approach to language modeling. This approach is significantly more scalable than traditional methods and can be improved over time. 

Firstly, I want you to notice that the `x_ink` here is made up of one-hot vectors. These one-hot vectors are then multiplied by the `W` matrix. We often think of this as multiple neurons being forwarded in a fully connected manner. However, what's actually happening is that, for example, if you have a one-hot vector that has a one at the fifth dimension, then because of the way the matrix multiplication works, multiplying that one-hot vector with `W` actually ends up plucking out the fifth row of `W`. The `log_logits` would then become just the fifth row of `W`. This is due to the way the matrix multiplication works.

This is exactly what happened before. Remember, when we had a bigram, we took the first character and that first character indexed into a row of this array. That row gave us the probability distribution for the next character. So, the first character was used as a lookup into a matrix to get the probability distribution. 

This is exactly what's happening here because we're taking the index, encoding it as one-hot, and multiplying it by `W`. So `log_logits` literally becomes the appropriate row of `W`. This, just as before, gets exponentiated to create the counts and then normalized to become a probability. 

So, this `W` here is literally the same as this array here. But `W` is the log counts, not the counts. So it's more precise to say that `W` exponentiated, `W.x`, is this array. But this array was filled in by counting and populating the counts of bi-grams. Whereas in the gradient-based framework, we initialize it randomly and then we let the loss guide us to arrive at the exact same array. 

This array here is basically the array `W` at the end of optimization. We arrived at it piece by piece by following the loss. That's why we also obtain the same loss function at the end.

Secondly, remember the smoothing where we added fake counts to our counts in order to smooth out and make more uniform the distributions of these probabilities. This prevented us from assigning zero probability to any one bigram. Now, as we increase the count, the probability becomes more and more uniform. This is because these counts go only up to around 900. So, if we're adding a million to every single number here, you can see how the row and its probability then become more uniform.
# Understanding Regularization in Machine Learning

The divide in machine learning is going to become more and more close to exactly even probability, which is a uniform distribution. It turns out that the gradient-based framework has an equivalent to smoothing. 

In particular, let's think through these weights (W's) here, which we initialized randomly. We could also think about initializing W's to be zero. If all the entries of W are zero, then you'll see that logits will become all zero. And then, exponentiating those logits becomes all one. The probabilities then turn out to be exactly uniform. 

So basically, when W's are all equal to each other or say, especially zero, then the probabilities come out completely uniform. Trying to incentivize W to be near zero is basically equivalent to label smoothing. The more you incentivize that in the loss function, the more smooth distribution you're going to achieve. 

This brings us to something that's called regularization. We can actually augment the loss function to have a small component that we call a regularization loss. In particular, what we're going to do is, we can take W and we can, for example, square all of its entries. Because we're squaring, there will be no signs anymore. Negatives and positives all get squashed to be positive numbers. 

The way this works is, you achieve zero loss if W is exactly zero. But if W has non-zero numbers, you accumulate loss. We can actually take this and we can add it on here. So we can do something like `loss + W.square().mean()`. We can choose the regularization strength and then we can just optimize this. 

Now, this optimization actually has two components. Not only is it trying to make all the probabilities work out, but in addition to that, there's an additional component that simultaneously tries to make all W's be zero. Because if W's are non-zero, you feel a loss. Minimizing this, the only way to achieve that is for W to be zero. 

You can think of this as adding a spring force or like a gravity force that pushes W to be zero. So W wants to be zero and the probabilities want to be uniform. But they also simultaneously want to match up your probabilities as indicated by the data. 

The strength of this regularization is exactly controlling the amount of counts that you add here. Adding a lot more counts here corresponds to increasing this number. Because the more you increase it, the more this part of the loss function dominates this part and the more these weights will be unable to grow. Because as they grow, they accumulate way too much loss. 

If this is strong enough, then we are not able to overcome the force of this loss and we will never make predictions that are not uniform. I thought that's kind of cool.
# Sampling from a Neural Net Model

Before we wrap up, I wanted to show you how you would sample from this neural net model. I've copied the sampling code from before, where we sampled five times. We started at zero, grabbed the current `ix` row of `p`, and that was our probability row. From this, we sampled the next index and accumulated that, breaking when zero. Running this gave us these results. We still have the `p` in memory, so this is fine.

Now, the speed doesn't come from the row of `b`. Instead, it comes from this neural net. First, we take `ix` and encode it into a one-hot row of `x_inc`. This `x_inc` multiplies `rw`, which really just plucks out the row of `w` corresponding to `ix`. That's what's happening. This gets our logits, and then we normalize those logits. We exponentiate to get counts and then normalize to get the distribution. Then, we can sample from the distribution.

Depending on how you look at it, it's kind of anticlimactic or climatic, but we get the exact same result. That's because this is the identical model. Not only does it achieve the same loss, but as I mentioned, these are identical models. This `w` is the log counts of what we've estimated before, but we came to this answer in a very different way. It's got a very different interpretation, but fundamentally, this is basically the same model and gives the same samples.

We've actually covered a lot of ground. We introduced the bigram character-level language model. We saw how we can train the model, how we can sample from the model, and how we can evaluate the quality of the model using the negative log-likelihood loss. Then, we actually trained the model in two completely different ways that actually get the same result and the same model.

In the first way, we just counted up the frequency of all the bigrams and normalized. In the second way, we used the negative log-likelihood loss as a guide to optimizing the counts matrix or the counts array so that the loss is minimized in a gradient-based framework. We saw that both of them give the same result.

The second one of these, the gradient-based framework, is much more flexible. Right now, our neural network is super simple. We're taking a single previous character and taking it through a single linear layer to calculate the logits. This is about to complexify. In the follow-up videos, we're going to be taking more and more of these characters and feeding them into a neural net. But this neural net will still output the exact same thing. The neural net will output logits, and these logits will still be normalized in the exact same way. All the loss and everything else and the gradient-based framework stay identical. It's just that this neural net will now complexify all the way to transformers. So, that's going to be pretty awesome.