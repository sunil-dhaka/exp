---
layout: default
title: 6 - Lets build gpt
parent: Andrej Karpathy
has_children: false
nav_order: 6
---

# Understanding Chachi PT and the Transformer Architecture

Hello everyone! By now, you have probably heard of Chachi PT. It has taken the world and the AI community by storm. Chachi PT is a system that allows you to interact with an AI and give it text-based tasks. 

For example, we can ask Chachi PT to write us a small haiku about how important it is for people to understand AI. They can then use it to improve the world and make it more prosperous. When we run this, we get something like: 

> "AI knowledge brings prosperity for all to see. Embrace its power."

Not bad, right? You can see that Chachi PT generated these words sequentially. I asked it the exact same prompt a little bit earlier and it generated a slightly different outcome:

> "AI is power to grow. Ignorance holds us back. Learn. Prosperity waits."

Both are pretty good and slightly different. This shows that Chachi PT is a probabilistic system. For any one prompt, it can give us multiple answers. 

This is just one example of a prompt. People have come up with many examples and there are entire websites that index interactions with Chachi PT. Many of them are quite humorous. For instance, "Explain HTML to me like I'm a dog", "Write release notes for Chess 2", "Write a note about Elon Musk buying Twitter", and so on. 

As another example, "Please write a breaking news article about a leaf falling from a tree". The AI might respond with:

> "In a shocking turn of events, a leaf has fallen from a tree in the local park. Witnesses report that the leaf, which was previously attached to a branch of a tree, detached itself and fell to the ground."

Very dramatic, right? This is a pretty remarkable system. It is what we call a language model because it models the sequence of words, characters, or tokens more generally. It knows how words follow each other in the English language. From its perspective, it is completing the sequence. I give it the start of a sequence and it completes the sequence with the outcome. 

Now, I would like to focus on the components under the hood of what makes Chachi PT work. What is the neural network under the hood that models the sequence of these words? 

That comes from a paper called "Attention is All You Need" published in 2017. This was a landmark paper in AI that proposed the Transformer architecture. 

GPT is short for Generative Pretrained Transformer. The Transformer is the neural network that does all the heavy lifting under the hood. It comes from this 2017 paper. 

If you read this paper, it reads like a machine translation paper. That's because I think the authors didn't fully anticipate the impact that the Transformer would have on the field. This architecture, produced in the context of machine translation, ended up taking over the rest of AI in the next five years. This architecture, with minor changes, was copied into a huge number of applications in AI in more recent years, including at the core of Chachi PT. 

Now, what I'd like to do is build out something like Chachi PT. Of course, we're not going to be able to reproduce Chachi PT. This is a very serious, production-grade system. It is trained on a good chunk of the internet and there are a lot of pre-training and fine-tuning stages to it. It's very complicated. 

What I'd like to focus on is just to train a Transformer-based language model. In our case, it's going to be a character level.
# Understanding Language Models and Transformers

A language model is a fascinating tool that can be very educational in understanding how these systems work. However, to train such a model, we don't need to use a chunk of the internet. Instead, we can work with a smaller dataset. In this case, I propose that we work with my favorite toy dataset, called Tiny Shakespeare.

Tiny Shakespeare is a concatenation of all the works of Shakespeare. It's essentially all of Shakespeare's works in a single file. This file is about one megabyte and contains all of Shakespeare's works. 

We are going to model how these characters follow each other. For example, given a chunk of these characters, or given some context of characters in the past, the Transformer neural network will look at the characters that I've highlighted and predict that 'g' is likely to come next in the sequence. 

We're going to train the Transformer on Shakespeare's works, and it's going to try to produce character sequences that look like this. In that process, it's going to model all the patterns inside this data. Once we've trained the system, we can generate infinite Shakespeare. Of course, it's a fake thing that looks kind of like Shakespeare. 

You can see how this is going character by character and it's kind of like predicting Shakespeare-like language. For example, "Verily my Lord, the sights have left the again the king coming with my curses with precious pale," and then Tronio says something else. This is just coming out of the Transformer in a very similar manner as it would come out in Chachi PT. In our case, it's character by character. In Chachi PT, it's coming out on the token by token level and tokens are these little sub-word pieces. They're not word level, they're kind of like word chunk level.

I've already written the entire code to train these Transformers. It is in a GitHub repository that you can find, and it's called Nano GPT. Nano GPT is a repository for training Transformers on any given text. What I think is interesting about it is that it's a very simple implementation. It's just two files of the GPT model, the Transformer, and one file trains it on some given text dataset. 

Here, I'm showing that if you train it on an OpenWebText dataset, which is a fairly large dataset of web pages, then I reproduce the performance of GPT-2. GPT-2 is an early version of OpenAI's GPT from 2017. I've only so far reproduced the smallest 124 million parameter model, but this is just proving that the codebase is correctly arranged and I'm able to load the neural network weights that OpenAI has released later.

You can take a look at the finished code here in Nano GPT. But what I would like to do in this lecture is to write this repository from scratch. We're going to begin with an empty file and we're going to define a Transformer piece by piece. We're going to train it on the Tiny Shakespeare dataset and we'll see how we can then generate infinite Shakespeare. 

Of course, this can be copy-pasted to any arbitrary text dataset that you like. But my goal here is to make you understand and appreciate how, under the hood, Chat GPT works. Really, all that's required is a proficiency in Python and some basic understanding of calculus and statistics. It would also help if you have seen my previous lectures.
# Building a Transformer Neural Network Language Model

In previous videos on the same YouTube channel, particularly my "Make More" series, I defined smaller and simpler neural network language models such as multilevel perceptrons. This really introduces the language modeling framework. In this video, we're going to focus on the Transformer neural network itself.

I created a new Google Colab Jupyter notebook. This will allow me to easily share the code that we're going to develop together. You can find this in the video description for you to follow along.

I've done some preliminaries. I downloaded the dataset, the tiny Shakespeare dataset, from a specific URL. It's about a one-megabyte file. I opened the `input.txt` file and read in all the text as a string. We are working with roughly 1 million characters. The first 1000 characters, if we just print them out, are basically what you would expect. This is the first 1000 characters of the tiny Shakespeare dataset.

Next, we're going to take this text, which is a sequence of characters in Python. When I call the set constructor on it, I get the set of all the characters that occur in this text. I then call list on that to create a list of those characters, instead of just a set, so that I have an ordering, an arbitrary ordering, and then I sort that.

We get all the characters that occur in the entire dataset and they're sorted. The number of them is going to be our vocabulary size. These are the possible elements of our sequences. When I print the characters, there are 65 of them in total. There's a space character, all kinds of special characters, and then capitals and lowercase letters. That's our vocabulary and those are the possible characters that the model can see or emit.

Next, we would like to develop some strategy to tokenize the input text. When people say "tokenize", they mean convert the raw text as a string to some sequence of integers according to some vocabulary of possible elements. As an example, we are going to be building a character-level language model, so we're simply going to be translating individual characters into integers.

Let me show you a chunk of code that does that for us. We're building both the encoder and the decoder. When we encode an arbitrary text like "hi there", we're going to receive a list of integers that represents that string, for example, 46, 47, etc. We also have the reverse mapping so we can take this list and decode it to get back the exact same string. It's really just like a translation to integers and back for an arbitrary string, and for us, it is done on a character level.

The way this was achieved is we just iterate over all the characters here and create a lookup table from the character to the integer and vice versa. To encode some string, we simply translate all the characters individually and to decode it back, we use the reverse mapping and concatenate all of it.

This is only one of many possible encodings or tokenizers and it's a very simple one. There are many other schemas that people have come up with in practice. For example, Google uses a sentence piece. Sentence piece will also encode.
# Text Encoding and Tokenization in Natural Language Processing

In Natural Language Processing (NLP), we often need to convert text into integers. This process can be done in different ways, using different schemas and vocabularies. One such method is the use of a sub-word tokenizer, such as SentencePiece. 

A sub-word tokenizer does not encode entire words, nor does it encode individual characters. Instead, it operates at a sub-word unit level. This approach is commonly adopted in practice. For example, OpenAI has a library called TickToken that uses a byte pair encoding tokenizer, which is what GPT (Generative Pretrained Transformer) uses. 

You can also encode words into a list of integers. For instance, the phrase "hello world" can be encoded into a list of integers. As an example, using the TickToken library, you can get the encoding for GPT-2. Instead of having 65 possible characters or tokens, GPT-2 has 50,000 tokens. So, when you encode the phrase "Hi there", you get a list of three integers. However, these integers are not between 0 and 64, they are between 0 and 50,256.

Essentially, you can trade-off the codebook size and the sequence lengths. You can have very long sequences of integers with very small vocabularies, or you can have short sequences of integers with very large vocabularies. Typically, sub-word encodings are used in practice. However, for simplicity, we will use a character-level tokenizer in this discussion. 

Using a character-level tokenizer means we have very small codebooks and very simple encode and decode functions. However, we do get very long sequences as a result. But for the purpose of this lecture, we will stick to this level because it's the simplest.

Now that we have an encoder and a decoder, effectively a tokenizer, we can tokenize the entire training set of Shakespeare. To do this, we will use the PyTorch library, specifically the `torch.tensor` from the PyTorch library. We will take all of the text in Tiny Shakespeare, encode it, and then wrap it into a `torch.tensor` to get the data tensor. 

The data tensor is a massive sequence of integers. This sequence of integers is an identical translation of the characters in the text. For example, zero might be a newline character and one might be a space. From now on, the entire dataset of text is represented as a single, very large sequence of integers.

Before we move on, let's separate our dataset into a train and a validation split. We will take the first 90% of the dataset as the training data for the Transformer and withhold the last 10% as the validation data. This will help us understand to what extent our model is overfitting. We don't want a perfect memorization of this exact Shakespeare text, we want a neural network that creates Shakespeare-like text. Therefore, it should be fairly likely for it to produce the actual stowed away Shakespeare text. We will use this to get a sense of the overfitting.

Finally, we would like to start plugging these text sequences, or integer sequences, into the Transformer so that it can train and learn those patterns. However, it's important to realize that we're never going to actually feed the entire text into the Transformer all at once.
# Training Transformers with Chunks of Data

Training a Transformer on large datasets can be computationally expensive and prohibitive. To overcome this, we work with chunks of the dataset. We sample random chunks from the training set and train them a few at a time. These chunks have a certain length, often referred to as the 'block size' or 'context length'.

Let's consider a block size of eight and look at the first nine characters in the training set sequence. Why nine and not eight? I'll explain in a moment.

When you sample a chunk of data like this, it actually contains multiple examples. This is because all of these characters follow each other. So, when we plug this into a Transformer, we simultaneously train it to make predictions at every one of these positions.

In a chunk of nine characters, there are actually eight individual examples packed in there. For instance, in the context of '18 47', '56' is likely to come next. In the context of '18 47 56', '57' can come next, and so on.

Let's illustrate this with some code. The inputs to the Transformer, denoted as 'X', will be the first block size characters. The targets, denoted as 'Y', will be the next block size characters, offset by one. This is because 'Y' are the targets for each position in the input.

We iterate over all the block size of eight. The context is always all the characters in 'X' up to and including 't', and the target is always the 't' character but in the targets array 'Y'. Running this spells out the eight examples hidden in a chunk of nine characters that we sampled from the training set.

We train on all the eight examples here with context between one all the way up to the block size. This is not just done for computational efficiency, but also to make the Transformer Network accustomed to seeing contexts all the way from as little as one all the way to block size.

This is useful during inference because while we're sampling, we can start the sampling generation with as little as one character of context. The Transformer knows how to predict the next character with all the way up to just one context of one. Then it can predict everything up to block size. After block size, we have to start truncating because the Transformer will never receive more than block size inputs when it's predicting the next character.

So far, we've looked at the time dimension of the tensors that we'll be feeding into the Transformer. There's one more dimension to care about, and that is the batch dimension. As we're sampling these chunks of text, we're going to be feeding them into a Transformer every time.
# Understanding Transformers and Batching

In the context of Transformers, we're going to have many batches of multiple chunks of text. These chunks are all stacked up in a single tensor. This is done for efficiency, to keep the GPUs busy because they are very good at parallel processing of data. We want to process multiple chunks all at the same time. However, these chunks are processed completely independently; they don't communicate with each other.

Let's generalize this concept and introduce a batch dimension. Here's a chunk of code. We're going to start sampling random locations in the dataset to pull chunks from. To ensure reproducibility, I am setting the seed in the random number generator. This way, the numbers I see here will be the same numbers you see later if you try to reproduce this.

The batch size here is how many independent sequences we are processing every forward and backward pass of the Transformer. The block size, as I explained, is the maximum context length to make those predictions. Let's say the batch size is 4 and the block size is 8. Here's how we get a batch for any arbitrary split. If the split is a training split, then we're going to look at train data, otherwise, we'll look at validation data.

That gets us the data array. When I generate random positions to grab a chunk out of, I actually generate a batch size number of random offsets. So, because this is four, we are going to have four numbers that are randomly generated between 0 and the length of data minus block size. These are just random offsets into the training set.

The inputs, denoted as 'X', are the first block size characters starting at the offset. The targets, denoted as 'Y', are offset by one from that. We're going to get those chunks for every one of the integers in the offsets and use `torch.stack` to take all those one-dimensional tensors and stack them up as rows. They all become a row in a four by eight tensor.

The inputs to the Transformer now are a four by eight tensor, with four rows of eight columns. Each one of these is a chunk of the training set. The targets are in the associated array 'Y' and they will come in through the Transformer all the way at the end to create the loss function. They will give us the correct answer for every single position inside 'X'.

These four by eight arrays contain a total of 32 examples and they're completely independent as far as the Transformer is concerned. For instance, when the input is 24, the target is 43. When the input is 24, 43, 58, the target is 58. You can see this spelled out. These are the 32 independent examples packed into a single batch of the input 'X' and then the desired targets are in 'Y'.

This integer tensor of 'X' is going to feed into the Transformer. The Transformer is going to simultaneously process all these examples and then look up the correct integers to predict in every one of these positions in the tensor 'Y'.

Now that we have our batch of input that we'd like to feed into a Transformer, let's start feeding this into the model.
# Implementing Neural Networks with PyTorch

In this tutorial, we're going to start off with the simplest possible neural network, which, in the case of language modeling, is the bigram language model. We've covered the bigram language model in depth in my previous series, so here, we're going to move a bit faster and directly implement the PyTorch module that implements the bigram language model.

First, we import PyTorch and the module for reproducibility. Then, we construct a bigram language model, which is a subclass of the `nn.Module`. We call it by passing in the inputs and the targets. 

When the inputs and targets come in, we take the index of the inputs (which we rename to `idx`) and pass them into the token embedding table. 

In the constructor, we create a token embedding table of size `vocab_size` by `vocab_size`. We use `nn.Embedding`, which is a thin wrapper around a tensor of shape `vocab_size` by `vocab_size`. 

When we pass `idx` here, every single integer in our input refers to this embedding table and plucks out a row of that embedding table corresponding to its index. For example, 24 will go to the embedding table and pluck out the 24th row, and 43 will go and pluck out the 43rd row, etc. 

PyTorch arranges all of this into a batch by time by channel tensor. In this case, the batch is `vocab_size` or 65. We pluck out all those rows and arrange them in a `b` by `t` by `c` tensor. 

We interpret this as the logits, which are basically the scores for the next character in the sequence. We are predicting what comes next based on just the individual identity of a single token. 

Currently, the tokens are not talking to each other and they're not seeing any context except for themselves. For example, if I'm token number five, I can make pretty decent predictions about what comes next just by knowing that I'm token five. Some characters naturally follow other characters in typical scenarios. 

Once we've made predictions about what comes next, we'd like to evaluate the loss function. A good way to measure the quality of the predictions is to use the negative log-likelihood loss, which is also implemented in PyTorch under the name cross-entropy. 

We'd like to measure the loss as the cross-entropy on the predictions and the targets. This measures the quality of the logits with respect to the targets. In other words, we have the identity of the next character, so how well are we predicting the next character based on the logits? 

Intuitively, the correct dimension of logits, depending on whatever the target is, should have a very high number and all the other dimensions should be very low. However, this won't actually run as it is and will return an error message. But intuitively, this is what we want to measure.
# PyTorch Cross Entropy

In this tutorial, we are going to discuss the PyTorch Cross Entropy. We will be calling the cross entropy in its functional form, which means we don't have to create a module for it. 

## Understanding PyTorch Cross Entropy

When we go to the [PyTorch Cross Entropy documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), we have to look into the details of how PyTorch expects these inputs. The issue here is that PyTorch expects multi-dimensional input, which we do have because we have a B by T by C tensor. It actually wants the channels to be the second dimension here. So, it wants a B by C by T instead of a B by T by C. 

## Reshaping Logits

We need to reshape our logits. Here's what I like to do: I like to take the dimensions of logits.shape, which is B by T by C, and unpack those numbers. Then, let's say that logits equals logits.view, and we want it to be a B times T by C, just a two-dimensional array. 

We're going to take all of these positions here and stretch them out in a one-dimensional sequence, preserving the channel dimension as the second dimension. We're just kind of stretching out the array so it's two-dimensional. In that case, it's going to better conform to what PyTorch expects in its dimensions.

## Reshaping Targets

We have to do the same to targets because currently, targets are of shape B by T and we want it to be just B times T, so one dimensional. Alternatively, you could always still just do -1 because PyTorch will guess what this should be if you want to lay it out. But let me just be explicit and say B times T. Once we've reshaped this, it will match the cross entropy case, and then we should be able to evaluate.

## Evaluating the Loss

With that, we can do loss. Currently, we see that the loss is 4.87. Now, because we have 65 possible vocabulary elements, we can actually guess at what the loss should be. In particular, we covered negative log likelihood in a lot of detail. We are expecting log or long of 1 over 65 and negative of that. So we're expecting the loss to be about that. That's telling us that the initial predictions are not super diffuse. They've got a little bit of entropy and so we're guessing wrong. But actually, we are able to evaluate the loss.

## Generating from the Model

Now that we can evaluate the quality of the model on some data, we'd likely also be able to generate from the model. Let's do the generation now. 

We take some input idx here. This is the current context of some characters in a batch. So it's also B by T. The job of generate is to take this B by T and extend it to be B by T plus one, plus two, plus three. So it's just basically it contains the generation in all the batch dimensions in the time dimension. That's its job and we'll do that for max new tokens. 

In conclusion, whatever is predicted is the result of the generation from the model.
# Text Generation with PyTorch

In this article, we will discuss the process of text generation using PyTorch. We will start by concatenating on top of the previous index along the first dimension, which is the time dimension, to create a `b` by `T+1`. 

The job of the generator is to take a `b` by `T` and transform it into a `b` by `T+1`, `T+2`, `T+3`, and so on, depending on the maximum tokens we want. This is the generation process from the model.

Inside the generation process, we take the current indices and get the predictions, which are in the logits. The loss here is ignored because we are not using it and we have no targets or ground truth targets to compare with.

Once we get the logits, we focus only on the last step. Instead of a `b` by `T` by `C`, we pluck out the last element in the time dimension because those are the predictions for what comes next. We then convert these logits to probabilities via softmax.

We use `torch.multinomial` to sample from these probabilities and ask PyTorch to give us one sample. The `idx_next` will become a `b` by `1` because in each one of the batch dimensions, we will have a single prediction for what comes next. The `num_samples=1` will make this be a one.

We then take those integers that come from the sampling process according to the probability distribution given here. These integers get concatenated on top of the current running stream of integers, giving us a `b` by `T+1`. We can then return that.

One thing to note here is that when calling `self` of `idx`, which will end up going to the forward function, we are not providing any targets. This would give an error because targets are not given, so targets have to be optional. If targets are none, there's no loss to create, so it's just loss is none. But if targets are provided, all of this happens and we can create a loss.

This will generate from the model. Let's create a batch where `b` is just one and `time` is just one. We create a little one by one tensor and it's holding a zero. The data type is integer. Zero is going to be how we kick off the generation and remember that zero is the element standing for a new line character. It's a reasonable thing to feed in as the very first character in a sequence.

We feed in `idx` and ask for 100 tokens. The `generate` function will continue that. Because `generate` works on the level of batches, we then have to index into the zeroth row to unplug the single batch dimension that exists. That gives us a one-dimensional array of all the indices which we convert to a simple Python list from a PyTorch tensor. This can feed into our decode function and convert those integers into text.
# Training a Character-Level Language Model

In this tutorial, we will be generating 100 tokens and running a character-level language model. The initial output might seem like gibberish, but that's because we are using a completely random model. The next step is to train this model to improve its performance.

One thing to note here is that the function we are using is written to be general. However, it might seem a bit overkill at this point because we are feeding in a lot of context and concatenating it all, even though we are always feeding it all into the model. This might seem unnecessary for a simple background model. For instance, to make a prediction about 'K', we only needed 'W'. But what we did was feed the entire sequence into the model and then only looked at the very last piece to predict 'K'.

The reason for this approach is that while this is a bigram model, we want to keep this function fixed and make it work for future scenarios where our character might need to look further into the history. So, while the history is not used right now and this approach might seem silly, the history will eventually be used, and that's why we want to do it this way.

Now that we have seen that the initial output is random, let's train the model to make it less random.

## Training the Model

First, we will create a PyTorch optimization object. We are using the optimizer 'AdamW'. In the 'Make More' series, we have only ever used stochastic gradient descent, which is the simplest possible optimizer. But for this tutorial, we want to use 'Adam', which is a much more advanced and popular optimizer. It works extremely well for typical settings. A good learning rate is roughly negative three or even higher.

We will create the optimizer object, which will take the gradients and update the parameters using the gradients. Our batch size was only four, so let's increase it to 32. For a certain number of steps, we will sample a new batch of data, evaluate the loss, zero out all the gradients from the previous step, get the gradients for all the parameters, and then use those gradients to update our parameters. This is a typical training loop, as we saw in the 'Make More' series.

Let's run this for 100 iterations. We started around 4.7, and now we are down to 4.6. The optimization is definitely happening, but we need to increase the number of iterations and only print at the end because we probably will need to train for longer.

Let's run this for 10,000 iterations. We are expecting something a bit more reasonable. We are down to about 2.5, which is a significant improvement. Let's see what we get.

The output is a dramatic improvement on what we had initially. Let's increase the number of tokens. We are starting to get something that looks like a more reasonable output. Of course, it's not going to be Shakespeare from a background model, but at least we see that the loss is improving.
# Building a Simple Language Model

The model we've been working on is reasonable-ish. It's certainly not Shakespeare, but it's making progress. This is the simplest possible model. 

Now, what I'd like to do is to make this model more complex. This is a very simple model because the tokens are not talking to each other. Given the previous context of whatever was generated, we're only looking at the very last character to make predictions about what comes next. 

These tokens have to start talking to each other and figuring out what is in the context so that they can make better predictions for what comes next. This is how we're going to kick off the Transformer.

Next, I took the code that we developed in this Jupyter notebook and converted it to be a script. I'm doing this because I want to simplify our intermediate work into just the final product that we have at this point. 

At the top of the script, I put all the hyperparameters that we've defined. I introduced a few new ones, which I'm going to speak to in a little bit. Otherwise, a lot of this should be recognizable: reproducibility, reading data, getting the encoder and the decoder, creating the training and test splits. 

I used a data loader that gets a batch of the inputs and targets. This is new and I'll talk about it in a second. 

Now, this is the background language model that we developed. It can forward and give us logits and loss, and it can generate. 

Then, we are creating the optimizer and this is the training loop. Everything here should look pretty familiar. 

Now, some of the small things that I added. 

Number one, I added the ability to run on a GPU if you have it. If you have a GPU, then this will use CUDA instead of just CPU and everything will be a lot faster. Now, when the device becomes CUDA, then we need to make sure that when we load the data, we move it to the device. When we create the model, we want to move the model parameters to the device. 

For example, we have the NN embedding table and it's got a double weight inside it which stores the lookup table. That would be moved to the GPU so that all the calculations here happen on the GPU and they can be a lot faster. 

Finally, when I'm creating the context that feeds into generate, I have to make sure that I create it on the device. 

Number two, what I introduced is the fact that here in the training loop, I was just printing the loss.item inside the training loop. But this is a very noisy measurement of the current loss because every batch will be more or less lucky. 

So, what I want to do usually is to have an estimate loss function. The estimated loss basically then goes up here and it averages up the loss over multiple batches. In particular, we're going to iterate in valid iterations times and we're going to get our loss and then we're going to get the average loss for both splits. This will be a lot less noisy. 

So here, when we call the estimate loss, we're going to report the pretty accurate train and validation loss. 

Now, when we come back up, you'll notice a few things. Here, I'm setting the model to evaluation phase and down here, I'm resetting it back to training phase. Right now, for our model as is, this doesn't actually do anything because the only thing inside this model is this NN.embedding and this network would behave the same in both evaluation mode and training mode.
# Understanding PyTorch and Neural Networks

In this script, we don't have any dropout layers or batch normalization layers. However, it's good practice to think through what mode your neural network is in because some layers will have different behavior at inference time or training time. 

There's also this context manager, `torch.nograd`. This is just telling PyTorch that everything that happens inside this function, we will not call `backward` on. So, PyTorch can be a lot more efficient with its memory use because it doesn't have to store all the intermediate variables since we're never going to call `backward`. It can be a lot more memory efficient in that way. It's also a good practice to tell PyTorch when we don't intend to do back propagation.

The script is about 120 lines of code, and that's our starter code. I'm calling it `background.py` and I'll release it later. Running this script gives us output in the terminal. It provides the train loss and validation loss, and we see that we converge to somewhere around 2.5 with the migrant model. Here's the sample that we produced at the end. We have everything packaged up in the script and we're in a good position now to iterate on this.

We are almost ready to start writing our very first self-attention block for processing these tokens. Before we actually get there, I want to get you used to a mathematical trick that is used in the self-attention inside a Transformer. It's really at the heart of an efficient implementation of self-attention. 

Let's create a `b x t x c` where `b`, `t`, and `c` are just 4, 8, and 2 in this toy example. These are basically channels, and we have batches and we have the time component. We have some information at each point in the sequence, `c`.

We would like these tokens to communicate with each other. We have up to eight tokens here in a batch and these eight tokens are currently not talking to each other. We would like to couple them. In particular, we don't want the token at the fifth location to communicate with tokens in the sixth, seventh, and eighth location because those are future tokens in the sequence. The token on the fifth location should only talk to the one in the fourth, third, second, and first. So, information only flows from previous context to the current timestamp and we cannot get any information from the future because we are about to try to predict the future.

The easiest way for tokens to communicate is to do an average of all the preceding elements. For example, if I'm the fifth token, I would like to take the channels that make up the information at my step but also the channels from the fourth, third, second, and first step. I'd like to average those up and then that would become a feature vector that summarizes me in the context of my history. Of course, just doing a sum or an average is an extremely weak form of communication.
# Efficient Communication in Neural Networks

Interaction in communication is extremely lossy. We've lost a ton of information about the spatial arrangements of all those tokens. But that's okay for now. We'll see how we can bring that information back later. 

For now, what we would like to do is, for every single batch element independently, for every 't' token in that sequence, we'd like to now calculate the average of all the vectors in all the previous tokens and also at this token. 

Let's write that out. I have a small snippet here and instead of just fumbling around, let me just copy-paste it and talk to it. In other words, we're going to create 'X', and 'bow' is short for 'bag of words'. 

'Bag of words' is a term that people use when you are just averaging up things. So it's just a bag of words. Basically, there's a word stored on every one of these eight locations and we're doing a bag of words such as averaging. 

In the beginning, we're going to say that it's just initialized at zero. Then I'm doing a for loop here so we're not being efficient yet, that's coming. But for now, we're just iterating over all the batch dimensions independently, iterating over time. 

The previous tokens are at this batch dimension and then everything up to and including the 't' token. So when we slice out 'X' in this way, 'xrev' becomes of shape, how many 'T' elements there were in the past and then of course 'C'. So all the two-dimensional information from these log tokens. 

That's the previous chunk of tokens from my current sequence. Then I'm just doing the average or the mean over the zeroth dimension. So I'm averaging out the time here and I'm just going to get a little 'C' one-dimensional vector which I'm going to store in 'X bag of words'. 

This is not going to be very informative because let's see, this is 'x sub 0'. So this is the zeroth batch element and then 'exp0' at zero now. You see how at the first location here, you see that the two are equal and that's because we're just doing an average of this one token. But here, this one is now an average of these two, and now this one is an average of these three, and so on. 

This last one is the average of all of these elements. So vertical average just averaging up all the tokens now gives this outcome here. This is all well and good but this is very inefficient. Now the trick is that we can be very efficient about doing this using matrix multiplication. 

That's the mathematical trick and let me show you what I mean. Let's work with a toy example here. I have a simple matrix here that is a three by three of all ones, a matrix 'B' of just random numbers and it's a three by two, and a matrix 'C' which will be three by three multiply three by two which will give out a three by two. 

Here we're just using matrix multiplication. So 'A' multiply 'B' gives us 'C'. How are these numbers in 'C' achieved? This number in the top left is the first row of 'A' dot product with the first column of 'B'. Since all the row of 'A' right now is all just ones, then the dot product here with this column of 'B' is just going to do a sum of these of this column. So 2 plus 6 plus 6 is 14. The element here and the output of 'C' is.
# Understanding Matrix Multiplication in PyTorch

In this article, we will explore the concept of matrix multiplication in PyTorch, a popular open-source machine learning library. We will delve into the intricacies of matrix multiplication, and how we can manipulate matrices to perform complex operations such as averaging rows incrementally.

Let's start by examining a simple matrix multiplication operation. Consider two matrices, A and B. The first element of the resulting matrix C is obtained by multiplying the first row of A with the first column of B. The sum of these products gives us the first element of C.

This process is repeated for each element in C. For instance, the element at the second row and first column of C is obtained by multiplying the second row of A with the first column of B. The sum of these products gives us the corresponding element in C.

Now, let's consider a scenario where the first row of A consists of all ones. In this case, the resulting elements in C are simply the sum of the elements in the corresponding columns of B. This is because each element in the first row of A is 1, and any number multiplied by 1 is the number itself.

PyTorch provides a function called `tril`, which stands for 'triangular lower'. This function returns the lower triangular part of a matrix, effectively zeroing out the elements above the main diagonal. This can be useful in certain matrix multiplication scenarios.

For instance, consider a matrix A where the elements above the main diagonal are zero. When we multiply this matrix with another matrix B, the resulting matrix C will have some interesting properties. The first row of C is simply the first row of B, as the zeros in A cause the other elements to be ignored.

The second row of C is the sum of the first two rows of B, and so on. In essence, depending on the number of ones and zeros in each row of A, we are performing a variable sum of the rows of B.

This concept can be extended to calculate the average of the rows of B. By normalizing the rows of A such that they sum to one, the resulting matrix C will contain the average of the rows of B. This can be achieved by dividing each row of A by its sum.

In conclusion, by manipulating the elements of the multiplying matrix, we can perform complex operations such as incremental averaging during matrix multiplication. This is a powerful technique that can greatly enhance the efficiency of our computations in PyTorch.
# Understanding Matrix Multiplication in PyTorch

In this article, we will be discussing the concept of matrix multiplication in PyTorch, specifically focusing on batched matrix multiplication and softmax. We will be using a simple example to illustrate these concepts.

Let's start with the concept of weights. We have a matrix 'A' and we want to determine how much of every row we want to average up. This is an average because the sum of the rows equals 1. In our example, our matrix 'B' is 'X'.

What happens next is that we will have an 'Expo 2'. This 'Expo 2' is where we multiply 'RX'. Let's think this through. 'T' is a matrix and we are performing matrix multiplication in PyTorch with 'A' and 'B', which are 'T' by 'C'. 

PyTorch will see that these shapes are not the same, so it will create a batch dimension. This is a batched matrix multiplication and it will apply this matrix multiplication to all the batch elements in parallel and individually. For each batch element, there will be a 'T' by 'T' multiplying 'T' by 'C', exactly as we had below. This will now create 'B' by 'T' by 'C' and 'X' both 2 will now become identical to 'Expo'.

We can verify this by checking if 'Expo' and 'Expo 2' are close. If they are, it means that they are in fact the same. If we print the zeroth element of 'Expo' and 'Expo 2', we should see that they are identical, which they are.

The trick here is that we were able to use batched matrix multiplication to do this aggregation. It's a weighted aggregation and the weights are specified in this 'T' by 'T' array. We're basically doing weighted sums according to the weights inside here. They take on a triangular form, which means that a token at the 'T' dimension will only get information from the tokens preceding it. That's exactly what we want.

Finally, let's rewrite it in one more way. This third version is also identical to the first and second, but it uses softmax. 'Trill' here is a matrix of lower triangular ones. 'Way' begins as all zero. Then, we use 'masked fill'. What this does is, for all the elements where 'Trill' is equal to zero, it makes them be negative infinity. 

If we take a softmax along every single row (dim is negative one), what is that going to do? Softmax is a normalization operation. If we apply softmax, we get the exact same matrix. In softmax, we're going to exponentiate every single one of these elements. 

In conclusion, understanding matrix multiplication in PyTorch, specifically batched matrix multiplication and softmax, is crucial for deep learning. It allows us to perform complex operations efficiently and effectively.
# Understanding Self-Attention Mechanism

First, we're going to divide by the sum. If we exponentiate every single element here, we're going to get a one, and everywhere else, we're going to get basically zero. When we normalize, we just get one, and then softmax will again divide, and this will give us 0.5. This is also the same way to produce this mask.

The reason that this is a bit more interesting, and the reason we're going to end up using it in self-attention, is that these weights here begin with zero. You can think of this as an interaction strength or an affinity. Basically, it's telling us how much of each token from the past do we want to aggregate and average up. This line is saying tokens from the past cannot communicate by setting them to negative infinity, we're saying that we will not aggregate anything from those tokens.

This then goes through softmax and through the weighted aggregation through matrix multiplication. You can think of these zeros as currently just set by us to be zero, but a quick preview is that these affinities between the tokens are not going to be just constant at zero, they're going to be data-dependent. These tokens are going to start looking at each other, and some tokens will find other tokens more or less interesting. Depending on what their values are, they're going to find each other interesting to different amounts, which I'm going to call affinities.

Here, we are saying the future cannot communicate with the past, we're going to clamp them. When we normalize and sum, we're going to aggregate their values depending on how interesting they find each other. That's the preview for self-attention. 

In short, from this entire section, you can do weighted aggregations of your past elements by using matrix multiplication of a lower triangular fashion. The elements in the lower triangular part are telling you how much of each element fuses into this position. We're going to use this trick now to develop the self-attention block.

## Preliminaries

First, the thing I'm bothered by is that we're passing in vocab size into the constructor. There's no need to do that because vocab size is already defined up top as a global variable, so there's no need to pass this stuff around.

Next, I want to create a level of interaction where we don't directly go to the embedding for the logits but instead we go through this intermediate phase because we're going to start making that bigger. Let me introduce a new variable, `n_embed`, short for a number of embedding dimensions. `n_embed` here will be 32, which is a good number suggested by GitHub.

This is an embedding table and only then here, this is not going to give us logits directly. Instead, this is going to give us token embeddings. To go from the token embeddings to the logits, we're going to need a linear layer. So, `self.lm_head`, let's call it, short for language modeling head, is `nn.Linear` from `n_embed` up to `vocab_size`.
# Understanding Self-Attention Mechanism in Deep Learning

In this article, we will delve into the self-attention mechanism, a crucial component in deep learning models. We will start by examining the logits and then gradually build on top of that.

When we swing over to the logits, we need to be careful because the two 'C's we encounter are not equal. The first 'C' is an embedded 'C', while the second 'C' represents the vocabulary size. For simplicity, let's say that an embed is equal to 'C'. This creates a spurious layer of interaction through a linear.

At first glance, this might seem spurious, but we will build on top of this. So far, we've taken these 'C's and encoded them based on the identity of the tokens inside `idx`. The next thing that people often do is not just encoding the identity of these tokens but also their position.

We will have a second position, a position embedding table. This table is an embedding of block size by an embed. Each position from 0 to block size minus 1 will also get its own embedding vector.

Let's decode a 'b' by 'T' from `idx.shape`. We will also have a positional embedding, which is the positional embedding from the table. This is a torch arrange, essentially integers from 0 to 'T' minus 1. All of these integers from 0 to 'T' minus 1 get embedded through the table to create a 't' by 'C'.

Let's rename this to 'x', which will be the addition of the token embeddings with the positional embeddings. The broadcasting note will work out so 'B' by 'T' by 'C' plus 'T' by 'C' gets right aligned, a new dimension of one gets added, and it gets broadcasted across the batch.

At this point, 'x' holds not just the token identities but the positions at which these tokens occur. This is currently not that useful because we just have a simple model, so it doesn't matter if you're in the fifth position, the second position, or wherever. It's all translation invariant at this stage, so this information currently wouldn't help. But as we work on the self-potential block, we'll see that this starts to matter.

Now we get to the crux of self-attention, probably the most important part of this article. We're going to implement a small self-attention for a single individual head.

We start off with where we were, with all of this code being familiar. I'm working with an example where I changed the number of channels from 2 to 32. We have a 4x8 arrangement of tokens, and each token's information is currently random numbers.

The code, as we had it before, does a simple average of all the past tokens and the current token. It's just the previous information and current information being mixed together in an average. It does so by creating a lower triangular structure which allows us to mask out this weight matrix that we create.

We mask it out and then normalize it. When we initialize the affinities between all the different tokens or nodes (I'm going to use those terms interchangeably), to be zero, we see that it gives us this structure where every single row has these uniform numbers. That's what then, in this matrix.
# Understanding Self-Attention Mechanism in Deep Learning

In deep learning, the self-attention mechanism is a critical component that allows us to gather information from the past in a data-dependent way. This mechanism is particularly useful when dealing with sequences of tokens, where different tokens may find other tokens more or less interesting. 

For instance, if we consider a vowel as a token, it might be looking for consonants in its past. It would want to know what those consonants are and would want that information to flow to it. This is where the self-attention mechanism comes into play.

## How Self-Attention Works

Every single token at each position emits two vectors - a query and a key. The query vector, roughly speaking, represents what the token is looking for. On the other hand, the key vector represents what the token contains. 

The affinities between these tokens in a sequence are determined by performing a dot product between the keys and the queries. If the key and the query are aligned, they will interact to a high degree, allowing the token to learn more about that specific token as opposed to any other token in the sequence.

## Implementing Self-Attention

In this section, we will implement a single head of self-attention. The head size is a hyperparameter involved with these heads. We initialize the linear modules using `bias = false`, which means these modules will just apply a matrix multiply with some fixed weights.

Next, we produce a key (`K`) and a query (`Q`) by forwarding these modules on `x`. The size of this will now become `B x T x 16` because that is the head size. When we forward this linear on top of our `x`, all the tokens in all the positions in the `B x T` arrangement produce a key and a query in parallel and independently. No communication has happened yet.

The communication comes when all the queries dot product with all the keys. We want the affinities between these to be the result of the query multiplying the key. However, we need to be careful with matrix multiplication. We need to transpose `K`, but we also need to consider the batch dimension. In particular, we want to transpose the last two dimensions, dimension `-1` and dimension `-2`.

This matrix multiplication will give us a `T x T` matrix for every row of `B`, providing us with the affinities. These affinities are now derived from the dot product between the keys and the queries. This can now run, and the weighted aggregation is now a function in a data-dependent manner between the keys and queries of these nodes.

Inspecting the result, we see that the weights take on a different form for each batch element because each batch element contains different tokens at different positions. This makes the process data-dependent. When we look at just the zeroth row in the input, these are the affinities we get.
# Understanding Self-Attention Heads

The weights that came out can be seen, and it's clear that they're not just exactly uniform. As an example, let's consider the last row, which was the eighth token. The eighth token knows what content it has and it knows at what position it's in. 

Based on that, the eighth token creates a query. It's as if it's saying, "Hey, I'm looking for this kind of stuff. I'm a vowel, I'm on the eighth position, and I'm looking for any consonants at positions up to four." Then, all the nodes get to emit keys. One of the channels could be, "I am a consonant and I am in a position up to four." That key would have a high number in that specific channel. That's how the query and the key, when they dot product, can find each other and create a high affinity.

When they have a high affinity, like say this token was pretty interesting to the eighth token, through the softmax, the eighth token will end up aggregating a lot of its information into its position. So, it'll get to learn a lot about it. 

Now, let's look at what happens after this has already happened. Let's erase the masking and the softmax just to show you the under-the-hood internals and how that works. Without the masking and the softmax, what comes out is the outputs of the dot products. These are the raw outputs and they take on values from negative two to positive two, etc. These are the raw interactions and affinities between all the nodes.

But now, if I'm a fifth node, I will not want to aggregate anything from the sixth, seventh, and eighth nodes. So, we use the upper triangular masking so those are not allowed to communicate. We also want to have a nice distribution, so we don't want to aggregate negative 0.11 of this node. Instead, we exponentiate and normalize to get a nice distribution that sums to one. This tells us, in a data-dependent manner, how much information to aggregate from any of these tokens in the past. 

There's one more part to a single self-attention head. When you do the aggregation, we don't actually aggregate the tokens exactly. We produce one more value here, and we call that the value. In the same way that we produced the query, we're also going to create a value. We don't aggregate X, we calculate a V which is just achieved by propagating this linear on top of X again. Then, we output the product of the weights and V. V is the vector that we aggregate instead of the raw X. 

The output here of the single head will be 16 dimensional because that is the head size. You can think of X as kind of like private information to this token. For example, I'm a fifth token and I have some identity and my information is kept in Vector X. For the purposes of the single head, here's what I'm interested in, here's what I have, and if you find me interesting, here's what I will communicate to you. That's stored in V. So, V is the thing that gets aggregated.
# Understanding the Self-Attention Mechanism

The self-attention mechanism can be aggregated for the purposes of a single head between different nodes. This is essentially what it does. However, there are a few notes I would like to make about attention.

## Attention as a Communication Mechanism

Firstly, attention is a communication mechanism. You can really think about it as a communication mechanism where you have a number of nodes in a directed graph. Essentially, you have edges pointing between those nodes. Every node has some vector of information and it gets to aggregate information via a weighted sum from all the nodes that point to it. This is done in a data-dependent manner, depending on whatever data is actually stored at each node at any point in time.

Our graph doesn't look like this. Our graph has a different structure. We have eight nodes because the block size is eight and there are always eight tokens. The first node is only pointed to by itself. The second node is pointed to by the first node and itself, all the way up to the eighth node which is pointed to by all the previous nodes and itself. That's the structure that our directed graph has or happens to have in other aggressive scenarios like language modeling. But in principle, attention can be applied to any arbitrary directed graph. It's just a communication mechanism between the nodes.

## No Notion of Space

The second note is that there is no notion of space. Attention simply acts over a set of vectors in this graph. By default, these nodes have no idea where they are positioned in space. That's why we need to encode them positionally and give them some information that is anchored to a specific position so that they know where they are. This is different from, for example, a convolution operation over some input. There's a very specific layout of the information in space and the convolutional filters act in space. In attention, it's just a set of vectors out there in space. They communicate and if you want them to have a notion of space, you need to specifically add it, which is what we've done when we calculated the relative position encodings and added that information to the vectors.

## Independent Examples Across the Batch Dimension

The next thing that I hope is very clear is that the elements across the batch dimension, which are independent examples, never talk to each other. They are always processed independently. This is a batched matrix multiply that applies a matrix multiplication kind of in parallel across the batch dimension. So maybe it would be more accurate to say that in this analogy of a directed graph, we really have four separate pools of eight nodes because the batch size is four. Those eight nodes only talk to each other, but in total, there are 32 nodes that are being processed. There are four separate pools of eight nodes.

## Future Tokens and Past Tokens

Here, in the case of language modeling, we have this specific structure of a directed graph where the future tokens will not communicate to the past tokens. But this doesn't necessarily have to be the constraint in the general case. In fact, in many cases, you may want to have all of the nodes talk to each other fully. As an example, if you're doing sentiment analysis with a Transformer, you might have a number of tokens and you may want to have them all talk to each other fully because later you are predicting the sentiment of the sentence. It's okay for these nodes to talk to each other.
# Understanding Encoder and Decoder Blocks in Self-Attention

In some cases, you will use an encoder block of self-attention. An encoder block is a line of code that you will delete, allowing all the nodes to communicate with each other. What we're implementing here is sometimes called a decoder block. It's called a decoder because it is sort of like decoding language. It's got this auto-aggressive format where you have to mask with the triangular matrix so that nodes from the future never talk to the past because they would give away the answer. 

In encoder blocks, you would delete this line of code to allow all the nodes to talk. In decoder blocks, this line of code will always be present so that you have this triangular structure. Both are allowed and attention doesn't care. Attention supports arbitrary connectivity between nodes.

You might often hear the terms "attention" and "self-attention". There's also something called cross-attention. What's the difference? The reason this attention is self-attention is because the keys, queries, and the values are all coming from the same source, from X. These nodes are self-attending. 

But in principle, attention is much more general than that. For example, in encoder-decoder transformers, you can have a case where the queries are produced from X but the keys and the values come from a whole separate external source, sometimes from encoder blocks that encode some context that we'd like to condition on. The keys and the values will actually come from a whole separate source. These are nodes on the side and here we're just producing queries and we're reading off information from the side. 

Cross-attention is used when there's a separate source of nodes we'd like to pull information from into our nodes. It's self-attention if we just have nodes that would like to look at each other and talk to each other. This attention here happens to be self-attention, but in principle, attention is a lot more general.

In the "Attention is All You Need" paper, we've already implemented attention. Given query, key, and value, we've multiplied the query on the key, we've softmaxed it, and then we are aggregating the values. There's one more thing that we're missing here which is the dividing by one over the square root of the head size. 

This is important because it's a kind of normalization. If you have unit Gaussian inputs, so zero mean unit variance K and Q are unit Gaussian, and if you just do it naively then you see that your weight will be on the order of the head size, which in our case is 16. But if you multiply by one over the square root of the head size, then the variance of the weight will be one, so it will be preserved. 

This is important because the weight here will feed into softmax. It's really important, especially at initialization, that the weight be fairly diffuse. In our case here, we sort of lucked out and the weight had fairly diffuse numbers. The problem is that because of softmax, if the weight takes on very positive and very negative numbers inside it, softmax will actually converge towards one-hot vectors.
# Implementing Softmax to a Tensor

Let's illustrate the application of softmax to a tensor. Suppose we have a tensor of values that are very close to zero. When we apply softmax to this tensor, we get a diffuse output. However, if we take the same tensor and start sharpening it by making the values larger, for example by multiplying these numbers by eight, the softmax output will start to sharpen. In fact, it will sharpen towards the maximum value in the tensor. 

We need to be careful not to make these values too extreme, especially during initialization. Otherwise, softmax will be way too peaky and we'll essentially be aggregating information from a single node. This is not what we want, especially at initialization. Therefore, we use scaling to control the variance at initialization.

## Applying Softmax in Code

Now that we understand the theory, let's apply our softmax knowledge in code. I've created a module called `head` that implements a single head of self-attention. You give it a head size and it creates the key, query, and value linear layers. Typically, people don't use biases in these layers. These are the linear projections that we're going to apply to all of our nodes.

I've also created a variable called `Trill`. This is not a parameter of the module. In Python conventions, this is called a buffer. It's not a parameter and you have to assign it to the module using `register_buffer`. This creates the lower triangular matrix.

When we're given the input `X`, we calculate the keys and the queries. We calculate the attention scores inside, normalize it using scaled attention, and ensure that a feature doesn't communicate with the past. This makes it a decoder block. We then apply softmax and aggregate the value to output.

In the language model, I've created a head in the constructor and called it `self_attention_head`. The head size is the same as the embed size. Once we've encoded the information with the token embeddings and the position embeddings, we feed it into the self-attention head. The output of that goes into the decoder language modeling head and creates the logits. This is the simplest way to plug a self-attention component into our network.

## Changes to the Generate Function

I had to make one more change to the `generate` function. We have to ensure that our `idx` that we feed into the model never exceeds the block size. This is because we're now using positional embeddings. If `idx` is more than block size, our position embedding table will run out of scope because it only has embeddings for up to block size. Therefore, I added some code to crop the context that we're going to feed into self, so that we never pass in more than block size elements.

## Training the Network

I also updated the script to decrease the learning rate because self-attention can't tolerate very high learning rates. I also increased the number of iterations because the learning rate is lower. After training, we were able to achieve a lower loss than before. Previously, we were only able to get to up to 2.5, but with these changes, we were able to improve on that.
# Implementing Multi-Head Attention in PyTorch

In our previous discussion, we saw an improvement in our model's performance, with the validation loss going down to 2.4. However, we still have a long way to go. The next step in our journey is implementing multi-head attention, as described in the "Attention is All You Need" paper.

## What is Multi-Head Attention?

Multi-head attention is the process of applying multiple attentions in parallel and concatenating the results. The paper provides a diagram to illustrate this, but to simplify, it's just multiple attentions running in parallel.

## Implementing Multi-Head Attention

Implementing multi-head attention in PyTorch is fairly straightforward. If we want multi-head attention, we need multiple heads of self-attention running in parallel. We can achieve this by creating multiple heads, determining the head size of each, running all of them in parallel, and then concatenating all of the outputs over the channel dimension.

In our case, instead of having a single attention with a head size of 32 (as our embed size is 32), we now have four communication channels running in parallel. Each of these channels will be smaller, so we have four 8-dimensional self-attention vectors that concatenate to give us 32, which matches our original embed size.

This approach is similar to group convolutions in the sense that instead of having one large convolution, we have convolutional groups. This is what we refer to as multi-headed self-attention.

After implementing multi-headed self-attention and running the model, we see an improvement in the validation loss, which is now down to approximately 2.28. The output generation is still not perfect, but the validation loss is improving.

## The Benefits of Multi-Head Attention

Having multiple communication channels is beneficial because tokens have a lot to communicate. They need to find consonants, vowels, and other elements from certain positions. By creating multiple independent channels of communication, we can gather lots of different types of data and then decode the output.

## The Next Steps

Looking back at the "Attention is All You Need" paper, we can see that we've already implemented several components, including positional encodings, token encodings, and masked multi-headed attention. The paper also mentions a feed-forward part, which is a simple multi-layer perceptron (MLP). This is something we will look at implementing in the future.

In conclusion, multi-head attention is a powerful tool that can improve the performance of our model by allowing for multiple parallel communication channels. By implementing this in PyTorch, we've taken another step towards improving our model's performance.
# Implementing Computation into the Network

In a similar fashion, we are also adding computation into the network. This computation is on a per-node level. I've already implemented it and you can see the differences highlighted on the left, where I've added or changed things.

Before, we had the multi-headed self-attention that did the communication, but we went way too fast to calculate the logits. The tokens looked at each other but didn't really have a lot of time to think about what they found from the other tokens.

So, what I've implemented here is a little feed-forward single layer. This little layer is just a linear followed by a relative nonlinearity, and that's it. It's just a little layer and then I call it feed-forward and embed. This feed-forward is just called sequentially right after the self-attention. So, we self-attend, then we feed-forward.

You'll notice that the feed-forward here, when it's applying linear, is on a per-token level. All the tokens do this independently. The self-attention is the communication and then once they've gathered all the data, now they need to think about that data individually. That's what feed-forward is doing and that's why I've added it here.

Now, when I train this, the validation loss actually continues to go down. It still looks kind of terrible, but at least we've improved the situation. As a preview, we're going to now start to intersperse the communication with the computation. That's also what the Transformer does when it has blocks that communicate and then compute, and it groups them and replicates them.

## Implementing Blocks

Let me show you what we'd like to do. We'd like to do something like this: we have a block and this block is basically this part here, except for the cross-attention. The block basically intersperses communication and then computation. The communication is done using multi-headed self-attention and then the computation is done using the feed-forward network on all the tokens independently.

You'll notice that this takes the number of embeddings, the embedding dimension, and the number of heads that we would like, which is kind of like group size in group convolution. I'm saying that the number of heads we'd like is four. So, because this is 32, we calculate that the number of heads should be four. The head size should be eight so that everything sort of works out channel-wise. This is how the Transformer structures the sizes typically.

This is how we want to intersperse them. Here, I'm trying to create blocks which is just a sequential application of block, block, so that we're interspersing communication feed-forward many times and then finally we decode.

I actually tried to run this and the problem is this doesn't actually give a very good result. The reason for that is we're starting to actually get a pretty deep neural net and deep neural nets suffer from optimization issues. I think that's where we're kind of slightly starting to run into, so we need one more idea that we can borrow from the Transformer paper to resolve those difficulties.

There are two optimizations that dramatically help with the depth of these networks and make sure that the networks remain optimizable. Let's talk about the first one.
# Understanding Residual Connections in Neural Networks

The first concept we need to understand from this diagram is the arrow, which represents skip connections, also known as residual connections. These connections were introduced in a 2015 paper titled "Procedural Learning Form and Recognition". 

Residual connections mean that you transform the data, but then you have a skip connection with addition from the previous features. I like to visualize it as a computation that happens from the top to bottom. You have this residual pathway and you are free to fork off from the residual pathway, perform some computation, and then project back to the residual pathway via addition. 

You go from the inputs to the targets only through addition. This is useful because during backpropagation, addition distributes gradients equally to both of its branches that act as the input. The supervision or the gradients from the loss hop through every addition node all the way to the input and also fork off into the residual blocks. 

You have this gradient superhighway that goes directly from the supervision all the way to the input unimpeded. The residual blocks are usually initialized in the beginning so they contribute very little, if anything, to the residual pathway. They come online over time and start to contribute during the optimization. But at least at the initialization, you can go directly from supervision to the input. The gradient is unimpeded and just flows, and then the blocks over time kick in. This dramatically helps with the optimization. 

Let's implement this in our block. We want to do `x = x + self.attention` and `x = x + self.feed_forward`. This is `x` and then we fork off and do some computation and come back. We fork off again, do some computation, and come back. These are residual connections. 

We also have to introduce a projection, `nn.Linear`. After we concatenate, this is the output of the `self.attention` itself. But then we actually want to apply the projection. The projection is just a linear transformation of the outcome of this layer. That's the projection back into the residual pathway. 

In the feed forward, it's going to be the same thing. I could have a `self.projection` here as well, but let me simplify it and couple it inside the same sequential container. This is the projection layer going back into the residual pathway. 

One more small change to note is that the dimensionality of the input and output's inner layer in the feed forward network, as per the paper, has a dimensionality of 2048. This means there's a multiplier of four. So, the inner layer of the feed forward network should be multiplied by four in terms of channel sizes. I have done this by multiplying four times `embed_dim` for the inner layer. 

And that's it! Now we can train this model.
# Optimizing Deep Neural Networks with Feed Forward and Layer Norm

In this article, we will discuss two innovations that are very helpful for optimizing very deep neural networks: feed forward and layer norm. 

Firstly, let's talk about the feed forward. We start from four times n embed and come back down to an embed when we go back to the project. This process adds a bit of computation and grows the layer that is in the residual block on the side of the residual pathway. 

After training this, we actually get down all the way to a 2.08 validation loss. We also see that the network is starting to get big enough that our train loss is getting ahead of validation loss. This indicates that we're starting to see a little bit of overfitting. 

Our generations here are still not amazing, but at least we can see that we are making progress. For instance, we can see phrases like "is here this now grieve sank" which starts to almost look like English. So, we're starting to really get there.

The second innovation that is very helpful for optimizing very deep neural networks is layer norm. Layer norm is implemented in PyTorch. It's a concept that was introduced in a paper a while back. Layer norm is very similar to batch normalization. 

Remember back to our make more series part three where we implemented batch normalization. Batch normalization basically ensures that across the batch dimension, any individual neuron has a unit Gaussian distribution. This means it has a zero mean and one standard deviation output. 

In the case of layer norm, we normalize the rows instead of the columns. For every individual example, its 100-dimensional vector is normalized in this way. Because our computation now does not span across examples, we can delete all of the buffer stuff because we can always apply this operation and don't need to maintain any running buffers. 

There's no distinction between training and test time with layer norm. We do keep gamma and beta, but we don't need the momentum. 

Before I incorporate the layer norm, I just wanted to note that very few details about the Transformer have changed in the last five years. However, this is actually something that slightly departs from the original paper. You see that the add and norm is applied after the transformation. But now, it is a bit more common to apply the layer norm before the transformation. This is called the pre-norm formulation and that's the one that we're going to implement as well. 

In conclusion, the feed forward and layer norm are two important innovations in optimizing deep neural networks. They help in reducing the validation loss and improving the performance of the network.
# Implementing Layer Norms in Transformers

In the original paper on transformers, we need two layer norms. The first layer norm, Norm 1, is an N dot layer norm. We specify the embedding dimension for this norm. The second layer norm, Norm 2, is applied immediately on x. So, `self.layer_norm_1` is applied on x, and `self.layer_norm_2` is applied on x before it goes into self-attention and feed-forward.

The size of the layer norm here is an embed size of 32. When the layer norm is normalizing our features, the normalization happens over 32 numbers. The batch and the time act as batch dimensions for both of them. This is kind of like a per-token transformation that just normalizes the features and makes them a unit mean unit Gaussian at initialization.

However, because these layer norms inside have these gamma and beta trainable parameters, the layer norm can eventually create outputs that might not be unit Gaussian. The optimization will determine that.

After incorporating the layer norms and training them, we see that we get down to 2.06, which is better than the previous 2.08. This is a slight improvement by adding the layer norms. I'd expect that they help even more if we had a bigger and deeper network.

One more thing I forgot to add is that there should be a layer norm at the end of the Transformer and right before the final linear layer that decodes into the vocabulary. So, I added that as well.

At this stage, we actually have a pretty complete Transformer according to the original paper. It's a decoder-only Transformer. The major pieces are in place, so we can try to scale this up and see how well we can push this number.

In order to scale out the model, I had to perform some cosmetic changes to make it nicer. I introduced a variable called `n_layer` which specifies how many layers of the blocks we're going to have. I created a bunch of blocks and we have a new variable `n_heads` as well.

I pulled out the layer norm and added a dropout. Dropout is something that you can add right before the residual connection back or right before the connection back into the original pathway. We can drop out that as the last layer. We can also drop out at the end of the multi-headed attention. And we can also drop out when we calculate the affinities and after the softmax.

Dropout comes from a paper from 2014. It takes your neural net and it randomly, every forward-backward pass, shuts off some subset of neurons. It randomly drops them to zero and trains without them. This effectively trains an ensemble of subnetworks. At test time, everything is fully enabled and all of those subnetworks are merged into a single ensemble.

I added dropout because I'm about to scale up the model quite a bit and I was concerned about overfitting. Now, I changed a number of parameters to scale up the model.
# Hyperparameters in Neural Networks

In this article, we will discuss the hyperparameters of our neural network. I have made the batch size much larger, now at 64. I also changed the block size to be 256. Previously, it was just eight characters of context, but now it is 256 characters of context to predict.

I brought down the learning rate a little bit because the neural net is now much bigger. The embedding dimension is now 384 and there are six heads. This means that every head is 64 dimensional, as is standard. There are going to be six layers of that.

The dropout will be of 0.2. So, every forward-backward pass, 20 percent of all of these intermediate calculations are disabled and dropped to zero.

I have already trained this and ran it. So, how well does it perform? We get a validation loss of 1.48, which is actually quite a bit of an improvement on what we had before, which was 2.07. We went from 2.07 all the way down to 1.48 just by scaling up this neural network with the code that we have.

This of course ran for a lot longer. This may be trained for about 15 minutes on my A100 GPU. That's a pretty good GPU and if you don't have a GPU, you're not going to be able to reproduce this on a CPU. I would not run this on the CPU or a MacBook. You'll have to break down the number of layers and the embedding dimension and so on. But in about 15 minutes, we can get this kind of result.

I'm printing some of the Shakespeare here, but what I did also is I printed 10,000 characters, so a lot more, and I wrote them to a file. It's a lot more recognizable as the input text file. The input text file, just for reference, looked like this. There's always someone speaking in this manner.

Our predictions now take on that form, except of course, they're nonsensical when you actually read them. It is every crimpy bee house oh those preparation we give heed. Anyway, you can read through this. It's nonsensical of course, but this is just a Transformer trained on the character level for 1 million characters that come from Shakespeare. They sort of blabber on in a Shakespeare-like manner, but it doesn't make sense at this scale.

I think it's still a pretty good demonstration of what's possible. That kind of concludes the programming section of this video. We did a pretty good job of implementing this Transformer, but the picture doesn't exactly match up to what we've done. So, what's going on with all these additional parts?

Basically, what's happening here is what we implemented here is a decoder-only Transformer. There's no component here, this part is called the encoder, and there's no cross-attention block here. Our block only has a self-attention and the feed-forward, so it is missing this third in-between piece. This piece does cross-attention, so we don't have it, and we don't have the encoder, we just have the decoder.

The reason we have a decoder only is because we are just generating text and it's unconditioned on anything. We're just blabbering on according to a given dataset. What makes it a decoder is that we are using the triangular mask in our Transformer, so it has this auto-regressive property where we can just go on.
# Understanding the Encoder-Decoder Architecture in Machine Translation

The fact that we're using a triangular mask to mask out the attention makes it a decoder, and it can be used for language modeling. The original paper had an encoder-decoder architecture because it is a machine translation paper. It is concerned with a different setting. 

In particular, it expects some tokens that encode, say for example, French, and then it is expected to decode the translation in English. Typically, these are special tokens. You are expected to read in this and condition on it, and then you start off the generation with a special token called 'start'. This is a special new token that you introduce and always place in the beginning. 

The network is then expected to put 'neural networks are awesome' and then a special 'end' token to finish a generation. This part here will be decoded exactly as we've done it. 'Neural networks are awesome' will be identical to what we did. But unlike what we did, they want to condition the generation on some additional information. In that case, this additional information is the French sentence that they should be translating. 

So, what they do now is they bring in the encoder. The encoder reads this part here, so we're only going to take the part of French and we're going to create tokens from it exactly as we've seen in our video. We're going to put a Transformer on it, but there's going to be no triangular mask and so all the tokens are allowed to talk to each other as much as they want. They're just encoding whatever's the content of this French sentence. 

Once they've encoded it, they've basically come out at the top here. Then what happens here is in our decoder, which does the language modeling, there's an additional connection here to the outputs of the encoder. That is brought in through a cross-attention. The queries are still generated from X, but now the keys and the values are coming from the side. The keys and the values are coming from the top, generated by the nodes that came outside of the encoder. 

Those tops, the keys and the values there, the top of it, are feeding on the side into every single block of the decoder. That's why there's an additional cross-attention. Really, what it's doing is it's conditioning the decoding not just on the past of this current decoding but also on having seen the fully encoded French prompt. 

So, it's an encoder-decoder model, which is why we have those two Transformers, an additional block, and so on. We did not do this because we have nothing to encode. There's no conditioning. We just have a text file and we just want to imitate it. That's why we are using a decoder-only Transformer, exactly as done in GPT. 

Now, I wanted to do a very brief walkthrough of Nano GPT, which you can find on my GitHub. Nano GPT is basically two files of interest: `train.py` and `model.py`. `train.py` is all the boilerplate code for training the network. It is basically all the stuff that we had here. It's the training loop. 

It's just that it's a lot more complicated because we're saving and loading checkpoints and pre-trained weights. We are decaying the learning rate, compiling the model, and using distributed training across multiple nodes or GPUs. So, the `train.py` gets a little bit more complicated. There are more options, etc.
# Understanding the Model.pi in Transformer Models

The `model.pi` in transformer models should look very similar to what we've done here. In fact, the model is almost identical. 

First, we have the causal self-attention block, which should look very recognizable. We're producing queries, keys, and values. We're doing dot products, masking, applying softmax, optionally dropping out, and pooling the values. 

What is different here is that in our code, I have separated out the multi-headed attention into just a single individual head. Then, I have multiple heads and I explicitly concatenate them. Whereas here, all of it is implemented in a batched manner inside a single causal self-attention. So, we don't just have a `b` and a `t` and a `c` dimension, we also end up with a fourth dimension which is the heads. 

It just gets a lot more complex because we have four-dimensional array tensors now, but it is equivalent mathematically. The exact same thing is happening as what we have, it's just a bit more efficient because all the heads are treated as a batch dimension as well. 

Then we have the multiply perceptron. It's using the gelu nonlinearity, which is defined here, except instead of ReLU. This is done just because OpenAI used it and I want to be able to load their checkpoints. 

The blocks of the transformer are identical - the communicate and the compute phase as we saw. The GPT will be identical - we have the position encodings, token encodings, the blocks, the layer norm at the end, the final linear layer. This should all look very recognizable. 

There's a bit more here because I'm loading checkpoints and stuff like that. I'm separating out the parameters into those that should be weight decayed and those that shouldn't. But the generate function should also be very similar. A few details are different, but you should definitely be able to look at this file and understand a lot of the pieces. 

## Training Chat GPT

So, let's now bring things back to Chat GPT. What would it look like if we wanted to train Chat GPT ourselves and how does it relate to what we learned today? 

Well, to train Chat GPT, there are roughly two stages - first is the pre-training stage and then the fine-tuning stage. In the pre-training stage, we are training on a large chunk of the internet and just trying to get a first decoder-only transformer to babble text. So, it's very similar to what we've done ourselves, except we've done a tiny little baby pre-training step. 

In our case, this is how you print the number of parameters. I printed it and it's about 10 million. So, this transformer that I created here to create little Shakespeare was about 10 million parameters. Our dataset is roughly 1 million characters, so roughly 1 million tokens. But you have to remember that OpenAI uses a different vocabulary. They're not on the character level, they use these subword chunks of words. So, they have a vocabulary of roughly 50,000 elements and so their sequences are a bit more condensed. 

Our dataset, the Shakespeare dataset, would be probably around 300,000 tokens in the OpenAI vocabulary, roughly. So, we trained a 10 million parameter model on roughly 300,000 tokens. 

Now, when you go to the GPT-3 paper and you look at the transformers that they trained, they trained a number of transformers of different sizes. But the biggest transformer here has 175 billion parameters. So, ours is again 10 million. They used this number of layers in the...
# Transformer: This is the End

The Transformer model we've been discussing has a number of heads, a head size, and a batch size. In our case, the batch size was 65. The learning rate is similar to what we've used before. 

When the creators of the Transformer model trained it, they used 300 billion tokens. To put that into perspective, our model used about 300,000 tokens. That's a million-fold increase. And by today's standards, even that number isn't considered large. Nowadays, you'd be looking at one trillion tokens and above. 

So, they are training a significantly larger model on a good chunk of the internet. This is the pre-training stage. But otherwise, these hyperparameters should be fairly recognizable to you. The architecture is nearly identical to what we implemented ourselves. Of course, it's a massive infrastructure challenge to train this. You're talking about typically thousands of GPUs having to communicate with each other to train models of this size. 

That's just the pre-training stage. Now, after you complete the pre-training stage, you don't get something that responds to your questions with answers. It's not helpful in that way. What you get is a document completer. It babbles, but it doesn't babble Shakespeare, it babbles internet. It will create arbitrary news articles and documents and it will try to complete documents. That's what it's trained for, it's trying to complete the sequence. 

So, when you give it a question, it might just give you more questions. It will do whatever it looks like some closed document would do in the training data on the internet. You're getting kind of undefined behavior. It might answer your questions with other questions, it might ignore your question, it might just try to complete some news article. It's totally undefined, as we say. 

The second fine-tuning stage is to actually align it to be an assistant. This is the second stage. 

The OpenAI blog post about GPT talks a little bit about how this stage is achieved. There are roughly three steps to this stage. 

First, they start to collect training data that looks specifically like what an assistant would do. If you have documents that have the format where the question is on top and then an answer is below, they have a large number of these. But probably not on the order of the internet. This is probably on the order of maybe thousands of examples. They then fine-tune the model to basically only focus on documents that look like that. You're starting to slowly align it so it's going to expect a question at the top and it's going to expect to complete the answer. These very large models are very sample efficient during their fine-tuning, so this actually somehow works. 

The second step is to let the model respond and then different raters look at the different responses and rank them for their preference as to which one is better than the other. They use that to train a reward model so they can predict, using a different network, how much any candidate response would be desirable. 

Once they have a reward model, they run PPO, which is a form of policy gradient reinforcement learning optimizer, to fine-tune this sampling policy. The goal is that the answers that GPT now generates are expected to score a high reward according to the reward model. 

In conclusion, there's a whole process to training these models, and it's not as simple as it might seem at first glance.
# Training a Decoder-Only Transformer: A Deep Dive

In this article, we will be discussing the training process of a decoder-only Transformer, specifically focusing on the pre-training stage. This process is multi-faceted and involves several steps, each of which is crucial to the development of the model.

The training process can be likened to a fine-tuning stage, with multiple steps in between. This process takes the model from being a document completer to a question answerer. This is a separate stage in itself and a lot of the data used in this stage is not publicly available. It is internal to OpenAI and is much harder to replicate.

This process is what gives you a child GPT. Nano GPT, on the other hand, focuses on the pre-training stage. 

We trained a decoder-only Transformer following the famous paper, "Attention is All You Need". This is essentially a GPT. We trained it on a tiny Shakespeare dataset and got sensible results. All of the training code is available in this codebase, which also includes all the git log commits along the way as we built it up.

In addition to this code, I will be releasing a Google Colab notebook. I hope this gives you a sense of how you can train models like GPT-3. Architecturally, they are basically identical to what we have, but they are somewhere between ten thousand and one million times bigger, depending on how you count.

We did not talk about any of the fine-tuning stages that would typically go on top of this. If you're interested in something that's not just language modeling but you actually want to perform tasks, or you want them to be aligned in a specific way, or you want to detect sentiment, or anything like that, you have to complete further stages of fine tuning.

This could be simple supervised fine tuning or it can be something more fancy like we see in ChatGPT. We actually train a reward model and then do rounds of PPO to align it with respect to the reward model. There's a lot more that can be done on top of it.

I hope you enjoyed this lecture. Go forth and transform. See you later.