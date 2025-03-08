---
layout: default
title: 0 - Micrograd
parent: Andrej Karpathy
has_children: false
nav_order: 0
---

# Understanding Neural Network Training: A Deep Dive

Hello, my name is Andre. I've been training deep neural networks for a bit more than a decade. In this lecture, I'd like to show you what neural network training looks like under the hood. In particular, we are going to start with a blank Jupiter notebook, and by the end of this lecture, we will define and train a neural net. You'll get to see everything that goes on under the hood and exactly how that works on an intuitive level.

Specifically, I would like to take you through the building of Micrograd. Micrograd is a library that I released on GitHub about two years ago. At the time, I only uploaded the source code, and you'd have to go in by yourself and really figure out how it works. So in this lecture, I will take you through it step by step and comment on all the pieces of it.

So, what is Micrograd and why is it interesting? Micrograd is basically an autograd engine. Autograd is short for automatic gradient, and really what it does is it implements backpropagation. Backpropagation is an algorithm that allows you to efficiently evaluate the gradient of some kind of a loss function with respect to the weights of a neural network. What that allows us to do then is we can iteratively tune the weights of that neural network to minimize the loss function and therefore improve the accuracy of the network. Backpropagation would be at the mathematical core of any modern deep neural network library like PyTorch or Jaxx.

The functionality of Micrograd is best illustrated by an example. Micrograd basically allows you to build out mathematical expressions. Here, we are building out an expression where we have two inputs, A and B. A and B are -4 and 2, but we are wrapping those values into a value object that we are going to build out as part of Micrograd. This value object will wrap the numbers themselves.

We are going to build out a mathematical expression here where A and B are transformed into C, D, and eventually E, F, and G. I'm showing some of the functions and the operations that Micrograd supports. You can add two value objects, you can multiply them, you can raise them to a constant power, you can offset by one, negate, squash at zero, square, divide by a constant, divide by it, etc.

We're building out an expression graph with these two inputs, A and B, and we're creating an output value of G. Micrograd will, in the background, build out this entire mathematical expression. For example, it will know that C is also a value. C was a result of an addition operation, and it will maintain pointers to A and B value objects. It will basically know exactly how all of this is laid out.

Not only can we do what we call the forward pass, where we actually look at the value of G (which is pretty straightforward, we will access that using the .data attribute and so the output of the forward pass, the value of G, is 24.7), but the big deal is that we can also take this G value object and we can call that backward. This will basically initialize backpropagation at the node G. Backpropagation is going to do...
# Understanding Back Propagation and Chain Rule in Neural Networks

The process is going to start at `g` and it's going to go backwards through that expression graph. It will recursively apply the chain rule from calculus. This allows us to evaluate the derivative of `g` with respect to all the internal nodes like `e`, `d`, and `c`, but also with respect to the inputs `a` and `b`. 

We can actually query this derivative of `g` with respect to `a` for example, that's `a.grad` in this case. It happens to be 138 and the derivative of `g` with respect to `b`, which also happens to be here 645. This derivative is very important information because it's telling us how `a` and `b` are affecting `g` through this mathematical expression. 

In particular, `a.grad` is 138 so if we slightly nudge `a` and make it slightly larger, 138 is telling us that `g` will grow and the slope of that growth is going to be 138. The slope of growth of `b` is going to be 645. This tells us about how `g` will respond if `a` and `b` get tweaked a tiny amount in a positive direction.

Now, you might be confused about what this expression is that we built out here. This expression is completely meaningless, I just made it up. I'm just flexing about the kinds of operations that are supported by Micrograd. What we actually really care about are neural networks. But it turns out that neural networks are just mathematical expressions just like this one, but actually slightly bit less crazy. 

Neural networks are just a mathematical expression. They take the input data as an input and they take the weights of a neural network as an input. It's a mathematical expression and the output are your predictions of your neural net or the loss function. We'll see this in a bit, but basically, neural networks just happen to be a certain class of mathematical expressions.

Back propagation is actually significantly more general. It doesn't actually care about neural networks at all. It only tells us about arbitrary mathematical expressions and then we happen to use that machinery for training of neural networks.

One more note I would like to make at this stage is that Micrograd is a scalar valued auto grad engine. It's working on the level of individual scalars like negative four and two. We're taking neural nets and we're breaking them down all the way to these atoms of individual scalars and all the little pluses and times. 

This is excessive and so obviously, you would never be doing any of this in production. It's really just put down for pedagogical reasons because it allows us to not have to deal with these n-dimensional tensors that you would use in modern deep neural network library. 

This is done so that you understand and refactor out back propagation and chain rule and understanding of neural network training. If you actually want to train bigger networks, you have to be using these tensors. But none of the math changes. This is done purely for efficiency. 

We are basically taking scale value, all the scale values, we're packaging them up into tensors which are just arrays of these scalars. Then because we have these large arrays, we're making operations on those large arrays. That allows us to take advantage of the parallelism in a computer and all those operations can be done in parallel. Then the whole thing runs faster. 

But really, none of the math changes and that's done purely for efficiency. So, don't think that it's pedagogically different.
# Understanding Micrograd and Neural Networks

It's useful to be dealing with tensors from scratch. That's why I fundamentally wrote Micrograd. You can understand how things work at the fundamental level and then speed it up later. 

Here's the fun part: my claim is that Micrograd is what you need to train your networks and everything else is just efficiency. You'd think that Micrograd would be a very complex piece of code, but that turns out to not be the case. 

If we just go to Micrograd, you'll see that there are only two files here. In Micrograd, this is the actual engine. It doesn't know anything about neural networks. This is the entire neural networks library on top of Micrograd: engine and nn.py. 

The actual backpropagation autograd engine that gives you the power of neural networks is literally 100 lines of code of very simple Python, which we'll understand by the end of this lecture. 

Then, nn.py, this neural network library built on top of the autograd engine, is like a joke. We have to define what is a neuron, then we have to define what is a layer of neurons, and then we define what is a multi-layer perceptron, which is just a sequence of layers of neurons. So, it's just a total joke. 

Basically, there's a lot of power that comes from only 150 lines of code. That's all you need to understand to understand neural network training and everything else is just efficiency. Of course, there's a lot to efficiency, but fundamentally that's all that's happening. 

Now, let's dive right in and implement Micrograd step by step. The first thing I'd like to do is to make sure that you have a very good understanding, intuitively, of what a derivative is and exactly what information it gives you. 

Let's start with some basic imports that I copy-paste in every Jupyter notebook, always. Let's define a function, a scalar-valued function, f of x, as follows. I just make this up randomly. I just want a scalar-valued function that takes a single scalar x and returns a single scalar y. We can call this function, of course. So, we can pass in, say, 3.0 and get 20 back. 

We can also plot this function to get a sense of its shape. You can tell from the mathematical expression that this is probably a parabola. It's a quadratic. If we just create a set of scalar values that we can feed in using, for example, a range from negative five to five in steps of 0.25, we can actually call this function on this numpy array as well. So, we get a set of y's if we call f on axis. These y's are basically also applying a function on every one of these elements independently. 

We can plot this using matplotlib. So, plt.plot x's and y's and we get a nice parabola. Previously, we fed in 3.0 somewhere here and we received 20 back, which is here the y coordinate. 

Now, I'd like to think through what is the derivative of this function at any single input point x. What is the derivative at different points x of this function? If you remember back to your calculus class, you've probably derived derivatives. We take this mathematical expression 3x squared minus 4x plus 5 and you would write out on a piece of paper and you would apply the rules of differentiation.
# Understanding Derivatives in Neural Networks

In the world of neural networks, we often talk about derivatives. However, no one actually writes out the expression for the neural net. It would be a massive expression, with thousands or tens of thousands of terms. No one actually derives the derivative of course. So, we're not going to take this kind of symbolic approach.

Instead, let's look at the definition of a derivative and make sure that we really understand what a derivative is measuring and what it's telling you about the function.

The definition of a derivative is the limit as h goes to zero of f(x + h) - f(x) / h. Basically, what it's saying is if you're at some point x that you're interested in, and if you slightly bump up or increase it by a small number h, how does the function respond? What is the sensitivity of the function? What is the slope at that point? Does the function go up or does it go down and by how much? That's the slope of that function.

We can evaluate the derivative numerically by taking a very small h. Of course, the definition would ask us to take h to zero, but we're just going to pick a very small h, say 0.001. Let's say we're interested in point 3.0. We can look at f(x), which is 20, and now f(x + h). If we slightly nudge x in a positive direction, how is the function going to respond?

Just looking at this, do you expect f(x + h) to be slightly greater than 20 or do you expect it to be slightly lower than 20? Since this 3 is here and this is 20, if we slightly go positively, the function will respond positively. So, you'd expect this to be slightly greater than 20. Now, by how much is it telling you the strength of that slope? The size of the slope is f(x + h) - f(x), which is how much the function responded in the positive direction. We have to normalize by the run, so we have the rise over run to get the slope.

This is just a numerical approximation of the slope because we have to make h very small to converge to the exact amount. At some point, we're going to get an incorrect answer because we're using floating point arithmetic and the representations of all these numbers in computer memory is finite. So, we can converge towards the right answer with this approach.

At 3, the slope is 14. You can see that by taking 3x squared - 4x + 5 and differentiating it in our head. So, 3x squared would be 6x - 4, and then we plug in x equals 3, so that's 18 - 4, which is 14. So, this is correct.

Now, how about the slope at say negative 3? Would you expect the slope to be the same? Telling the exact value is really a challenge, but understanding the concept is the key.
# Understanding the Concept of Derivatives

Let's start with a simple function, `f(x) = x^2`. We can plot this function and observe its behavior. For instance, at `x = -3`, the function value is `9`. However, what is the sign of the slope at this point?

If we slightly move in the positive direction at `x`, the function would actually go down. This indicates that the slope would be negative. Therefore, we'll get a number slightly below `20`. If we calculate the slope, we expect something negative, say `-22`.

At some point, the slope would be zero. For this specific function, I looked it up previously and it's at `0.67` (approximately `2/3`). At this precise point, if we nudge in a positive direction, the function doesn't respond; it stays the same. That's why the slope is zero.

Now, let's look at a more complex case. We have a function with output variable `d` that is a function of three scalar inputs `a`, `b`, and `c`. `a`, `b`, and `c` are some specific values, three inputs into our expression graph, and a single output `d`. If we print `d`, we get `4`.

What we need to do now is to look at the derivatives of `d` with respect to `a`, `b`, and `c`, and understand the intuition of what this derivative is telling us. To evaluate this derivative, we're going to use a very small value of `h`. We're going to fix the inputs at some values that we're interested in. These are the points `a`, `b`, `c` at which we're going to be evaluating the derivative of `d` with respect to all `a`, `b`, and `c`.

We have `d1` as that expression. For example, to look at the derivative of `d` with respect to `a`, we'll take `a` and bump it by `h`. We'll get `d2` to be the exact same function. We're going to print `d1`, `d2`, and the slope. The derivative or slope here will be `d2 - d1 / h`. `d2 - d1` is how much the function increased when we bumped the specific input that we're interested in by a tiny amount. This is then normalized by `h`.

So, if we run this, we're going to print `d1`, which we know is `4`. Now, `d2` will be bumped; `a` will be bumped by `h`. Let's think through what `d2` will be printed out here. In particular, `d1` will be `4`. Will `d2` be a number slightly greater than `4` or slightly lower than `4`? That's going to tell us the sign of the derivative. We're bumping `a` by `h`.
# Understanding Derivatives in Neural Networks

Let's consider the equation `b as -3` and `c as 10`. You can intuitively think through this derivative and what it's doing. `a` will be slightly more positive, but `b` is a negative number. So, if `a` is slightly more positive because `b` is -3, we're actually going to be adding less to `d`. You'd actually expect that the value of the function will go down. 

Let's see this in action. We went from 4 to 3.9996, which tells you that the slope will be negative. The exact amount of slope is -3. You can also convince yourself that -3 is the right answer mathematically and analytically. If you have `a * b + c` and you are using calculus, then differentiating `a * b + c` with respect to `a` gives you just `b`. Indeed, the value of `b` is -3, which is the derivative that we have. So, you can tell that that's correct.

Now, if we do this with `b`, so if we bump `b` by a little bit in a positive direction, we'd get different slopes. So, what is the influence of `b` on the output `d`? If we bump `b` by a tiny amount in a positive direction then because `a` is positive, we'll be adding more to `d`. 

What is the sensitivity or the slope of that addition? It might not surprise you that this should be 2. Why is it 2? Because `d/d` by `db` differentiating with respect to `b` would give us `a`. The value of `a` is two, so that's also working well.

Then, if `c` gets bumped a tiny amount by `h`, then of course `a * b` is unaffected. Now `c` becomes slightly bit higher. What does that do to the function? It makes it slightly bit higher because we're simply adding `c`. It makes it slightly bit higher by the exact same amount that we added to `c`. That tells you that the slope is one. That will be the rate at which `d` will increase as we scale `c`.

We now have some intuitive sense of what this derivative is telling you about the function. We'd like to move to neural networks now. As I mentioned, neural networks will be pretty massive expressions, mathematical expressions. So, we need some data structures that maintain these expressions. That's what we're going to start to build out now.

We're going to build out this value object that I showed you in the readme page of micrograd. Let's create a skeleton of the first very simple value object. The class `Value` takes a single scalar value that it wraps and keeps track of. For example, we can create a `Value` object with data equals two. Python will internally use the wrapper function to return this string.
# Implementing Addition and Multiplication in Python

In Python, we can create objects that have more than two values. However, if we want to add these objects, we would currently get an error. This is because Python doesn't know how to add two value objects. So, we have to tell it how to do so.

## Addition

To define these operators for these objects, we have to use special double underscore methods in Python. For instance, if we use the plus operator, Python will internally call `a.__add__(b)`. Here, `b` will be the other and `self` will be `a`.

We can see that what we're going to return is a new value object. It's going to be wrapping the plus of their data. Remember, because `data` is the actual Python number, this operator here is just the typical floating point plus addition. It's not an addition of value objects. We will return a new value so now `a + b` should work and it should print `value of -1`, because that's `2 + -3`.

## Multiplication

Let's now implement multiplication so we can recreate this expression. Multiplication will be fairly similar to addition. Instead of `add`, we're going to be using `mul`. Here, of course, we want to do times.

Now, we can create a `c` value object which will be `10.0` and now we should be able to do `a * b`. That's `value of -6`. 

By the way, suppose that we didn't have the `__repr__` function here. Then, you'll get some kind of an ugly expression. What `__repr__` is doing is it's providing us a way to print out a nicer looking expression in Python. So we don't just have something cryptic, we actually have `value of -6`.

This gives us `a * b` and then we should now be able to add `c` to it because we've defined and told Python how to do `mul` and `add`. This will call `a.__mul__(b)` and then this new value object will be `.__add__(c)`. Let's see if that worked. Yep, so that worked well. That gave us `4` which is what we expect from before. We can just call them manually as well.

## Keeping Pointers

What we are missing now is the connective tissue of this expression. As I mentioned, we want to keep these expression graphs so we need to know and keep pointers about what values produce what other values.

Here, for example, we are going to introduce a new variable which we'll call `children` and by default, it will be an empty tuple. We're actually going to keep a slightly different variable in the class which we'll call `_prev` which will be the set of children. This is how I did it in the original Micrograd. Looking at my code here, I can't remember exactly the reason. I believe it was efficiency but this `_children` will be a tuple for convenience but then when we actually maintain it in the class it will be just a set.
# Building a Data Structure for Mathematical Expressions

In this tutorial, we will be discussing how to build a data structure for mathematical expressions. This is particularly useful for efficiency in programming.

When we create a value with a constructor, the children will be empty and `prev` will be the empty set. However, when we're creating a value through addition or multiplication, we're going to feed in the children of this value, which in this case is `self` and `other`. These are the children here. 

For instance, if we execute `d.prev`, we'll see that the children of `d` are the values of -6 and 10. This is the value resulting from `a` times `b` and the `c` value which is 10.

The last piece of information we need is the operation that created this value. We need one more element here, let's call it `_pop`. By default, this is the empty set for leaves. We'll maintain it here and the operation will be a simple string. In the case of addition, it's `+` and in the case of multiplication, it's `*`. 

So now, not only do we have `d.prev`, we also have `d.pop`. We know that `d` was produced by an addition of those two values. We have the full mathematical expression and we're building out this data structure. We know exactly how each value came to be, by which expression and from what other values.

As these expressions are about to get quite a bit larger, we'd like a way to nicely visualize these expressions that we're building out. For that, I'm going to use a bunch of code that's going to visualize these expression graphs for us.

This code creates a new function `drawdot` that we can call on some root node. It's going to visualize it. If we call `drawdot` on `d`, which is the final value here that is `a` times `b` plus `c`, it creates a visual representation of the expression.

This code uses Graphviz, an open-source graph visualization software. We're building out this graph using Graphviz's API. The helper function `trace` enumerates all of the nodes and edges in the graph. It builds a set of all the nodes and edges and then we iterate for all the nodes. We create special node objects for them using `dot.node` and we also create edges using `dot.edge`.

The only slightly tricky part here is that I add these fake nodes which are these operation nodes. For example, this node here is a `+` node. I create these special `op` nodes and connect them accordingly. These nodes are not actual nodes in the original graph. They're not actually a value object. The only value objects here are the things in squares, those are actual value objects or representations thereof. These `op` nodes are just created in this code. 

In conclusion, this method allows us to build a data structure for mathematical expressions and visualize them for better understanding and efficiency.
# Building Mathematical Expressions with Back Propagation

Let's start by refining our `drawdot` routine to make it visually appealing. We'll also add labels to these graphs for better understanding of the variables. 

```python
label = ''
```

By default, we'll keep the label empty and save it. Now, we'll label 'a' and 'e'. 

```python
a.label = 'a'
e.label = 'e'
```

We'll also create a new variable 'e' and add it to 'c'. 

```python
e = e + c
```

Next, we'll label 'd'. 

```python
d.label = 'd'
```

Nothing really changes here, we've just added a new function 'e'. Now, when we print this, we'll also print the label. 

```python
print("%s bar" % label)
```

Now, we have the label on the left. It says 'a' and 'b' are creating 'e', and then 'e' plus 'c' creates 'd'. 

Let's make this expression one layer deeper. 'd' will not be the final output node. Instead, after 'd', we'll create a new value object called 'f'. 

```python
f = -2.0
f.label = 'f'
```

Next, we'll create 'l' which will be the output of our graph. 

```python
l = p * f
```

The output of 'l' will be -8. 

```python
l.label = 'l'
```

Let's quickly recap what we've done so far. We've built mathematical expressions using only addition and multiplication. They are scalar-valued. We can do this forward pass and build out a mathematical expression. We have multiple inputs here: 'a', 'b', 'c', and 'f', going into a mathematical expression that produces a single output 'l'. This visualizes the forward pass, so the output of the forward pass is -8. 

Next, we'd like to run back propagation. In back propagation, we start at the end and reverse, calculating the gradient along all these intermediate values. We're computing the derivative of that node with respect to 'l'. The derivative of 'l' with respect to 'l' is just 1. Then we're going to derive what is the derivative of 'l' with respect to 'f', 'd', 'c', 'e', 'b', and 'a'. 

In the neural network setting, you'd be very interested in the derivative of this loss function 'l' with respect to the weights of a neural network. Here, of course, we have just these variables 'a', 'b', 'c', and 'f', but some of these will eventually represent the weights of a neural net. We'll need to know how those weights are impacting the loss function. We'll be interested in the derivative of the output with respect to some of its leaf nodes.
# Neural Networks: Nodes, Leaf Nodes, and Back Propagation

In a neural network, nodes and leaf nodes play a significant role. The weights of the neural net are represented by these nodes and leaf nodes. The other leaf nodes, of course, will be the data itself. However, we usually do not want or use the derivative of the loss function with respect to data because the data is fixed. The weights, on the other hand, will be iterated on using the gradient information.

Next, we are going to create a variable inside the `Value` class that maintains the derivative of `l` with respect to that value. We will call this variable `grad`. So, there's a `data` and there's a `self.grad`. Initially, it will be zero. Remember that zero basically means no effect. At initialization, we're assuming that every value does not impact or affect the output. If the gradient is zero, that means that changing this variable is not changing the loss function. So, by default, we assume that the gradient is zero.

We are then able to visualize it after `data`. Here, `grad` is `0.4f` and this will be in the graph. Now, we are going to be showing both the `data` and the `grad`, initialized at zero. We are just about getting ready to calculate the back propagation.

This `grad`, as I mentioned, is representing the derivative of the output, in this case `l`, with respect to this value. So, with respect to `f`, `d`, and so on. Let's now fill in those gradients and actually do back propagation manually.

Let's start filling in these gradients, starting all the way at the end. First, we are interested in filling in this gradient here. What is the derivative of `l` with respect to `l`? In other words, if I change `l` by a tiny amount of `h`, how much does `l` change? It changes by `h`, so it's proportional and therefore the derivative will be one.

We can, of course, measure these or estimate these numerical gradients numerically, just like we've seen before. If I take this expression and create a `def` function here, I can put this here. Now, the reason I'm creating a gating function here is because I don't want to pollute or mess up the global scope. This is just kind of like a little staging area and, as you know, in Python all of these will be local variables to this function, so I'm not changing any of the global scope here.

Here, `l1` will be `l`, and then I'm copy-pasting this expression in for `a`. This would be measuring the derivative of `l` with respect to `a`. So here, this will be `l2`. Then, we want to print this derivative. So, print `l2 - l1`, which is how much `l` changed, and then normalize it by `h`. This is the rise over run. We have to be careful because `l` is a value node, so we actually want its data, so that these are floats dividing by `h`. This should print the derivative of `l` with respect to `a` because `a` is the one that we bumped a little bit by `h`.

So, what is the derivative of `l` with respect to `a`? It's six. Obviously, if we change `l` by `h`, then that would be here effectively. This looks really awkward, but changing `l` by `h`, you see the derivative here is 1. That's kind of like the base case of what we are doing here.
# Understanding Backpropagation

So, essentially, we cannot come up here and manually set `l.grad` to one. This is our manual backpropagation. `l.grad` is one, and let's redraw and we'll see that we filled in `grad` as 1 for `l`. 

We're now going to continue the backpropagation. Let's look at the derivatives of `l` with respect to `d` and `f`. Let's do `d` first. What we are interested in is, we'd like to know what is `dl/dd`. 

If you know your calculus, `l` is `d` times `f`, so `dl/dd` would be `f`. If you don't believe me, we can also just derive it because the proof would be fairly straightforward. We go to the definition of the derivative which is `f(x + h) - f(x) / h` as a limit of `h` goes to zero of this kind of expression. 

When we have `l` is `d` times `f`, then increasing `d` by `h` would give us the output of `d + h` times `f`. That's basically `f(x + h)` right, minus `d` times `f`, and then divide `h`. Symbolically expanding out here, we would have `d` times `f` plus `h` times `f` minus `d` times `f` divide `h`. 

You see how the `df - df` cancels, so you're left with `h` times `f` divide `h`, which is `f`. So, in the limit as `h` goes to zero of the derivative definition, we just get `f` in the case of `d` times `f`. 

Symmetrically, `dl/df` will just be `d`. So, what we have is that `f.grad` is just the value of `d`, and `d.grad` is the value of `f`. 

Let's just make sure that these were correct. We seem to think that `dl/dd` is negative two, so let's double-check. We want the derivative with respect to `f`. Let's do a `+h` here, and this should print the derivative of `l` with respect to `f`. We expect to see four, and this is four up to floating point funkiness. 

`dl/dd` should be `f` which is negative two. `grad` is negative two. `d.data += h` right here, so we've added a little `h` and then we see how `l` changed. We expect to print negative two. 

We've numerically verified what we're doing here is what kind of like an inline gradient check. A gradient check is when we are deriving this backpropagation and getting the derivative with respect to all the intermediate results, and then numerical gradient is just estimating it using a small step size. 

Now we're getting to the crux of backpropagation. This will be the most important node to understand because if you understand the gradient for this node, you understand all of backpropagation and all of training of neural nets basically. 

We need to derive `dl/dc`, in other words, the derivative of `l` with respect to `c`. We've computed all these other gradients already, now we're coming here and we're going to compute the derivative with respect to `c`.
# Continuing the Back Propagation Manually

We want to derive `dl/dc` and `dl/de`. Now, here's the problem: how do we derive `dl/dc`? We actually know the derivative `l` with respect to `d`, so we know how `l` is assessed to `d`. But how is `l` sensitive to `c`? So, if we wiggle `c`, how does that impact `l` through `d`? We also know how `c` impacts `d`. 

So, intuitively, if you know the impact that `c` is having on `d` and the impact that `d` is having on `l`, then you should be able to somehow put that information together to figure out how `c` impacts `l`. Indeed, this is what we can actually do. 

In particular, we know, just concentrating on `d` first, let's look at what is the derivative of `d` with respect to `c`. So, here we know that `d` is `c` times `c` plus `e`. That's what we know and now we're interested in `dd/dc`. 

If you just know your calculus again and you remember that differentiating `c` plus `e` with respect to `c`, you know that that gives you `1.0`. We can also go back to the basics and derive this because again we can go to our `f(x + h) - f(x) / h`, that's the definition of a derivative as `h` goes to zero. 

Focusing on `c` and its effect on `d`, we can basically do the `f(x + h)` will be `c` is incremented by `h` plus `e`, that's the first evaluation of our function minus `c` plus `e`, and then divide `h`. 

Just expanding this out, this will be `c + h + e - c - e / h` and then you see here how `c - c` cancels `e - e` cancels we're left with `h / h` which is `1.0`. 

By symmetry also `dd/de` will be `1.0` as well. So basically, the derivative of a sum expression is very simple and this is the local derivative. I call this the local derivative because we have the final output value all the way at the end of this graph and we're now like a small node here. 

This is a little plus node and it doesn't know anything about the rest of the graph that it's embedded in. All it knows is that it did a plus, it took a `c` and an `e`, added them and created `d`. This plus node also knows the local influence of `c` on `d` or rather the derivative of `d` with respect to `c` and it also knows the derivative of `d` with respect to `e`. 

But that's not what we want, that's just a local derivative. What we actually want is `dl/dc` and `l` could be just one step away but in a general case, this little plus node could be embedded in a massive graph. 

Again, we know how `l` impacts `d` and now we know how `c` and `e` impact `d`. How do we put that information together to write `dl/dc`? The answer, of course, is the chain rule in calculus. 

I pulled up a chain rule here from Wikipedia and I'm going to go through this very briefly. Chain rule Wikipedia sometimes can be very confusing and calculus can be very confusing. This is the way I learned the chain rule and it was very confusing.
# Understanding the Chain Rule in Calculus

Let's dive into a complex topic - the chain rule in calculus. It's not as complicated as it seems, and I find this expression much more digestible: if a variable `z` depends on a variable `y`, which itself depends on the variable `x`, then `z` depends on `x` as well, obviously through the intermediate variable `y`. 

In this case, the chain rule is expressed as follows: if you want `dz/dx`, then you take `dz/dy` and multiply it by `dy/dx`. So, the chain rule fundamentally tells us how we chain these derivatives together correctly. To differentiate through a function composition, we have to apply a multiplication of those derivatives. That's really what the chain rule is telling us.

Here's a nice little intuitive explanation which I also think is kind of cute. The chain rule says that knowing the instantaneous rate of change of `z` with respect to `y` and `y` relative to `x` allows one to calculate the instantaneous rate of change of `z` relative to `x` as a product of those two rates of change. Simply put, it's the product of those two.

Here's a good analogy: if a car travels twice as fast as a bicycle, and the bicycle is four times as fast as a walking man, then the car travels two times four, or eight times as fast as the man. This makes it very clear that the correct thing to do is to multiply. So, the car is twice as fast as the bicycle, and the bicycle is four times as fast as the man. Therefore, the car will be eight times as fast as the man. We can take these intermediate rates of change, if you will, and multiply them together. That justifies the chain rule intuitively.

So, what does this mean for us? There's a very simple recipe for deriving what we want, which is `dl/dc`. What we have so far is we know `dl/dd` (the derivative of `l` with respect to `d`). We know that `dl/dd` is `-2`. Now, because of this local reasoning, we know `dd/dc` (how `c` impacts `d`). In particular, this is a plus node, so the local derivative is simply `1.0`. It's very simple.

The chain rule tells us that `dl/dc`, going through this intermediate variable, will just be simply `dl/dd` times `dd/dc`. That's the chain rule. So, this is identical to what's happening here, except `z` is our `l`, `y` is our `d`, and `x` is our `c`. We literally just have to multiply these local derivatives. `dd/dc` is just `1`, so we basically just copy over `dl/dd` because this is just times `1`.

So, what does it do? Because `dl/dd` is `-2`, what is `dl/dc`? Well, it's the local gradient `1.0` times `dl/dd`, which is `-2`. So, literally, what a plus node does, you can look at it that way, is it literally just routes the gradient. Because the plus node's local derivatives are just `1`, we basically just copy over `dl/dd`.
# Understanding the Chain Rule and Backpropagation

In this article, we will delve into the chain rule and backpropagation. We will start with the chain rule, which states that the derivative of a function of a function is the derivative of the outer function times the derivative of the inner function. 

Let's consider a simple example. We have a function `L` which is a function of `D`. The derivative of `L` with respect to `D` is simply `DL/DD`. This derivative gets routed to both `C` and `E`. 

We can imagine this as a backpropagating signal carrying the information of the derivative of `L` with respect to all the intermediate nodes. It flows backwards through the graph. A plus node will simply distribute the derivative to all the children nodes of it. 

Let's verify this. We will increment `C` so `C.data` will be credited by `H`. When we run this, we expect to see `-2`. The same applies for `E`. We increment `E.data` by `H` and we expect to see `-2`. 

Now, let's recurse our way backwards again and apply the chain rule. We have `DL/DE` as `-2`. We want `DL/DA` and the chain rule tells us that it's `-2` times the local gradient. 

The local gradient is `DE/DA`. We have `E` as `A` times `B`. We're asking what is `DE/DA` and of course, it's the value of `B`. So, `A.grad` is `DL/DE` which is `-2` times `DE/DA` which is the value of `B`. 

Similarly, `B.grad` is `DL/DE` which is `-2` times `DE/DB` which is the value of `A`. These are our claimed derivatives. 

In conclusion, the chain rule is a fundamental concept in calculus that is extensively used in machine learning algorithms, particularly in the training of neural networks through backpropagation. Understanding how it works is crucial for understanding the inner workings of these algorithms.
# Backpropagation: A Step-by-Step Guide

In this article, we will be discussing the concept of backpropagation, a fundamental algorithm in machine learning. We will be going through the process manually, step by step, to understand how it works.

Let's start by redrawing our computation graph. We see here that `a dot grad` turns out to be 6 because that is the result of `-2` times `-3`. Similarly, `b dot grad` is `-4`, which is the result of `-2` times `2`.

Now that we have made our claims, let's verify them. We have claimed that `a dot grad` is `6`. Let's verify this. Indeed, it is `6`. We also have `beta data` plus `h`. So, when we nudge `b` by `h` and observe what happens, we claim it's `-4`. And indeed, it's `-4` plus or minus some float oddness.

That's it! This was the manual backpropagation process, all the way from the start to all the leaf nodes. We've done it piece by piece. All we've done is iterated through all the nodes one by one and locally applied the chain rule. We always know what the derivative of `l` with respect to this little output is. Then we look at how this output was produced. This output was produced through some operation, and we have the pointers to the children nodes of this operation. In this little operation, we know what the local derivatives are, and we just multiply them onto the derivative. So, we just go through and recursively multiply on the local derivatives. That's what backpropagation is - a recursive application of the chain rule backwards through the computation graph.

Let's see this power in action. What we're going to do is nudge our inputs to try to make `l` go up. In particular, we're going to change `a.data`. If we want `l` to go up, that means we just have to go in the direction of the gradient. So, `a` should increase in the direction of the gradient by some small step amount. This is the step size. We don't just want this for `a`, but also for `b`, `c`, and `f`. These are leaf nodes which we usually have control over. If we nudge in the direction of the gradient, we expect a positive influence on `l`. So, we expect `l` to become less negative, maybe up to `-6` or something like that.

We have to rewrite the forward pass to see this. Let's do that here. This is effectively the forward pass, and `f` would be unchanged. Now, if we print `l.data`, we expect it to be less negative because we nudged all the values, all the inputs, in the direction of the gradient. Maybe it's `-6` or so. Let's see what happens. Okay, it's `-7`.

This is basically one step of an optimization that we'll end up running. The gradient gives us some power because we know how to influence the final outcome. This will be extremely useful for training neural networks, as you'll see.

In the next part of this series, we will do one more example of manual backpropagation using a bit more complex and useful example. We are going to back propagate through a more complex computation graph. Stay tuned!
# Understanding Neurons and Neural Networks

In our journey to understand and build neural networks, we start with the most basic unit - the neuron. In the simplest case, we have multilayer perceptrons, also known as a two-layer neural net. This network consists of hidden layers made up of neurons, all of which are fully connected to each other.

Biologically, neurons are complex devices. However, for our purposes, we use a simple mathematical model to represent them.

## A Mathematical Model of a Neuron

Consider a neuron with some inputs, represented as `x`. These inputs interact with synapses, which have weights on them, represented as `w`. The synapse interacts with the input to the neuron multiplicatively, resulting in `w` times `x` flowing to the cell body of the neuron. 

In a scenario with multiple inputs, we have multiple `w` times `x` flowing into the cell body. The cell body also has a bias, which can be thought of as the innate 'trigger happiness' of the neuron. This bias can make the neuron more or less trigger happy, regardless of the input.

Essentially, we're taking all the `w` times `x` of all the inputs, adding the bias, and then passing it through an activation function. This activation function is usually a squashing function, like a sigmoid or `tanh`.

## The Activation Function

As an example, let's consider the `tanh` function. In Python, we can use the `numpy` library to call this function on a range and plot it. The `tanh` function squashes the inputs on the y-coordinate. At zero, we get exactly zero. As you go more positive in the input, the function will only go up to one and then plateau out. If you pass in very positive inputs, it will cap it smoothly at one. On the negative side, it will cap it smoothly to negative one. This is the squashing function or an activation function.

The output of this neuron is just the activation function applied to the dot product of the weights and the inputs.

## An Example

Let's consider a two-dimensional neuron with two inputs, `x1` and `x2`. The weights of this neuron are `w1` and `w2`, representing the synaptic strengths for each input. The neuron also has a bias, `b`.

According to our model, we need to multiply `x1` times `w1` and `x2` times `w2`, and then add the bias on top of it. This can be represented as `x1*w1 + x2*w2 + b`. 

In conclusion, understanding the basic mathematical model of a neuron and its activation function is crucial in building and understanding neural networks.
# Implementing the Activation Function in Neural Networks

In this article, we will discuss the implementation of the activation function in neural networks. We will start with the raw cell body, without the activation function for now. This should be enough to give us the product of x1 times w1 and x2 times w2, which are then added together. The bias is then added on top of this sum, which we will denote as 'n'.

Next, we will take 'n' through an activation function. Let's say we use the hyperbolic tangent function (tanh) to produce the output. We will denote the output as 'o', which is equal to 'n' dot tanh. However, we haven't yet written the tanh function.

The reason we need to implement the tanh function is that it is a hyperbolic function. So far, we have only implemented addition and multiplication, and we can't make a tanh out of just these two operations. We also need exponentiation. Tanh is represented by a specific formula, which involves exponentiation, something we have not yet implemented for our low-value node. Therefore, we won't be able to produce tanh yet and we have to go back up and implement something like it.

One option here is to implement exponentiation. We could return the exponent of a value instead of the tanh of a value. If we had the exponent, we would have everything else that we need because we know how to add and multiply. Therefore, we would be able to create tanh if we knew how to exponentiate.

However, for the purposes of this example, I specifically wanted to show you that we don't necessarily need to have the most atomic pieces in this value object. We can create functions at arbitrary points of abstraction. They can be complicated functions, but they can also be very simple functions like addition. It's totally up to us. The only thing that matters is that we know how to differentiate through any one function. As long as we know how to create the local derivative of how the inputs impact the output, that's all we need. Therefore, we're going to cluster up all of this expression and we're not going to break it down to its atomic pieces. We're just going to directly implement tanh.

Let's do that. We will define tanh and the output will be a value of a specific expression. Let's grab 'n', which is a cell.theta, and use it in the tanh formula: `math.exp(2*n) - 1 / math.exp(2*n) + 1`. We can call this 'x', just so that it matches exactly. This will be 't' and the children of this node will be just one child, 'self'. We're wrapping it in a tuple, so this is a tuple of one object. The name of this operation will be 'tanh'.
# Implementing Tanh in Neural Networks

Okay, so now we should be implementing `tanh`. We can scroll all the way down here, and we can actually do `n.tanh` and that's going to return the `tanh` output of `n`. Now, we should be able to draw it out of `o`, not of `n`. There we go, `n` went through `tanh` to produce this output. 

So now, `tanh` is a sort of our little micro grad supported node here as an operation. As long as we know the derivative of `tanh`, then we'll be able to back propagate through it. Now let's see this `tanh` in action. Currently, it's not squashing too much because the input to it is pretty low. So if the bias was increased to say eight, then we'll see that what's flowing into the `tanh` now is two, and `tanh` is squashing it to 0.96. So we're already hitting the tail of this `tanh` and it will sort of smoothly go up to 1 and then plateau out over there.

Okay, so now I'm going to do something slightly strange. I'm going to change this bias from 8 to this number 6.88 etc. I'm going to do this for specific reasons because we're about to start back propagation, and I want to make sure that our numbers come out nice. They're not like very crazy numbers, they're nice numbers that we can sort of understand in our head. Let me also add a pose label, `o` is short for output here, so that's zero. Okay so, 0.88 flows into `tanh` comes out 0.7 so on.

So now we're going to do back propagation and we're going to fill in all the gradients. What is the derivative `o` with respect to all the inputs here? Of course, in the typical neural network setting, what we really care about the most is the derivative of these neurons on the weights, specifically the `w2` and `w1` because those are the weights that we're going to be changing as part of the optimization.

The other thing that we have to remember is here we have only a single neuron but in the neural networks, typically have many neurons and they're connected. So this is only like a one small neuron, a piece of a much bigger puzzle. Eventually, there's a loss function that sort of measures the accuracy of the neural net and we're back propagating with respect to that accuracy and trying to increase it.

So let's start off by propagation here in the end. What is the derivative of `o` with respect to `o`? The base case sort of we know always is that the gradient is just 1.0. So let me fill it in and then let me split out the drawing function here, clear this output here. Okay, so now when we draw `o` we'll see that `o.grad` is one.

So now we're going to back propagate through the `tanh`. So to back propagate through `tanh` we need to know the local derivative of `tanh`. So if we have that `o` is `tanh` of `n`, then what is `do/dn`? Now what you could do is you could come here and you could take this expression and you could do your calculus derivative taking and that would work. But we can also just scroll down Wikipedia here into a section that hopefully tells us.
# Understanding Derivatives and Backpropagation

Let's start by understanding the derivative, `d/dx`, of `10h(x)`. Among the various options, I prefer this one: `1 - 10h^2(x)`. So, this simplifies to `1 - 10h(x)^2`. 

What this essentially means is that `dO/dN` equals `1 - 10h(N)^2`. We already know that `10h(N)` is just `O`, so it's `1 - O^2`. Here, `O` is the output, so the output is this number, `data`. 

This implies that `dO/dN` is `1 - data^2`. Conveniently, `1 - data^2` equals `0.5`. Therefore, the local derivative of this `10h` operation is `0.5`, which would be `dO/dN`. 

So, this is exactly `0.5` or `one-half`. Now, we're going to continue the backpropagation. This is `0.5` and this is a plus node. 

You might wonder, how is backpropagation going to work here? If you remember our previous example, a plus is just a distributor of gradient. So, this gradient will simply flow to both of these equally. That's because the local derivative of this operation is one for every one of its nodes. So, `1 * 0.5` is `0.5`. 

Therefore, we know that this node here, which we called `this`, its gradient is just `0.5`, and we know that `b.grad` is also `0.5`. 

Continuing, we have another plus. `0.5` again will just distribute it, so `0.5` will flow to both of these. So, we can set `x2w2.grad` as `0.5`. 

Let's redraw. Pluses are my favorite operations to backpropagate through because it's very simple. Now, it's flowing into these expressions is `0.5`. 

Keep in mind what the derivative is telling us at every point in time along here. This is saying that if we want the output of this neuron to increase, then the influence on these expressions is positive on the output. Both of them are positive. 

Now, backpropagating to `x2` and `w2`, this is a times node, so we know that the local derivative is the other term. So, if we want to calculate `x2.grad`, it's going to be `w2.data * x2w2.grad`. And `w2.grad` will be `x2.data * x2w2.grad`. That's the local piece of the chain rule. 

Let's set them and redraw. Here, we see that the gradient on our weight `2` is `0` because `x2.data` was `0`. But `x2` will have the gradient `0.5` because `data` here was `1`. 

What's interesting here is that because the input `x2` was `0`, then because of the way the times works, this gradient will be zero. Think about intuitively why that is. The derivative always tells us the influence of this on the final output. If I wiggle `w2`, how is the output changing?
# Understanding Backpropagation and Derivatives

Let's start by understanding why the value isn't changing. The reason is that we're multiplying by zero. So, because it's not changing, there's no derivative and zero is the correct answer. This is because we're squashing it at zero.

Now, let's do it here. Point five should come here and flow through this times. So, we'll have that `x1.grad` is. Can you think through a little bit what the local derivative of times with respect to `x1` is going to be? It's going to be `w1`. So `w1` is `data` times `x1`, `w1.dot.grad`, and `w1.grad` will be `x1.data` times `x1`, `w2`, `w1` with graph.

Let's see what those came out to be. So this is 0.5, so this would be negative 1.5, and this would be 1. We've backpropagated through this expression. These are the actual final derivatives. So, if we want this neuron's output to increase, we know that what's necessary is that `w2` has no gradient. `w2` doesn't actually matter to this neuron right now. But this neuron, this weight, should go up. So, if this weight goes up, then this neuron's output would have gone up and proportionally because the gradient is one.

Doing the backpropagation manually is obviously ridiculous. So, we're now going to put an end to this suffering and we're going to see how we can implement the backward pass a bit more automatically. We're not going to be doing all of it manually out here. It's now pretty obvious to us by example how these pluses and times are backpropagating gradients.

Let's go up to the `value` object and we're going to start codifying what we've seen in the examples below. We're going to do this by storing a special `self.backward` and `_backward`. This will be a function which is going to do that little piece of chain rule at each little node that computed inputs and produced output. We're going to store how we are going to chain the output's gradient into the input's gradients.

By default, this will be a function that doesn't do anything. This is an empty function, and that would be the case for a leaf node. For a leaf node, there's nothing to do. But now, when we're creating these `out` values, these `out` values are an addition of `self` and `other`. So, we will want to set `out.backward` to be the function that propagates the gradient.

We're going to store it in a closure. Let's define what should happen when we call for an addition. Our job is to take `out.grad` and propagate it into `self.grad` and `other.grad`. So basically, we want to set `self.grad` to something and we want to set `other.grad` to something.

The way we saw below how the chain rule works, we want to take the local derivative times the global derivative, which is the derivative of the final output of the expression with respect to `out.data`.
# Understanding the Local Derivative of Self in Addition and Multiplication Operations

In this article, we will discuss the local derivative of self in addition and multiplication operations. We will also delve into the chain rule and how it applies to these operations.

The local derivative of self in an addition operation is 1.0. Therefore, it's simply 1.0 times the `outs grad`. This is a direct application of the chain rule. Similarly, `others.grad` will be 1.0 times `outgrad`. Essentially, `outgrad` will be copied onto `selfs grad` and `others grad` as we saw happens for an addition operation. We will later call this function to propagate the gradient.

Having discussed addition, let's now move on to multiplication. We're going to set its backward to chain `outgrad` into `self.grad` and `others.grad`. This will be a small piece of the chain rule for multiplication. The local derivative here is `others.data` times `outgrad`. This is the chain rule in action. Similarly, we have `self.data` times `outgrad`.

Finally, let's discuss the `tanh` function. We want to set `out` backwards to back propagate. We have `outgrad` and `self.grad` will be the local derivative of this operation, which is `tanh`. The local gradient is 1 minus the `tanh` of x squared, which here is `t`. That's the local derivative because `t` is the output of this `tanh` so 1 minus `t` squared is the local derivative. The gradient has to be multiplied because of the chain rule. So `outgrad` is chained through the local gradient into `self.grad`.

To implement this, we're going to redefine our value node and our expression. We need to ensure that all the grads are zero. However, we don't have to do this manually anymore. We are going to be calling the `dot backward` in the right order.

First, we want to call `o's` backward. `o` was the outcome of `tanh`. Calling `o's` backward will execute this function. However, we have to be careful because there's a times `out.grad`, and `out.grad` is initialized to zero. As a base case, we need to set `out.grad` to 1.0. Once this is 1, we can call `o's` backward. This should propagate this grad through `tanh`. The local derivative times the global derivative, which is initialized at one, should adopt.

However, we need to be careful not to call the function because that returns none. These functions return none; we just want to store the function. After redefining the value object, we can proceed.

In conclusion, understanding the local derivative of self in addition and multiplication operations is crucial in understanding how the chain rule applies to these operations. This knowledge is fundamental in the field of calculus and has wide applications in various fields, including machine learning and data science.
# Understanding Backward Propagation

Let's redefine the expression "draw a dot". Everything is great, `o.dot.grad` is one. `o.dot.grad` is one and now, this should work of course. Okay, so all that backward should be working fine. This grant should now be 0.5 if we redraw and if everything went correctly. 

And it's not awkward, sorry. The process ends backward. So instead, backward routed the gradient to both of these, so this is looking great. Now we could, of course, call `b.grad`. But what's going to happen? Well, `b` doesn't have it backward, `b` is backward because `b` is a leaf node. `b's` backward is by initialization the empty function, so nothing would happen but we can call it on it. 

But when we call this one, then we expect this 0.5 to get further routed. Right, so there we go 0.5.5. And then finally, we want to call both of those, and there we go. So we get 0, 0.5, negative 1.5, and 1, exactly as we did before but now, we've done it through calling that backward manually. 

So we have the last one last piece to get rid of which is us calling `_backward` manually. So let's think through what we are actually doing. We've laid out a mathematical expression and now we're trying to go backwards through that expression. Going backwards through the expression just means that we never want to call `a.dot.backward` for any node before we've done everything after it. 

So we have to do everything after it before we're ever going to call that backward on any one node. We have to get all of its full dependencies, everything that it depends on has to propagate to it before we can continue back propagation. 

This ordering of graphs can be achieved using something called topological sort. Topological sort is basically a laying out of a graph such that all the edges go only from left to right. Here we have a graph, it's a directory acyclic graph (DAG), and this is two different topological orders of it. You'll see that it's laying out of the notes such that all the edges go only one way from left to right. 

Implementing topological sort can be found on Wikipedia and so on. I'm not going to go through it in detail, but basically, this is what builds a topological graph. We maintain a set of visited nodes and then we are going through starting at some root node, which for us is `o`. That's where we want to start the topological sort. 

Starting at `o`, we go through all of its children and we need to lay them out from left to right. This starts at `o`, if it's not visited then it marks it as visited and then it iterates through all of its children and calls build topological on them. After it's gone through all the children, it adds itself. 

This node that we're going to call it on, like say `o`, is only going to add itself to the topo list after all of the children have been processed. That's how this function is guaranteeing that you're only going to be in the list once all your children are in the list.
# Understanding Back Propagation

Let's start by understanding the invariant that is being maintained. If we build upon `o` and then inspect this list, we're going to see that it ordered our value objects. The last one is the value of `0.707`, which is the output. So, this is `o` and then this is `n`, and then all the other nodes get laid out before it. 

This process builds the topological graph. What we're doing now is we're just calling `dot._backward` on all of the nodes in a topological order. If we reset the gradients, they're all zero. 

So, what did we do? We started by setting `o.grad` to `b1`. That's the base case. Then we went for `node` in `reversed` of `topo`. Now, in the reverse order because this list goes from, we need to go through it in reversed order. So starting at `o`, note that `backward`, and this should be it. There we go, those are the correct derivatives. 

Finally, we are going to hide this functionality. So, I'm going to copy this and we're going to hide it inside the `valley` class because we don't want to have all that code lying around. So instead of an `_backward`, we're now going to define an actual `backward`. So that's `backward` without the underscore, and that's going to do all the stuff that we just arrived. 

Let me just clean this up a little bit. So, build a topological graph starting at `self`. So `build topo` of `self` will populate the topological order into the `topo` list which is a local variable. Then we set `self.grad` to be one. Then for each node in the reversed list, so starting at us and going to all the children, `_backward`. And that should be it. 

Now, all the gradients are zero, and now what we can do is `o.backward` without the underscore. And there we go, that's back propagation in place for one neuron. 

However, we shouldn't be too happy with ourselves because we have a bug. We have not surfaced the bug because of some specific conditions that we have to think about right now. Here's the simplest case that shows the bug. 

Let's say I create a `b` that is `a` plus `a`. So what's going to happen is `a` is `3`, and then `b` is `a` plus `a` so there's two. Then we can see that `b` is, of course, the forward pass works, `b` is just `a` plus `a` which is `6`. But the gradient here is not actually correct that we calculated automatically. 

That's because, of course, just doing calculus in your head, the derivative of `b` with respect to `a` should be `2` (one plus one), it's not `1`. Intuitively, what's happening here is right.
# Understanding Backward Propagation and Gradient Accumulation

In this article, we will discuss the concept of backward propagation and gradient accumulation in the context of neural networks. 

Let's start with a simple example. Suppose `b` is the result of `a` plus `a`. When we call `backward` on `b`, `self.grad` is set to one. However, because we're doing `a` plus `a`, `self` and `other` are actually the exact same object. So, we are overriding the gradient by setting it to one and then setting it again to one. That's why it stays at one. This is a problem.

To illustrate this further, let's consider another example. Here, we have `a` and `b`. Then, `d` will be the multiplication of the two and `e` will be the addition of the two. We then multiply `e` times `d` to get `f` and then we call `f.backward()`. If you check, these gradients will be incorrect.

Fundamentally, what's happening here is that we're going to see an issue anytime we use a variable more than once. Until now, in these expressions above, every variable is used exactly once so we didn't see the issue. But here, if a variable is used more than once, what's going to happen during the backward pass is that we're backpropagating from `f` to `e` to `d` so far so good. But now, `e.backward()` is called and it deposits its gradients to `a` and `b`. But then we come back to `d` and call `d.backward()` and it overwrites those gradients at `a` and `b`. That's obviously a problem.

The solution to this problem lies in the multivariate case of the chain rule and its generalization. We have to accumulate these gradients. These gradients add. So instead of setting those gradients, we can simply do `+=`. We need to accumulate those gradients. Remember, we are initializing them at zero so they start at zero and then any contribution that flows backwards will simply add.

So now, if we redefine this with `+=`, this now works because `a.grad` started at zero and we called `b.backward()`, we deposit one and then we deposit one again and now this is two which is correct. 

Similarly, in the second example, this will also work and we'll get correct gradients. When we call `e.backward()`, we will deposit the gradients from this branch and then we get back to `d.backward()`, it will deposit its own gradients and then those gradients simply add on top of each other. So, we just accumulate those gradients and that fixes the issue.

Before we move on, let's do a bit of cleanup here and delete some of the intermediate work. We are going to keep the definition of `value` as we will come back to it later.

In conclusion, understanding the concept of backward propagation and gradient accumulation is crucial in the field of neural networks. It helps us to understand how the gradients flow and accumulate during the backward pass, which is essential for the training of neural networks.
# Understanding the Tanh Function

In our previous discussion, we mentioned that we could have broken down the Tanh function into its explicit atoms in terms of other expressions if we had the `x` function. As you may recall, Tanh is defined in a certain way and we chose to develop Tanh as a single function. We can do this because we know its derivative and we can back propagate through it. 

However, we can also break down Tanh into and express it as a function of `x`. I would like to do that now because I want to prove to you that you get all the same results and all those ingredients. But also because it forces us to implement a few more expressions. It forces us to do exponentiation, addition, subtraction, division, and things like that. I think it's a good exercise to go through a few more of these.

Let's scroll up to the definition of value. Here, one thing that we currently can't do is we can do like a value of say `2.0`, but we can't do something like this:

```python
a = Value(2.0)
b = a + 1
```

We can't do it because it says "object has no attribute data". That's because `a + 1` comes right here to `add`, and then `other` is the integer `1` and then here Python is trying to access `1.data` and that's not a thing. That's because basically `1` is not a `Value` object and we only have addition for `Value` objects. 

As a matter of convenience, so that we can create expressions like this and make them make sense, we can simply do something like this:

```python
class Value:
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data)
```

Basically, we let `other` alone if `other` is an instance of `Value` but if it's not an instance of `Value` we're going to assume that it's a number like an integer float and we're going to simply wrap it in `Value` and then `other` will just become `Value` of `other` and then `other` will have a `data` attribute and this should work.

Let's do the exact same thing for `multiply` because we can't do something like this:

```python
a = Value(2.0)
b = a * 2
```

Again, for the exact same reason. So we just have to go to `mul` and if `other` is not a `Value` then let's wrap it in `Value`.

Now here's a kind of unfortunate and not obvious part. `a * 2` works, we saw that, but `2 * a` is that gonna work? You'd expect it to right? But actually it will not. And the reason it won't is because Python doesn't know how to multiply `Value` by `2`. 

So instead what happens is in Python, the way this works is you are free to define something called the `__rmul__`. `__rmul__` is kind of like a fallback. So if Python can't do `2 * a` it will check if by any chance `a` knows how to multiply `2` and that will be called into `__rmul__`. 

So because Python can't do `2 * a`, it will check is there an `__rmul__` in `Value` and because there is, it will now call that. And what we'll do here is we will swap the order of the operands. So basically `2 * a` will redirect to `__rmul__` and `__rmul__` will basically call `a * 2`. And that's how that will work.

Looking at the other elements that we still need, we...
# Understanding Exponentiation and Division in Python

In this tutorial, we will explore how to exponentiate and divide in Python. We will start with the explanation of exponentiation.

## Exponentiation

We will introduce a single function `x` here. `x` is going to mirror `10h` in the sense that it's a simple single function that transforms a single scalar value and outputs a single scalar value. 

In Python, we use `math.x` to exponentiate it and create a new value object. The tricky part, of course, is how do you propagate through `e` to the `x`. 

So, what is the local derivative of `e` to the `x`? The derivative `d/dx` of `e` to the `x` is famously just `e` to the `x`. We've already just calculated `e` to the `x` and it's inside `out.data`. So we can do `out.data * out.grad`. That's the chain rule. We're just chaining on to the current running `grad`. 

This is what the expression looks like. It might look a little confusing, but that's the exponentiation. 

## Division

The last thing we'd like to do, of course, is to be able to divide. I will implement something slightly more powerful than division because division is just a special case of something a bit more powerful. 

If we have some kind of `a/b = value` of `4.0`, we'd like to be able to do `a/b` and we'd like this to give us `0.5`. 

Division can be reshuffled as follows: if we have `a/b`, that's actually the same as `a * (1/b)`, and that's the same as `a * b^-1`. 

What I'd like to do instead is to implement the operation of `x` to the `k` for some constant `k` (an integer or a float), and we would like to be able to differentiate this. As a special case, `-1` will be division. I'm doing that just because it's more general. 

We can redefine division as `self/other` can actually be rewritten as `self * other^-1`. A value raised to the power of `-1` we have now defined that. 

We need to implement the `pow` function. This function will be called when we try to raise a value to some power and `other` will be that power. I'd like to make sure that `other` is only an `int` or a `float`. Usually, `other` is some kind of a different value object, but here `other` will be forced to be an `int` or a `float` otherwise the math won't work. 

We create the output value which is just this data raised to the power of `other`.
# Understanding the Power of Back Propagation

In this article, we will explore the power of back propagation and how it can be implemented in Python. We will start by defining the power function, where the power is a constant. We will then look at how to implement the chain rule for back propagation through this power function. 

Let's start by defining the power function. Here, 'other' represents the power to which we are raising our base. For example, 'other' could be negative one, which is what we are hoping to achieve. 

```python
def __pow__(self, other):
    return Val(self.data ** other, (self, other))
```

Next, we will define the backward step. This is where the fun part begins - defining the chain rule expression for back propagating through the power function. 

```python
def backward(self, output_grad=1):
    self.grad += output_grad
    for v, grad in self._prev:
        v.backward(grad * output_grad)
```

If you are unsure about what to put here, you can refer to the derivative rules from calculus. In particular, we are looking for the power rule. The power rule tells us that if we are trying to take the derivative of x to the power of n, it is just n times x to the power of n minus 1. 

This tells us about the local derivative of this power operation. In our case, 'n' is 'other' and 'self.data' is 'x'. So, our expression becomes:

```python
other * (self.data ** (other - 1))
```

This is the local derivative only. We now have to chain it, which we do simply by multiplying by 'output_grad'. This is the chain rule. 

```python
other * (self.data ** (other - 1)) * output_grad
```

This should technically work, and we will find out soon. 

Now, let's move on to subtraction. Right now, 'a - b' will not work. To make it work, we need to add one more piece of code. We will implement subtraction by addition of a negation. To implement negation, we will multiply by negative one. 

```python
def __sub__(self, other):
    return self + (-1 * other)
```

Now, 'a - b' should work. 

Finally, let's look at a two-dimensional neuron that has a tanh function. We will compute the backward pass and draw the gradients for all the leaf nodes. 

```python
o = tanh(n)
o.backward()
draw()
```

We can break up the tanh function into the following expression: 

```python
e = exp(2 * n)
o = (e - 1) / (e + 1)
```

Here, 'e' is 'e to the power of 2x'. We create an intermediate variable 'e' because we are using it twice. 

This is how we can implement back propagation through the power function in Python.
# Understanding the Forward and Backward Passes in Deep Learning

In this article, we will delve into the intricacies of forward and backward passes in deep learning. We will start by examining a mathematical expression, `e - 1 / e + 1`, and then proceed to break it down into its constituent operations. 

Our expectation is to see a much longer graph because we've broken up `10h` into several other operations. However, these operations are mathematically equivalent. Therefore, we expect to see the same result for the forward pass and, due to the mathematical equivalence, the same backward pass and the same gradients on the leaf nodes. 

Let's run this and verify our expectations. 

Firstly, instead of a single `10h` node, we now have `x`, `+`, `* -1`, and a division operation. Despite this, we end up with the same forward pass result. 

Secondly, we need to be careful with the gradients as they might be in a slightly different order. The gradients for `w2x2` should be `0` and `0.5`, and indeed, `w2` and `x2` are `0` and `0.5`. Similarly, `w1` and `x1` are `1` and `-1.5`, just as expected. 

This means that both our forward and backward passes were correct because this turned out to be equivalent to `10h` before. 

The reason I wanted to go through this exercise is twofold. Firstly, we got to practice a few more operations and writing more backward passes. Secondly, I wanted to illustrate the point that the level at which you implement your operations is entirely up to you. 

You can implement backward passes for tiny expressions like a single individual `+` or `*`, or you can implement them for `10h`, which is a composite operation made up of all these more atomic operations. 

However, all of this is a conceptual construct. All that matters is that we have some inputs and some output, and this output is a function of the inputs in some way. As long as you can do a forward pass and the backward pass of that operation, it doesn't matter what that operation is and how composite it is. If you can write the local gradients, you can chain the gradient and continue back propagation. The design of what those functions are is entirely up to you. 

Now, I would like to show you how you can do the exact same thing using a modern deep neural network library like PyTorch, which I've roughly modeled Micrograd by. 

PyTorch is something you would use in production, and I'll show you how you can do the exact same thing but in PyTorch API. 

In PyTorch, we need to define these value objects like we have here. Now, Micrograd is a scalar-valued engine, so we only have scalar values like `2.0`. But in PyTorch, everything is based around tensors. Tensors are just n-dimensional arrays of scalars. 

That's why things get a little bit more complicated here. I just need a scalar value to tensor, a tensor with just a single element. But by default, when you work with PyTorch, you would use more complicated tensors. 

In conclusion, understanding the forward and backward passes in deep learning is crucial. It allows you to implement operations at any level, depending on your needs. Whether you're working with scalar values or tensors, the principles remain the same.
# Working with PyTorch

In this tutorial, we will be working with PyTorch, a popular open-source machine learning library for Python. We will be creating a tensor that has only a single element, 2.0, and then casting it to be double because Python uses double precision for its floating point numbers by default. 

```python
import torch

x1 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
x2 = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
x3 = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
```

By default, the data type of these tensors will be `float32`, so it's only using a single precision float. We are casting it to double so that we have `float64` just like in Python. 

The next thing we have to do is because these are leaf nodes, by default PyTorch assumes that they do not require gradients. So, we need to explicitly say that all of these nodes require gradients. This is going to construct scalar-valued, one-element tensors and make sure that PyTorch knows that they require gradients. 

By default, these are set to `False` for efficiency reasons because usually, you would not want gradients for leaf nodes like the inputs to the network. This is just trying to be efficient in the most common cases.

Once we've defined all of our values in Python, we can perform arithmetic just like we can here in microgradlend. 

```python
o = (x1 * x2) + torch.sin(x3)
```

What we get back is a tensor again and we can just like in micrograd, it's got a `data` attribute and it's got `grad` attributes. These tensor objects just like in micrograd have a `.data` and a `.grad` attribute. The only difference here is that we need to call `.item()` because otherwise, PyTorch `.item()` basically takes a single tensor of one element and it just returns that element stripping out the tensor.

```python
print('forward pass: ', o.data.item())
o.backward()
print('backward pass: ', x1.grad.item(), x2.grad.item(), x3.grad.item())
```

So, PyTorch agrees with us and just to show you here basically `o` here's a tensor with a single element and it's a double, and we can call `.item()` on it to just get the single number out. So that's what `.item()` does and `o` is a tensor object like I mentioned and it's got a `backward` function just like we've implemented. 

All of these also have a `.grad` so like `x2` for example in the `grad` and it's a tensor and we can pop out the individual number with `.item()`. 

So basically, PyTorch can do what we did in micrograd as a special case when your tensors are all single-element tensors. But the big deal with PyTorch is that everything is significantly more efficient because we are working with these tensor objects and we can do lots of operations in parallel on all of these tensors. 

What we've built very much agrees with the API of PyTorch. Now that we have some machinery to build out pretty complicated mathematical expressions, we can also...
# Building Neural Nets

In this tutorial, we will start building out neural nets. As previously mentioned, neural nets are just a specific class of mathematical expressions. We're going to start building out a neural net piece by piece and eventually, we'll build out a two-layer multi-layer perceptron, also known as a layer perceptron. I'll show you exactly what that means.

Let's start with a single individual neuron. We've implemented one here, but now, I'm going to implement one that also subscribes to the PyTorch API in how it designs its neural network modules. Just like we saw that we can match the API of PyTorch on the autograd side, we're going to try to do that on the neural network modules side.

Here's the class neuron. For the sake of efficiency, I'm going to copy-paste some sections that are relatively straightforward. The constructor will take the number of inputs to this neuron, which is how many inputs come to a neuron. For example, this one has three inputs. Then, it's going to create a weight, which is some random number between negative one and one for every one of those inputs, and a bias that controls the overall trigger happiness of this neuron.

Next, we're going to implement a `def __call__` of self and x, some input x. What we do here is `w` times `x` plus `b`, where `w` times `x` here is a dot product specifically. If you haven't seen `call`, let me explain. The way this works now is we can have an `x` which is, say like 2.0, 3.0, then we can initialize a neuron that is two-dimensional because these are two numbers. Then, we can feed those two numbers into that neuron to get an output. When you use this notation `n` of `x`, Python will use `call`.

Now, we'd like to actually do the forward pass of this neuron instead. What we're going to do here first is we need to basically multiply all of the elements of `w` with all of the elements of `x` pairwise. We need to multiply them. The first thing we're going to do is we're going to zip up `self.w` and `x`. In Python, `zip` takes two iterators and it creates a new iterator that iterates over the tuples of the corresponding entries.

For `w_i`, `x_i` in `zip(self.w, x)`, we want to multiply `w_i` times `x_i`, and then we want to sum all of that together to come up with an activation and add also `self.b` on top. That's the raw activation and then of course, we need to pass that through a non-linearity. What we're going to be returning is `act.tanh()`. 

To be a bit more efficient here, `sum` by the way takes a second optional parameter which is the start and by default, the start is zero. These elements of this sum will be added on top of zero to begin with but actually, we can just start with `self.b`. Then, we just have an expression like this and then the generator expression here must be parenthesized in Python. 

Now, we see that we are getting some outputs and we get a different output from a neuron each time because we are initializing different weights and biases.
# Defining a Layer of Neurons

Next, we're going to define a layer of neurons. Here, we have a schematic for a Multi-Layer Perceptron (MLP). We see that each layer of these MLPs has a number of neurons. They're not connected to each other, but all of them are fully connected to the input.

So, what is a layer of neurons? It's just a set of neurons evaluated independently. In the interest of time, I'm going to do something fairly straightforward here. A layer is just a list of neurons. We take the number of neurons as an input argument. This is the number of outputs in this layer. We then initialize completely independent neurons with this given dimensionality. When we call on it, we just independently evaluate them.

Now, instead of a neuron, we can make a layer of neurons. Let's say they are two-dimensional neurons and let's have three of them. Now we see that we have three independent evaluations of three different neurons.

# Defining a Multi-Layer Perceptron (MLP)

Finally, let's complete this picture and define an entire Multi-Layer Perceptron (MLP). As we can see here, in an MLP, these layers just feed into each other sequentially.

An MLP is very similar to a layer of neurons. We're taking the number of inputs as before, but now instead of taking a single `n_out` (which is the number of neurons in a single layer), we're going to take a list of `n_outs`. This list defines the sizes of all the layers that we want in our MLP. We then put them all together and iterate over consecutive pairs of these sizes to create layer objects. In the call function, we are just calling them sequentially. That's an MLP, really.

Let's re-implement this picture. We want three input neurons, then two layers of four, and one output unit. This, of course, is an MLP. And there we go, that's a forward pass of an MLP.

To make this a little bit nicer, you see how we have just a single element but it's wrapped in a list because a layer always returns lists. For convenience, we can return `outs[0]` if `len(out)` is exactly a single element, else return the full list. This will allow us to just get a single value out at the last layer that only has a single neuron.

Finally, we should be able to draw the dot of `n` of `x`. As you might imagine, these expressions are now getting relatively involved. This is an entire MLP that we're evaluating all the way until a single output.

Obviously, you would never differentiate on pen and paper these expressions. But with Micrograd, we will be able to backpropagate all the way through this and backpropagate into these weights of all these neurons. Let's see how that works.

Let's create ourselves a very simple example dataset. This dataset has four examples.
# Understanding Neural Networks: A Simple Binary Classifier

We have four possible inputs into the neural network, and four desired targets. We'd like the neural network to assign or output 1.0 when it's fed a specific example, -1 when it's fed other examples, and 1 when it's fed another example. So, it's a very simple binary classifier neural network that we would like to create here.

Now, let's consider what the neural network currently thinks about these four examples. We can just get their predictions by calling `n of x` for `x in axis`, and then print the results. These are the outputs of the neural network on those four examples.

The first one is 0.91, but we'd like it to be one, so we should push this one higher. The second one we also want to be higher. The third one says 0.88 and we want this to be -1. The fourth is 0.8, we want it to be -1, and the last one is 0.8, we want it to be 1.

So, how do we make the neural network, and how do we tune the weights to better predict the desired targets? The trick used in deep learning to achieve this is to calculate a single number that somehow measures the total performance of your neural network. We call this single number the loss.

The loss is a single number that we're going to define that basically measures how well the neural network is performing right now. We have the intuitive sense that it's not performing very well because we're not very much close to the desired targets. So the loss will be high, and we'll want to minimize the loss.

In this case, we're going to implement the mean squared error loss. What we're going to do is iterate over the ground truths and the predictions, pair them up, and for each pair, we're going to subtract them and square the result.

Let's first see what these losses are. These are individual loss components. For each one of the four examples, we are taking the prediction and the ground truth, subtracting them, and squaring the result. Because the first example is so close to its target (0.91 is almost 1), subtracting them gives a very small number. Squaring it just makes sure that regardless of whether we are more negative or more positive, we always get a positive number. We could also take, for example, the absolute value to discard the sign.

The expression is arranged so that you only get zero exactly when your output is equal to your ground truth. When those two are equal, so your prediction is exactly the target, you are going to get zero. If your prediction is not the target, you are going to get some other number. For example, we are way off in the third example, and that's why the loss is quite high. The more off we are, the greater the loss will be. We don't want a high loss, we want a low loss. So, the final loss here will be just the average of these individual loss components.
# Understanding Neural Networks: Minimizing Loss and Backward Pass

In our exploration of neural networks, we've been discussing the concept of loss. The sum of all the numbers in our loss function should ideally be zero. However, in practice, we often find that it's not zero, but a positive number. For instance, it could be seven. This means that our loss is about seven. 

Our goal is to minimize this loss. We want the loss to be as low as possible because a low loss indicates that each of our predictions is equal to its target. The lowest the loss can be is zero. The greater the loss, the worse off the neural network is at predicting. 

To minimize the loss, we perform a backward pass. When we execute this backward pass, something magical happens. We can look at the individual neurons in each layer of our neural network. Each layer in our Multi-Layer Perceptron (MLP) has neurons, and each neuron has weights. 

For instance, we can look at the weights of the first neuron in the first layer. These weights are represented by the variable 'w'. After the backward pass, this 'w' value now has a gradient. This gradient tells us how this particular weight influences the loss. 

If the gradient is negative, it means that slightly increasing this particular weight would make the loss go down. We have this information for every single one of our neurons and all their parameters. 

It's also worth noting the complexity of the loss function. We previously looked at the forward pass of a single neuron, which was already a complex expression. However, the loss function is even more complex. It includes the forward pass of every single example, and then the mean squared error loss on top of them. 

This results in a massive graph. This graph starts with the forward pass of each example, includes the loss on top of them, and ends with the value of the loss. The backward pass then propagates this loss through all the forward passes, all the way back to the weights, which are the inputs to the neural network. 

Interestingly, the input data also has gradients. However, these gradients are not useful to us because the input data is fixed and unchangeable. We can't modify it, even though we have gradients for it. 

On the other hand, the gradients for the neural network parameters, the weights (w) and biases (b), are extremely useful. These gradients help us adjust the parameters to minimize the loss and improve the performance of our neural network.
# Neural Network Parameters and Gradients

In this tutorial, we will be discussing how to gather all the parameters of a neural network so that we can operate on all of them simultaneously. Each parameter will be nudged a tiny amount based on the gradient information. 

Let's start by collecting the parameters of the neural network all in one array. We will create a `parameters` method in our `Neuron` class that returns `self.w` and `self.b`:

```python
def parameters(self):
    return self.w + self.b
```

This method will return a list of all the parameters of the neuron. The `+` operator is used here to concatenate the two lists. 

This method is named `parameters` because PyTorch also has a `parameters` method on every `nn.Module` which does exactly what we're doing here - it returns the parameter tensors for us.

Now, let's move on to our `Layer` class. This class is also a module, so it will have its own parameters. We want to gather all these parameters in one list. Here's how we can do it:

```python
def parameters(self):
    params = []
    for neuron in self.neurons:
        params.extend(neuron.parameters())
    return params
```

However, this code is a bit verbose. We can simplify it using a nested list comprehension:

```python
def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
```

This single line of code does exactly the same thing as the previous code block. It iterates over each neuron in the layer, gets its parameters, and adds them to the list.

Now, let's apply the same logic to our `NeuralNet` class:

```python
def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
```

Now, we can get all the parameters of the entire neural network with a single method call:

```python
n = NeuralNet(...)
params = n.parameters()
```

These are all the weights and biases inside the entire neural network. 

Now, let's say we have calculated the loss and we want to update the parameters of the network based on the gradients. We can do this by iterating over each parameter and nudging its value slightly according to its gradient:

```python
for p in n.parameters():
    p.data -= learning_rate * p.grad
```

In this line of code, `learning_rate` is a small constant that determines how much we want to change the parameters. This is the basic idea behind gradient descent, which is the most common optimization algorithm in deep learning.

And that's it! We have successfully gathered all the parameters of a neural network and updated them based on the gradients. This is a crucial step in training a neural network.
# Gradient Descent Scheme in Neural Networks

In this article, we will discuss a tiny update in the gradient descent scheme in neural networks. In gradient descent, we think of the gradient as a vector pointing in the direction of increased loss. We modify the data by taking a small step in the direction of the gradient. 

For instance, the step size could be a very small number like 0.01. This step size is then multiplied by the gradient of the data. However, we need to consider the signs here. 

Let's take a specific example. If we just left it like this, the value of a neuron would be increased by a tiny amount of the gradient. If the gradient is negative, the value of this neuron would decrease slightly. But if this neuron's value goes lower, it would actually increase the loss. 

This is because the derivative of this neuron is negative. So, increasing this makes the loss go down. Therefore, we want to increase it instead of decreasing it. Essentially, we are missing a negative sign here. 

We want to minimize the loss, not maximize it. Another interpretation is that the gradient vector, which is just the vector of all the gradients, points in the direction of increasing the loss. But we want to decrease it, so we actually want to go in the opposite direction. 

You can convince yourself that this approach does the right thing here with the negative sign because we want to minimize the loss. If we nudge all the parameters by this, we'll see that the data will have changed a little bit. 

For example, a neuron's value might increase from 0.854 to 0.857. This is a good thing because slightly increasing this neuron's data makes the loss go down according to the gradient. So, the correct thing has happened sign-wise. 

Now, because we've changed all these parameters, we expect that the loss should have gone down a bit. We want to re-evaluate the loss. If we recalculate the loss, we'd expect the new loss now to be slightly lower than before. 

For instance, the loss might decrease from 4.84 to 4.36. Remember, the way we've arranged this is that a low loss means that our predictions are matching the targets. So, our predictions now are probably slightly closer to the targets. 

All we have to do now is iterate this process. We've done the forward pass and calculated the loss. Now we can calculate the backward loss. We repeat this process until we achieve the desired results.
# Training a Neural Network with Gradient Descent

In this tutorial, we will be training a neural network using gradient descent. We will be using a simple forward pass and backward pass to iteratively improve the neural network's predictions.

Let's start by defining our data. We will then perform a forward pass and calculate the loss. After that, we will perform a backward pass to update the parameters of our neural network. This process is repeated iteratively to improve the predictions of our neural network.

```python
# Data definition
...

# Forward pass
...

# Calculate loss
...

# Backward pass
...

# Update parameters
...
```

After performing a few iterations, we should see a slightly lower loss. For example, a loss of 4.36 might go down to 3.9. 

The idea is to continue this process iteratively. This is known as gradient descent. We are just iteratively doing a forward pass, backward pass, and updating the parameters. 

As we continue this process, the neural network improves its predictions. For instance, if we look at the predicted values, we should see that they are getting closer to the actual values. Positive values should be getting more positive, and negative values should be getting more negative.

However, we need to be careful with the step size. If we go too fast, we might step into a part of the loss function that is completely different, which can destabilize training and make the loss blow up. 

On the other hand, if the step size is too small, the convergence will take too long. Finding the right step size is a subtle art. It's about setting the learning rate just right. If it's too low, it will take too long to converge. If it's too high, the whole thing gets unstable, and the loss might even explode.

After several iterations, we should see a very low loss, and the predictions should be almost perfect. We can look at the parameters of our neural network to see the setting of weights and biases that makes our network predict the desired targets very closely.

```python
# Look at the parameters
n.parameters
```

In conclusion, we have successfully trained a neural network using gradient descent. 

In the next tutorial, we will implement an actual training loop to make this process more efficient. We will re-initialize the neural network from scratch and perform several iterations of forward pass, backward pass, and parameter updates.

```python
# Re-initialize the neural network
...

# Training loop
for k in range(...):
    # Forward pass
    ...
    
    # Backward pass
    ...
    
    # Update parameters
    ...
```

Stay tuned for more!
# Neural Network Optimization

First, we load the data. Then, we perform a forward pass, followed by an update using gradient descent. We should be able to iterate this process and print the current step and the current loss. Let's just print the number of the loss. 

The learning rate of 0.01 is a little too small, and 0.1 is a little bit dangerously too high. Let's go somewhere in between. We'll optimize this for not 10 steps, but let's go for say 20 steps. 

As you can see, we've actually converged slower in a more controlled manner and got to a loss that is very low. So, I expect the results to be quite good. 

However, there is a subtle bug in our code, which is a very common one. I've actually tweeted about the most common neural net mistakes a long time ago. We are guilty of number three: forgetting to zero the gradient. 

All of these weights here have a `.data` and a `.grad`. The gradient starts at zero, then we do backward and fill in the gradients. We then do an update on the data, but we don't flush the gradient. It stays there. 

When we do the second forward pass and do backward again, all the backward operations do a `+=` on the gradient. So these gradients just add up and they never get reset to zero. 

To fix this, we need to iterate over all the parameters and make sure that `p.grad` is set to zero. We need to reset it to zero just like it is in the constructor. 

In PyTorch, this is done using the `zero_grad` function. 

After resetting the neural net and correcting the bug, we get a much more controlled descent. We still end up with pretty good results, and we can continue this a bit more to get down lower and lower.
# Understanding Neural Networks

The only reason that the previous thing worked, despite being extremely buggy, is that this is a very simple problem. It's very easy for this neural net to fit this data. As a result, the gradients ended up accumulating, effectively giving us a massive step size and making us converge. However, we now have to take more steps to reach very low values of loss and get the weight to be really good. We're going to get closer and closer to one minus one and one.

Working with neural networks can sometimes be tricky. You may have lots of bugs in the code, and your network might actually work, just like ours did. But chances are, if we had a more complex problem, this bug would have made us not optimize the loss very well. We were only able to get away with it because the problem is very simple.

Let's now bring everything together and summarize what we learned.

## What are Neural Networks?

Neural networks are mathematical expressions, fairly simple ones in the case of a multi-layer perceptron. They take input as the data and the weights and parameters of the neural net mathematical expression for the forward pass, followed by a loss function. The loss function tries to measure the accuracy of the predictions. Usually, the loss will be low when your predictions are matching your targets, or when the network is behaving well. We manipulate the loss function so that when the loss is low, the network is doing what you want it to do on your problem.

We then backward the loss, use backpropagation to get the gradient, and then we know how to tune all the parameters to decrease the loss locally. But then we have to iterate that process many times in what's called the gradient descent. We simply follow the gradient information, and that minimizes the loss. The loss is arranged so that when the loss is minimized, the network is doing what you want it to do.

We just have a blob of neural stuff, and we can make it do arbitrary things. That's what gives neural networks their power. This is a very tiny network with 41 parameters, but you can build significantly more complicated neural networks with billions, at this point almost trillions, of parameters. It's a massive blob of simulated neural tissue, roughly speaking, and you can make it do extremely complex problems. These neurons then have all kinds of very fascinating emergent properties.

For example, in the case of GPT, we have massive amounts of text from the internet, and we're trying to get a neural network to predict the next word in a sequence from a few words. That's the learning problem. It turns out that when you train this on all of the internet, the neural network actually has really remarkable emergent properties. But that neural network would have hundreds of billions of parameters. It works on fundamentally the exact same principles. The neural network will be a bit more complex, but otherwise, the value in the gradient is there and would be identical, and the gradient descent would be there and would be identical.
# Understanding Micrograd

The neural network setup and training are fundamentally identical and pervasive. By the end of this article, you will understand how it works under the hood. 

In the beginning, I mentioned that you would understand everything in Micrograd and then we would slowly build it up. Let's briefly prove that by stepping through all the code that is in Micrograd as of today. 

## Micrograd Code

Please note that some of the code may change by the time you read this article because I intend to continue developing Micrograd. 

Let's look at what we have so far. 

### Init.py

The `init.py` file is currently empty. 

### Engine.py

When you go to `engine.py`, it contains the value. Everything here should be mostly recognizable. We have the `data.grad` attributes, the `backward` function, the previous set of children, and the operation that produced this value. 

We have addition, multiplication, and raising to a scalar power. We also have the ReLU non-linearity, which is a slightly different type of non-linearity than the tanh that we used in this video. Both of them are non-linearities. 

Notably, tanh is not actually present in Micrograd as of right now, but I intend to add it later. The `backward` function is identical, and then all of these other operations are built up on top of operations here. 

There's no massive difference between ReLU, tanh, and sigmoid. They're all roughly equivalent and can be used in MLPs. I used tanh because it's a bit smoother and because it's a little bit more complicated than ReLU. Therefore, it stressed a little bit more the local gradients and working with those derivatives, which I thought would be useful.

### nn.py

The `nn.py` file is the neural networks library. You should recognize the identical implementation of neuron, layer, and MLP. 

Notably, we have a class module here. There is a parent class of all these modules. I did that because there's an `nn.module` class in PyTorch, and so this exactly matches that API. The `nn.module` in PyTorch also has a `zero_grad` function, which I've refactored. 

### Test.py

The `test.py` file creates two chunks of code, one in Micrograd and one in PyTorch. It makes sure that the forward and backward pass agree identically for a slightly less complicated expression and a slightly more complicated expression. Everything agrees, so we agree with PyTorch on all of these operations. 

### Demo.ipynb

Finally, there's a `demo.ipynb` file. It's a bit more complicated binary classification demo than the one I covered in this lecture. We only had a tiny dataset of four examples. Here, we have a bit more complicated example with lots of blue points and lots of red points. We're trying to again build a binary classifier to distinguish two-dimensional points as red or blue. 

It's a bit more complicated MLP here with a bigger MLP. The loss is a bit more complicated because it supports batches. So, because our dataset was so tiny, we didn't need to worry about batches. But in this demo, we do. 

That's the end of Micrograd. I hope this article has given you a better understanding of how it works under the hood.
# Micrograd: A Minimalistic Neural Network Library

In our previous discussions, we always performed a forward pass on the entire data set of four examples. However, when your data set consists of a million examples, what we usually do in practice is to pick out some random subset, which we call a batch. We then only process the batch forward, backward, and update. We don't have to forward the entire training set.

This approach supports batching because there's a lot more examples here. When we do a forward pass, the loss is slightly different. In this case, I implemented a max margin loss. The one that we used previously was the mean squared error loss because it's the simplest one. There's also the binary cross entropy loss. All of them can be used for binary classification and don't make too much of a difference in the simple examples that we looked at so far.

There's something called L2 regularization used here. This has to do with the generalization of the neural net and controls the overfitting in a machine learning setting. I did not cover these concepts in this video, but potentially will in later discussions.

The training loop should be familiar to you: forward, backward with zero grad, and update. You'll notice that in the update here, the learning rate is scaled as a function of the number of iterations and it shrinks. This is something called learning rate decay. In the beginning, you have a high learning rate and as the network sort of stabilizes near the end, you bring down the learning rate to get some of the fine details in the end.

In the end, we see the decision surface of the neural net and we see that it learns to separate out the red and the blue area based on the data points. That's the slightly more complicated example and then we'll demo that hyper parameter that you're free to go over.

As of today, that is Micrograd. I also wanted to show you a little bit of real stuff so that you get to see how this is actually implemented in a production-grade library like PyTorch. In particular, I wanted to show the backward pass for tanh in PyTorch. Here in Micrograd, we see that the backward pass for tanh is one minus t square times the grad, which is the chain rule. So we're looking for something that looks like this.

I went to PyTorch, which has an open-source GitHub codebase, and I looked through a lot of its code. Honestly, I spent about 15 minutes and I couldn't find tanh. That's because these libraries unfortunately grow in size and entropy. If you just search for tanh, you get apparently 2,800 results and 406 files. I don't know what these files are doing honestly, and why there are so many mentions of tanh. Unfortunately, these libraries are quite complex. They're meant to be used, not really inspected.

Eventually, I did stumble on someone who tries to change the tanh backward code for some reason. Someone here pointed to the CPU kernel and the CUDA kernel for tanh backward. So this basically depends on if you're using PyTorch on a CPU device or on a GPU. These are different devices and I haven't covered this. But this is the tanh backwards kernel for CPU.

The reason it's so large is that number one, this is like if you're using a complex type which we haven't even talked about. If you're using a specific data type of b-float 16 which we haven't talked about. And then if you're not, then this is the kernel. Deep here, we see something.
# Building Micrograd with PyTorch

In this lecture, we will be building out Micrograd, a very simple version of PyTorch. We will also be exploring how PyTorch works under the hood. 

This resembles our backward pass, so they have a times one minus b square. This 'b' here must be the output of the 10h and this is the health.grad. We found it deep inside PyTorch from this location for some reason inside binaryops kernel when 10h is not actually a binary op. We're not complex, we're here and here we go with one line of code. 

So we did find it, but unfortunately, these code pieces are very large and Micrograd is very, very simple. But if you actually want to use real stuff, finding the code for it can be difficult. 

I also wanted to show you a little example here where PyTorch is showing you how you can register a new type of function that you want to add to PyTorch as a Lego building block. So here, if you want to, for example, add a gender polynomial 3, here's how you could do it. 

You will register it as a class that subclasses storage.org that function, and then you have to tell PyTorch how to forward your new function and how to backward through it. So as long as you can do the forward pass of this little function piece that you want to add and as long as you know the local derivative, the local gradients which are implemented in the backward PyTorch will be able to back propagate through your function. 

Then you can use this as a Lego block in a larger Lego castle of all the different Lego blocks that PyTorch already has. And so that's the only thing you have to tell PyTorch and everything would just work and you can register new types of functions in this way following this example. 

And that is everything that I wanted to cover in this lecture. I hope you enjoyed building out Micrograd with me. I hope you find it interesting and insightful. 

I will post a lot of the links that are related to this video in the video description below. I will also probably post a link to a discussion forum or discussion group where you can ask questions related to this video and then I can answer or someone else can answer your questions. I may also do a follow-up video that answers some of the most common questions. 

But for now, that's it. I hope you enjoyed it. If you did, then please like and subscribe so that YouTube knows to feature this video to more people. 

Now here's the problem, we know dl by and that's everything I wanted to cover in this lecture. So I hope you enjoyed us building up Micrograd. 

Okay, now let's do the exact same thing for multiply because we can't do something like a times two. Oops.