---
layout: post
date: 2020-02-21
title: "Three Types of Gradient Descent"
---
## What is Gradient Descent?

Gradient descent is an optimiaztion algorithm used to minimize a cost function. It involves tweaking parameters iteratively in order to arrive at the minimum of the cost function. It works by measuring the local gradient of an error function, and goes in the direction of the descending gradient step by step. Once the gradient is zero, you have reached a minimum. This is initialized by random initialization as your starting point. 

The steps in gradient descent are determined byt the learning rate hyperparameter. When the learning rate is small (small steps), the algorithm has to go through many iterations to converge on the minimum, which could take a long time. However, when the learning rate is large, you could be taking such large steps that you're actually going from different sides of the function and could potentially end up higher/further away from the minimum than when you started!

One of the problems you can encounter while running a Gradient Descent Algorithm is that they can get stuck in a local minumim, as not all cost functions are a nicely concave bowl shape. So depending where the random initialization begins, you could converge to a minimum with a slope of zero, but it could be the local minimum and not the global minimum of the function. Additionally, the function can have a plateau, where the slope plateaus at a place that is zero, but is still not the global minumum 

![gradient-descent](https://luicyfruit.github.io/img/gd1.png)

## Gradient Descent for Linear Regression

When using gradient descent for linear regression, the cost function (Mean Squared Error, or MSE) is a convex function. Due to this, any line segment between any two points on the graph of the function lies above or on the graph, meaning there is no local minimum -- thus gradient descent will not run into problems with the local minima. 

---
## Let's explore three types of Gradient Descent

![gradient-descent-3](https://luicyfruit.github.io/img/three-types.png)

## Batch Gradient Descent

When using batch gradient descent, you need to calculate the partial derivative at each model parameter, or how much the cost functiion changes if you change a parameter a little bit. You can compute the partial derivative over the full training set at each step (aka it uses the whole *batch* of the training data at every set), which results in the gradient vector. This means that batch gradient descent can be slow on very large training sets. The gradient vector will point uphill, so to go downhill towards the minimum, just go in the opposite direction. 

Determining the correct learning rate is key to Batch Gradient Descent. When the learning rate is too low, the algorithm eventually reaches the solution, however it may take a really long time. When the learning rate is too high, the algorithm can diverge, jump all ober the place, and sometimes get further from the solution at every step. So to find a good learning rate, use Gridsearch. 

The number of iterations is also a trade-off: if the number is too high it will waste a lot of time while parameters don't change, but if the number is too low, you risk not stopping at the minimum. A way to combat this is to set the number as high, but stop the model when the gradient vector becomes very small. 

## Stochastic Gradient Descent

Because gradient uses the entire training set, it can be very slow. Stochastic Gradient Descent instead picks a random instance in the training set at every iteration, and computes the gradient vector at that step. Due to it's random nature, the algorithm is less regular than batch, and will not actually converge on the minimum. Instead it will bounce around the minimum, very close to the minimum, but will never settle down. Thus, when the algorithm stops, the final parameters are close to the best, but not necessarily the best. In cases where the cost function contains a local minima, the random nature of SGD means it could potentailly jump out of a local minimu, where other algorithms could get stuck. 

Thus, you have a tradeoff of randomness: it is good to escape from local minima, however it means you never settle at the minimum. A potential solution to this tradeoff is to gradually decrease the learning rate as the algortim continues, starting off large and then having smaller and smaller steps. This allows for convergence at the end to be closer to the minimum, and is known as the learning schedule. 

## Mini-Batch Gradient Descent

Now that we've discussed the ups and downs of Batch and Stochastic Gradient Descent, it's time we found a compromise. Mini-batch Gradient Descent works by computing the gradeints on small random sets (not the full traning set like Batch, and not a random instance like Stochastic). This results in it being less erratic than SGD, and allows it to approach the minimum closer than SGD would as well. However, it may suffer from local minima. 

## Conclusion
As we've seen, there are pros and cons to all three types of Gradient Descent. Batch reaches the global minimum and will stop at it, however it can be slow and subject to getting stuck in local minima. Stochastic is much faster, and due to its random nature has the ability to jump out of local minima, however will never converge on the minimum itself, instead producing final parameters that are close to optimal, but not optimal. Mini-batch is faster and less eratic than both Batch and Stochastic respectively, however can also suffer from local minima, and approaches the minimum but doesn't converge fully. You can tweak these algorithms through your learning rate, learning schedule, and watching the gradient vectors as the iterations continue. 


