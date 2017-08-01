---
layout: post
title: "A Paper A Day: Gradient Episodic Memory for Conintuum Learning"
date: 2017-07-31 0:17:00 -0600
categories:
- Data Science
- Tech
---

#### About APAD ####
_This summer I've been Interning at an AI lab at BCM, working under
Ankit Patel in his incredible Neuroscience-meets-Deep-Learning group. 
This field is moving faster than anything else out ther.
The only way I stay even close to updated
is through mailing lists and peers in research. A Paper A Day is a series
of blog posts that presents a palatable explanation of a paper. Hopefully,
I can make one post every few days and help others understand the concepts and 
mathematics of these papers._

<div style="text-align:center">
 <img src="http://www.explorestsimonsisland.com/images/widepics/tennis.jpg">
</div>

## Basics ##
* __Paper__: [Gradient Episodic Memory for Continuum Learning][GED]
* __Authors__: David Lopez-Paz, Marc'Aurelio Ranzato 
* __Organizaitons__: Facebook AI Research (FAIR)
* __Topic__: Coninuum Learning
* __In One Senetence__: _Paz and Ranzato define sorely needed metrics to the
subfield of Continual Learning while developing a gradient-inspired update rule 
that avoids catastrophic forgetting_

### Background ###

Paz and Ranzato's continuum learning targets a more general problem of 
_catastrophic forgetting_, which the authors describe as "the poor ability of models to
quickly solve new problems, without forgetting previously acquired knowledge."
Recently this has been a hot topic in AI recently, as a flurry of 
papers in early summer were released discussing this topic ([Elastic Weight Consolidation][EWC],
[PathNet][PathNet], [iCaRL][iCARL], [Sluice Network][Sluice Network], [Intelligent Synapses][Intelligent Synapses]).
Avoiding catastrophic forgetting and achieving nontrivial _backwards transfer_ (BT) and _forward transfer_ (FT) are major goals for
continual learning models, and in addition, general AI. 

__Analogy Alert!__ _As Ankit explained to me originally: 
If you know how to play tennis, your experience 
*should* aid your ability to pick up ping pong (FT). 
In addition, when you return to tennis, your aptitude
in tennis shouldn't decrease (some atheletes argue that they get better
at their primary sport because they've played secondary sports, i.e. BT)._ 
 
## Paper Accomplishments ##

1. Of the 5 papers mentioned above, 0 of them formally define metrics for a
continual learner. 
2. The gradient-aligning update rule is quite clever and pretty cool.

### The Metrics ###
First, let's take a look at their formal definitions for FT and BT. 
They're displayed below. The notation is a bit confusing, so I've done my
 best to parse it.
  
<div style="text-align:center">
    <img src="https://www.dropbox.com/s/qrj6sxkfruj42uk/Screen%20Shot%202017-07-31%20at%204.56.05%20PM.png?dl=1">
    <p style="font-size:13px"> T is the total number of tasks, enumerated from 1 to T. The bi vector is the random
    initialization for each task. I've omitted accuracy from this discussion because it seem too novel in the context
    of this paper.</p>
</div>

* Assume a fixed sequence of tasks (Numbered 1 through T)
* Forward transfer is the average accuracy of some task, task _i_, after each 
task in the sequence preceding _i_ is completed. 
    * Record the score for task _i_ upon random initialization
    * Learn task 1, record your score for task _i_. Learn task 2, record your
     score for task _i_, etc., up until task _i-1_.
    * Subtract from their score upon initialization and average these scores.
* Backwards Transfer is the average accuracy change for task _i_ after each 
task afterwards has been completed
    * Record the score for task _i_ after learning it
    * Learn task _i+1_. Now record the score for task _i_. Learn task _i+2_. Record the
    score for task _i_, etc.
    * For each score of task _i_ that recorded after completeing _i_, subtract from the first
    score for task _i_. Average these Scores

    
#### Gripes about the Metrics ####
In my opinion, these metrics don't generalize well. Continuum Learning (which I presume
is less general than _Continual_ Learning) specifies a sequence of tasks,
 meaning it is sensitive to the order of tasks. In their experiments section, 
 they use tasks that theoretically don't depend on the order in which they're learned, so in the 
 scheme of their paper this point is moot. However, Continual Learning in general
has no specification on order. Other papers concerning this topic have not discussed
task curriculums at all, while this paper glosses over it. 

__A metric I prefer: randomly sample from a pool of tasks
_n_ times. Learn these _n_ tasks in an arbitrary order. Lastly, evaluate accuracy on _i_ (for forward transfer).__ 
(This can be done over multiple trials to get a robust average)

### Gradient-Aligned Updates ###

For those new to Machine Learning, much of Deep Learning is powered by the 
[__backpropagation algorithm__][backprop]. This algorithm calculates an update that
will improve the accuracy for the problem based on an error metric. It does this by calculating what's 
called a gradient. 

__Analogy Alert!__ : _You shoot a dart at a board. You shot low by some distance __d__. 
You correct your mechanics, backwards reasoning from the missed distance, to your 
release point, and from there perhaps your throwing velocity. 
You can think of these corrections as the gradient, and the linked modification of
all the preceding components as an implementation
of the [chain rule][chainrule]. Disclaimer: 
 There is no evidence that the brain implements backpropagation_

So what does Gradient Episodic Memory (GEM) do exactly? Let's start with the Memory
part and go from there

* _Memory_: Recall those sequence of tasks we mentioned earlier? Well for each of those
tasks, let's make sure we don't forget them. We'll keep a portion of them in memory. 

* _Episodic_: Let's replay these memories to make sure we're not damaging our accuracy
on these tasks when we learn new ones. By playing them over again, we're basically going through an episode

* _Gradient_: When we look at the episode again, let's make sure the gradient doesn't
go the wrong way. What this means is: let's not unlearn what we learned on the 
previous task. 

### So How is it Done? ###

<div style="text-align:center">
    <img src="https://www.dropbox.com/s/jye3b3mco5fs277/Screen%20Shot%202017-07-31%20at%204.55.52%20PM.png?dl=1">
    <p style="font-size:13px"> g is the gradient for the current task, while gk is the gradient for each previous task, calculated
    over the episodes in memory (Mk). The big < > notation is a dot product operator </p>
</div>

Dot Products! In order for a gradient update to take place, they compute the dot product
of the current learning task with all the previous tasks in memory. The update is allowed
to take place if the gradient is greater than or equal to 0 for all the episodes. This translates
into constraining your update for one task to not conflict with an update for the previous task.

What if the gradient is going the wrong way? Paz and Ronzato take this gradient update
and project it to the closest possible vector that doesn't go the wrong way 
(The proof is in the pudding, eqn. 8 - 13 in the paper. It formulates the optimization as a projection
on to a cone).
 
<div style="text-align:center">
    <img src="https://www.dropbox.com/s/jkdkk8bmz6btl77/Screen%20Shot%202017-07-31%20at%207.36.14%20PM.png?dl=1">
    <p style="font-size:13px"> Above shows the graphical representation of the gradient update conditions, with the blue
    line being the update for the first task, while the red line is the gradient update for the current task. The right side
    shows an approximation of the optimized projection for the gradient when the dot product is negative.</p>
</div>


### Does it work? ###
Yes. Well, kind of. 

The focus of this paper was to minimize Backwards Loss (aka
maximize Backwards Transfer). In this sense, they appear to succeed. However, 
the small improvements lack error bars, making an unconvincing case (#DLNeedsErrorBars). 
Forward Transfer is negligible on all but one experiment (there were three total).

<div style="text-align:center">
<img src="https://www.dropbox.com/s/qvr95xydhtlnijw/Screen%20Shot%202017-07-31%20at%204.55.41%20PM.png?dl=1">
</div>

The plots above show performance. The right hand side demonstrates
the accuracy on the first task as the consequent tasks are learned, which each different
colored bar indicating a start to learning for a new task.

#### Knitpicking ####

* The experiments compare against Elastic Weight Consolidation (EWC). However, 
EWC was tested and optimized for Reinforcement Learning and Atari Games. I wonder
if an earnest job of optimizing EWC for the tasks at hand was done. 

* There is still no metric for parameter conservation as a result of continual/shared learning. 
A curve showing the change in accuracy across a set of tasks while increasing 
the size of the overall network would be nice. It would be interesting to compare 
all the papers on this metric. You could also evaluate the similarity of tasks
(or how well a network learns similarities in tasks) through this method.

### Summary ###
Cool Method.  Nice Paper. Less than satisfying results. But in general a
solid step forward for continual learning/overcoming catastrophic
forgetting. 


[EWC]: https://arxiv.org/pdf/1612.00796.pdf
[PathNet]: https://arxiv.org/pdf/1701.08734.pdf
[iCARL]: https://arxiv.org/pdf/1611.07725.pdf
[Sluice Network]: https://arxiv.org/pdf/1705.08142.pdf
[Intelligent Synapses]: https://openreview.net/pdf?id=rJzabxSFg
[Metrics]: https://www.dropbox.com/s/qrj6sxkfruj42uk/Screen%20Shot%202017-07-31%20at%204.56.05%20PM.png?dl=0
[Gradient]: https://www.dropbox.com/s/jye3b3mco5fs277/Screen%20Shot%202017-07-31%20at%204.55.52%20PM.png?dl=0
[Results]: https://www.dropbox.com/s/qvr95xydhtlnijw/Screen%20Shot%202017-07-31%20at%204.55.41%20PM.png?dl=0
[GED]: https://arxiv.org/pdf/1706.08840.pdf
[backprop]:https://en.wikipedia.org/wiki/Backpropagation
[chainrule]:https://en.wikipedia.org/wiki/Chain_rule

 
  

