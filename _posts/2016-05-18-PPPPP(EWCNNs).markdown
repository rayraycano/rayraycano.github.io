---
layout: post
title:  "PPPPP(EWCNNs)"
date:   2016-05-18 1:50:51 -0600
categories: data science
---
There's a common saying called the 5 Ps: 

Proper Preparation Prevents Poor Performance

I've heard it most with sports and academic testing. Given that those two aren't typically neighbors in who or what we associate with them, I find the phrase quite robust and applicable to most of life's challenges. In this project, I found out the phrase is true *Especailly With Convolutional Neural Networks*.

Anyways, for our term project in our Machine Learning class at Rice, we were tasked with taking on the CIFAR10 dataset and training a model to predict on the CIFAR10 test set. We were given the whole term to figure things and make progress, however, we didn't get to the juicy convolutional neural nets until last month and a half in class, and even then, we brushed over them in a brevity that left everything needed for the term project to be learned outside of the class. 

This is where our journey began. I was partners with Xilin Liu, a very accomplished senior with the most rigorously organized code I've every seen or contributed to (shouts out to the homie X). After stepping through some [Softmax][Softmax] and [SVM models][SVM], and playing with [Histogram of Gradients][HOG] pre processing, we were at 55% accuracy model, which at the time, was decent enough for top 25% in the class standings. 

We then waited a bit too long to get started on the CNN attempt for our image classification. The basic idea of a convolutional neural net is to look at a sliding window of the image, much like a Histogram of Gradients does, except that it will end up learning a different filter for each window, basically meaning that it can learn the importance of the window, and also how to transform it to extract the necessary features there. Moving our image through these convolutional layers as well as max pooling layers, which take a sliding window and simply sets all pixels in the window to the max, and normalizing layers, which basically recenters the convoluted mean and variance of our image, helps get us to an image that we can more accurately classify, with performace reaching *higher* than [human capability][better-than-human]. 

However, here goes the synopsis of my journey with Cifar-10

1. We got wind that TensorFlow, a Google ML Library had a tutorial that achieved 86% accuracy on test data.
2. We heard that [GPUs greatly speed up ML code][GPUs] if the code has been optimized
3. We realize that neither of our laptops are gonna get the job done
4. I start focusing on using Amazon EC2 for our development, as they have GPU and multicore CPU instance Linux images
5. Struggle mightily with instillations
6. Give up on using the most updated instillations
7. Run TensorFlow, which then prints for us the accuracy it got using it's test data
8. Try and figure out wtf TensorFlow does and how it works
9. Cry.gif
10. Finally understand how TensorFlow works, get it to run properly.
11. Start experimenting running tests all day and all night and all day and all night and all day and all night and all day and all night
12. Writereportmakeposterrunfinalexperiments (this is what this part felt like. It was like a week of ONLY comp540. Eat breath sleep 540.)

This whole process was just 2 weeks of constantly plugging away at code and feeling so incompetent, then triumphant, then incompetent, then triumphant, then moderately satisfied at the end. The focus of the rest of the blog will b on steps 5, 10, and 11. There will be some great links for anyone else who wants to try this, so keep an eye out ^.^

Tensorflow, Amazon Images, and Instillation Nightmares
------------------------------------------------------

[Tensorflow][Tensorflow] is a robust machine learning library in Python (my go to language) that was recently bought by google. Tensorflow is relatively new, which meant I was on the *bleeding edge* of technology!!!11!1 However, this meant that things that were supposed to be reliable were actually buggy: 

+ The source URL at one point had a typo in it. 
+ The program seg faulted when I attempted to imported sklearn along with tensorflow.
+ There were functions that in the library that didn't appear in the docs (wtf??).

A couple of versions of tensorflow were out there to use, the newest being 0.8, which the Cifar10 tutorial was written in. So naturally I wanted to use that one. Once I got the hang of using an EC2 image, which was moderately straightforward, I jumped into pip installing TF .8 onto a micro T2 instance, which is basically just a low power computer that Amazon gives out 750 free hours for when you sign up. It seemed to run to just fine. Woo! We're rolling. 

Now, I wanted to run the GPU optimized code. Uh oh. Turned out, the Amazon GPU instances came with CUDA 6.5 preloaded. TF needed CUDA 7.0. So on I went attempting to update the CUDA and cuDNN drivers. After a day of failing to successfully install, I decided to surf the interwebs for other options (Note, I was able to successfully instill at a later time, however, that was a bit later in the game and it took a bit of time to do all the instillations). I started looking through message boards that had the same issues I was having and found [this][mess-board] message board with everything needed for TF 0.7, running with Python3. Praise the lord for the person who made this [public ami][public-ami]. 

This made things easy, though I did lose a bit of functionality by using 0.7 instead of 0.8 (most notably the summary writer and a new batch normalization routine). Regardless I was finally able to run the tensorflow exmample. 

WTF is a Tensor: Understanding execution in TensorFlow
------------------------------------------------------

Understanding how tensorflow worked was quite the adventure as well. To start off, go to debugging methods failed me here. Print statements on the variables just didn't work. All I would get would be the size of the matrix I was printing and some message telling me it was a 'Tensor'. As my buddy Jonathan Jao phrased it (and I'm paraphrasing), TF seemed like a restructuring of dataflow and data processing. After reading some of the docs (which I should have done first - PPPPP), I realized everything ran like a [Data Driven Future][DDF], an elegant method of parallelization. Once you called a session on a variable, all the code needed for that variable would be ran. Within the session you would then get the variable of the type you expected, rather than a tensor. 

Tensorflow also used a different way of saving models, opting to put all of the parameters into a checkpoint file stored in a designated folder (in the tutorial it's ~/tmp/checkpoints or something of that sort), and then loading the checkpoint parameters into a model when loading for testing or resuming training. This differed from the the traditional [pickling][pickle] method I had used before. This initially threw me for a loop as a I scoure the code wondering where in the world the model was going to be saved once the training script had finished and I needed to run the test.

The test data in TF was loaded from the widely used [toronto site][toronto]. However, our professor had modified the test data to include random noise as well as 290,000 garbage photos and posted it as a .zip to download from the Kaggle comptetion site. The pictures were also in .jpg form, rather than binary format as given by the standard download data. After getting the dimensions right for flattening and appending an order label to the front of each test image (for sorting purposes - parallelizing the computations would lead to unsorted results, and we needed results in order for competitions submission), we finally were able to submit our results from the tutorial to our Kaggle competition. 86% in the bag. 

Going Backwards: Understanding after Doing
------------------------------------------

The following section details the rushed, uninformed, terrible approach I took to try and improve our model accuracy, along with the lessons I learned from this. 

So to harp back on the theme of Proper Preparation Prevents Poor Performance, I think this experiment was the confirmation of that, except in the contrapositive. Poor Preparation Precedes Poor Performance. We jumped into the tutorial without even understanding the basics of it, or knowing that it was derived from AlexNet. We didn't stop to understand the regularization methods, the gradient descent methods used, or any of the particulars with variable initialization.

Immediately after getting the tutorial to run, I jumped right into [leaky ReLUs][leaky-ReLUS], replacing the ReLUs that followed convolutional units with leaky ones. We had also lowered the batch size from 128 to 64, simply because that meant our code would finish training in half the time. Using alphas of .05, .1, and .3, we were able to attain accuracy of 86.4%, 86.8%, and 86.8% respectively. 

I was ecstatic! I had read on the [Kaggle Competition Interviews][Kaggle-Competition]  that the winning solution used tons of leaky ReLUs, and voila! We had already improved on our model. We were gonna hit 90% no problem, I was sure of it. 

Eager to see the results of the ReLU modification, I hadn't even stopped to write the code for validation. The tutorial didn't come with validation baked in, so after running the ReLU tests, I realized that to check the following results, the proper method would be to use validation sets. 

Next I attempted implementing the [ResNet][ResNet] model. The ResNet model showed amazing results due to it's residual inputs within each residual layer. This prevented gradient decay and had led to incredibly accurate results. I threw in the architecture implementation found [here][ResNet-Implementation], but figured I wouldn't need to change any of the other variables, such as batch size, learning rate particulars, learning rate decay or the image pre-processing methods (we used a per-image whitening used in the tutorial, while [He et al][He] specified subtracting by mean and dividing by variance of entire set of images.

Huge mistake. These parameters are vital to the convergence of the model and failure to imitate them as exactly specified by the paper resulted in a final accuracy of 79%, with 82% on validation. This was dishearteningly lower than the 92-93% promised by the paper. 

Our next attempt was with the [VGG net][VGG]. This time I attempted to mimic it more closely, however, again, the failure to exactly mimic the paper specifications prevented our model from even fitting the training data accurately, giving us a training loss that was 8 times higher than our tutorial after 50k steps. 

Admittedly, I was rushing through these experiments. Haphazardly I'd put together architectures and run them willy-nilly, careless about the vital particulars that largely affect the end results. The competition aspect was getting to me. Typically, academic assignments aren't competitions, but this felt like a dogfight. Our good buddies Krishna Thiagarajan and Ethan Perez were neck and neck with us, and Ethan was endlessly testing and iterating on his code just as I was. These guys were going all out --- rather than meddle with the troublesome Amazon EC2 instance, they used the opportunity as a reason/excuse/motive to *build their own computer* with [extra dank GPUs][extra-dank]. 

As I ran through these tests, compelled by a desire to win, my feelings of unease, bewilderment, and ignorance grew. Were these runs going to succeed? Were they not? Why? I had no intution towards any of things. I had not thoroughly read through the papers to understand what made the architecture and specific parameters so important. How was I going to even make improvements. There was no intution.

It was at this point I realized I had done everything wrong. This wasn't the proper way to approach this project. This was lazy. No preparation was put in. Poor performance was put out. There's a strong sense of disappointment in myself. I had done the job halfway and not stopped to clearly understand the task in front of me. This became palpable when writing the report. There were so many number to explain, graphs we just didn't care to have, and intutions we couldn't explain because our preparation severly lacked. 

A Note for Next Time
--------------------

1. Read the docs
2. Understand Why
3. Don't ignore what you don't understand. Google is an Alt+Tab away. 

I know this was a long read, but I got a lot out of this project. There were a lot of things I learned, but even more things I failed to learn. With my next project, I'll be taking caution to build off this experience and be better. 

Proper Preparation Prevents Poor Performance *Especially* with CNNs.

[Softmax]: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
[SVM]: https://en.wikipedia.org/wiki/Support_vector_machine
[HOG]: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
[GPUs]:http://www.nvidia.com/content/events/geoInt2015/LBrown_DL.pdf
[better-than-human]: http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
[Tensorflow]: https://www.tensorflow.org/
[mess-board]: http://ramhiser.com/2016/01/05/installing-tensorflow-on-an-aws-ec2-instance-with-gpu-support/
[public-ami]: https://gist.github.com/AlexJoz/1670baf0b32573ca7923 
[DDF]: https://wiki.rice.edu/confluence/download/attachments/4435861/comp322-s16-lec14-slides-v1.key.pdf?version=1&modificationDate=1455562128535&api=v2
[pickle]: https://docs.python.org/2/library/pickle.html
[extr-dank]:http://www.newegg.com/Product/Product.aspx?Item=N82E16814487157
[toronto]: https://www.cs.toronto.edu/~kriz/cifar.html
[leaky-ReLUS]: http://cs231n.github.io/neural-networks-1/
[Kaggle-Competition]: http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/
[ResNet]: http://arxiv.org/abs/1512.03385
[ResNet-Implementation]: https://github.com/xuyuwei/resnet-tf
[He]: http://arxiv.org/abs/1512.03385
[VGG]: http://www.robots.ox.ac.uk/~vgg/research/very_deep/