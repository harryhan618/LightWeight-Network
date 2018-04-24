# Mobilenet v2

Authors: Mark Sandler, Andrew Howard, et.al.,  Google

Date: April 2018

Tags: light-weight architecture



## Main objective

The paper proposes a new mobile architecture.



## Main contribution

* Inverted Residual Linear Bottleneck: 
  * Inverted channel elevation
  * Depthwise Separable Convolution
  * Linear Bottleneck





## Main incentive

* ####Inverted Residual Linear Bottleneck: 

  * In order to preserve more information, 1*1 conv is used to elevate the channel numbers.
  * Then DW Conv is used to reduce calculation.
  * Using linear in the end of block rather than relu, to reduce information loss due to relu activation





## Implemental Details

* #### About residual addition

  according to table 2, the same inverted residual block repeats several times. In the repetition, except the first one,  the stride is changed to 1. Then the residual addition is performed.

* #### hyper parameters trade-off

  use width multiplier as Mobilenet v1 did, change every layer's channel

  the network scales 32 times down, so we can choose the input size of multiples of 32





#### Question

* where dropout is used?
* ​