# Mobilenet v2

Authors:

Publication time:

Tags



## Main objective





## Main contribution



## Main incentive





## Implemental Details

* About residual addition: according to table 2, the same inverted residual block repeats several times. Thus in the first repetition, the input channel and output channel number may not match. A 1*1 conv to adapt the channel number is used.
* In the rest repetition, the stride is changed to 1



#### Question

* where dropout is used?
* ​