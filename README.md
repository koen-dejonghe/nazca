
# NumSca: NumPy for Scala

NumSca is NumPy for Scala.
For example, here's the famous [neural network in 11 lines of python code](http://iamtrask.github.io/2015/07/12/basic-python-network/), this time in scala:

```scala
val x = ns.array( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1).reshape(4, 3)
val y = ns.array(0, 1, 1, 0).T
val w0 = 2 * ns.rand(3, 4) - 1
val w1 = 2 * ns.rand(4, 1) - 1
for (j <- 0 until 60000) {
  val l1 = 1 / (1 + ns.exp(-ns.dot(x, w0)))
  val l2 = 1 / (1 + ns.exp(-ns.dot(l1, w1)))
  val l2_error = ns.mean(ns.abs(y - l2)).squeeze()
  val l2_delta = (y - l2) * (l2 * (1 - l2))
  val l1_delta = l2_delta.dot(w1.T) * (l1 * (1 - l1))
  w1 += l1.T.dot(l2_delta)
  w0 += x.T.dot(l1_delta)
}
``` 

## Importing numsca
```scala
import botkop.{numsca => ns}
import ns.Tensor
import ns.Tensor._
```

## Creating a Tensor

```scala
scala> Tensor(3,2,1,0)
[3.00,  2.00,  1.00,  0.00]

scala> val ta: Tensor = ns.arange(10).reshape(2, 5)
[[0.00,  1.00,  2.00,  3.00,  4.00],
 [5.00,  6.00,  7.00,  8.00,  9.00]]
 
scala> ns.zeros(3, 3)
[[0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00]]

scala> ns.ones(3, 2)
[[1.00,  1.00],
 [1.00,  1.00],
 [1.00,  1.00]]


```



