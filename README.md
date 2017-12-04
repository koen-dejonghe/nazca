
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
import ns._
import ns.Tensor._
```

## Creating a Tensor

```scala
scala> Tensor(3,2,1,0)
[3.00,  2.00,  1.00,  0.00]

scala> ns.zeros(3, 3)
[[0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00]]

scala> ns.ones(3, 2)
[[1.00,  1.00],
 [1.00,  1.00],
 [1.00,  1.00]]
 
scala> val ta: Tensor = ns.arange(10)
[0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val tb: Tensor = ns.reshape(ns.arange(9), 3, 3)
[[0.00,  1.00,  2.00],
 [3.00,  4.00,  5.00],
 [6.00,  7.00,  8.00]]
 
 scala> val tc: Tensor = ns.reshape(ns.arange(2 * 3 * 4), 2, 3, 4)
 [[[0.00,  1.00,  2.00,  3.00],
   [4.00,  5.00,  6.00,  7.00],
   [8.00,  9.00,  10.00,  11.00]],
 
  [[12.00,  13.00,  14.00,  15.00],
   [16.00,  17.00,  18.00,  19.00],
   [20.00,  21.00,  22.00,  23.00]]]
```

## Access
Single element
```scala
scala> ta(0)
res10: botkop.numsca.Tensor = 0.00

scala> tc(0, 1, 2)
res14: botkop.numsca.Tensor = 6.00
```
Get the value of a single element Tensor:
```scala
scala> ta(0).squeeze()
res11: Double = 0.0
```
Slice
```scala
scala> tc(0)
res7: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00,  3.00],
 [4.00,  5.00,  6.00,  7.00],
 [8.00,  9.00,  10.00,  11.00]]
 
scala> tc(0, 1)
res8: botkop.numsca.Tensor = [4.00,  5.00,  6.00,  7.00]
```

## Update
In place
```scala
scala> val t = ta.copy()
t: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> t(3) := -5
scala> t
res16: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  -5.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> t(0) += 7
scala> t
res18: botkop.numsca.Tensor = [7.00,  1.00,  2.00,  -5.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]
```

Array wise
```scala
scala> val a2 = 2 * ta
val a2 = 2 * ta
a2: botkop.numsca.Tensor = [0.00,  2.00,  4.00,  6.00,  8.00,  10.00,  12.00,  14.00,  16.00,  18.00]
```

## Slicing
Single dimension

```scala
scala> val a0 = ta.copy().reshape(10, 1)
val a0 = ta.copy().reshape(10, 1)
a0: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val a1 = a0(1 :>)
val a1 = a0(1 :>)
a1: botkop.numsca.Tensor = [1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00,  9.00]

scala> val a2 = a0(0 :> -1)
a2: botkop.numsca.Tensor = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  6.00,  7.00,  8.00]

scala> val a3 = a1 - a2
a3: botkop.numsca.Tensor = [1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00,  1.00]

scala> ta(:>, 5 :>)
res19: botkop.numsca.Tensor = [5.00,  6.00,  7.00,  8.00,  9.00]

scala> ta(:>, -3 :>)
res4: botkop.numsca.Tensor = [7.00,  8.00,  9.00]
```
