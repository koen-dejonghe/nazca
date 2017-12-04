
# NumSca: NumPy for Scala

NumSca is NumPy for Scala.

## Importing numsca
```scala
import botkop.{numsca => ns}
import ns.Tensor
import ns.Tensor._
```

## Creating a Tensor

```scala
scala> val ta: Tensor = ns.arange(10).reshape(2, 5)
val ta: Tensor = ns.arange(10).reshape(2, 5)
ta: botkop.numsca.Tensor =
[[0.00,  1.00,  2.00,  3.00,  4.00],
 [5.00,  6.00,  7.00,  8.00,  9.00]]
 
scala> ns.zeros(3, 3)
ns.zeros(3, 3)
res1: botkop.numsca.Tensor =
[[0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00],
 [0.00,  0.00,  0.00]]

scala> ns.ones(3, 2)
ns.ones(3, 2)
res5: botkop.numsca.Tensor =
[[1.00,  1.00],
 [1.00,  1.00],
 [1.00,  1.00]]

scala> Tensor(3,2,1,0)
Tensor(3,2,1,0)
res6: botkop.numsca.Tensor = [3.00,  2.00,  1.00,  0.00]

```



