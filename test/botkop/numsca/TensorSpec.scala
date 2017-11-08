package botkop.numsca

import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

import scala.language.postfixOps
import botkop.{numsca => ns}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex

class TensorSpec extends FlatSpec with Matchers with BeforeAndAfterEach {

  "A Tensor" should "slice by range" in {

    val t = Tensor(1, 2, 3, 4).reshape(4, 1)
    val u = t(:>)

    t.shape shouldEqual u.shape
    t.data shouldEqual u.data

    t(2 :> 4).data shouldEqual Tensor(3, 4).data

    t(2 :>).data shouldEqual Tensor(3, 4).data

    t(1 :> -1).data shouldEqual Tensor(2, 3).data

    t( :>(-1)).data shouldEqual Tensor(1, 2, 3).data
  }

  it should "slice by tensor" in {
    val t = ns.arange(0, 100).reshape(10, 10)
    val y = ns.arange(0, 10).reshape(10, 1)
    val r = t(y)

    println(r)

    r /= 3.14

    println(r)
    println(t)

  }

  it should "do sth simple" in {

    val t = ns.arange(0, 100).reshape(10, 10)
    val index = Seq(NDArrayIndex.point(0), NDArrayIndex.point(9))

    val x: INDArray = t.array.get(index:_*)
    println(x)

    x.assign(-1)

    println(t)

    val u = Tensor(0.13, -1.12).reshape(2, 1)
    println(u(0))

  }

}
