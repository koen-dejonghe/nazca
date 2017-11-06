package botkop.numsca

import org.scalatest.{BeforeAndAfterEach, FlatSpec, Matchers}

import scala.language.postfixOps

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

}
