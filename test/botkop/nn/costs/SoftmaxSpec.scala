package botkop.nn.costs

import botkop.nn.TestUtil
import botkop.numsca._
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}

class SoftmaxSpec extends FlatSpec with Matchers {


  "Softmax" should "be calculated correctly" in {

    Nd4j.setDataType(DataBuffer.Type.DOUBLE)

    rand.setSeed(231)

    val numClasses = 10
    val numInputs = 50

    val x = randn(numInputs, numClasses) * 0.001
    val y = randint(numClasses, Array(1, numInputs))

    val (loss, dx) = Softmax.costFunction(x, y)

    println(loss)
    loss should equal(2.3 +- 0.2)

    def fdx(a: Tensor) = Softmax.costFunction(a, y)._1
    val dxNum: Tensor = TestUtil.evalNumericalGradient(fdx, x)

    val dxError = TestUtil.relError(dx, dxNum)
    println(dxError)
    dxError should be < 1e-7

  }

}
