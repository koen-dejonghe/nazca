package botkop.nn.gates

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestActors, TestKit, TestProbe}
import botkop.nn.optimizers.{AdamOptimizer, Optimizer}
import botkop.numsca.Tensor
import org.scalatest._
import botkop.{numsca => ns}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

import scala.concurrent.duration._
import scala.language.postfixOps

class ConvGateSpec
    extends TestKit(ActorSystem("ConvGateSpec", ConfigFactory.parseString(
      """
        |akka {
        |  loglevel = "DEBUG"
        |}
      """.stripMargin)))
    with ImplicitSender
    with WordSpecLike
    with Matchers
    with BeforeAndAfterEach
    with BeforeAndAfterAll
    with LazyLogging {

  override def beforeEach(): Unit = {
    // must set to double type for gradient checks !!!!
    DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)
  }

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A ConvGate" must {

    "calculate the forward pass" in {

      val x_shape = Array(2, 3, 4, 4)
      val w_shape = Array(3, 3, 4, 4)
      val x = ns.linspace(-0.1, 0.5, num = x_shape.product).reshape(x_shape)
      val w = ns.linspace(-0.2, 0.3, num = w_shape.product).reshape(w_shape)
      val b = ns.linspace(-0.1, 0.2, num = 3).reshape(3, 1)

      val stride = 2
      val pad = 1

      val out = ConvGate.convForward(x, w, b, stride, pad)

      println(out)

      val correct_out = Tensor(-0.08759809, -0.10987781, -0.18387192,
        -0.2109216, 0.21027089, 0.21661097, 0.22847626, 0.23004637, 0.50813986,
        0.54309974, 0.64082444, 0.67101435, -0.98053589, -1.03143541,
        -1.19128892, -1.24695841, 0.69108355, 0.66880383, 0.59480972,
        0.56776003, 2.36270298, 2.36904306, 2.38090835,
        2.38247847).reshape(2, 3, 2, 2)

      val diff = relError(out, correct_out)
      println(s"diff = $diff")

      assert(diff < 2e-7)

    }

    "calculate the backward pass" in {

      ns.rand.setSeed(231)

      val x = ns.randn(4, 3, 5, 5)
      val w = ns.randn(2, 3, 3, 3)
      val b = ns.randn(2, 1)
      val dout = ns.randn(4, 2, 5, 5)

      val stride = 1
      val pad = 1

      val out = ConvGate.convForward(x, w, b, stride, pad)
      val (dx, dw, db) = ConvGate.convBackward(dout, x, w, b, stride, pad)

      def fdx(a: Tensor): Tensor = ConvGate.convForward(a, w, b, stride, pad)

      def fdw(a: Tensor): Tensor = ConvGate.convForward(x, a, b, stride, pad)

      def fdb(a: Tensor): Tensor = ConvGate.convForward(x, w, a, stride, pad)

      val dxNum = evalNumericalGradientArray(fdx, x, dout)
      val dxError = relError(dx, dxNum)
      println(dxNum)
      println(dx)
      println(dxError)
      dxError should be < 1e-8

      val dwNum = evalNumericalGradientArray(fdw, w, dout)
      val dwError = relError(dw, dwNum)
      println(dwError)
      dwError should be < 1e-8

      val dbNum = evalNumericalGradientArray(fdb, b, dout)
      val dbError = relError(db, dbNum)
      println(dbError)
      dbError should be < 1e-8
    }

    "forward the activation" in {

      val echo = system.actorOf(TestActors.echoActorProps)

      val probe = TestProbe()

      val cc = ConvConfig(numChannels = 3,
                          height = 32,
                          width = 32,
                          numFilters = 32,
                          filterSize = 7,
                          weightScale = 1e-3,
                          regularization = 0.0,
                          optimizer = AdamOptimizer(1e-3),
                          stride = 1,
                          pad = 1,
                          seed = 231)

      val gate = system.actorOf(ConvGate.props(probe.ref, cc))

      // val x = ns.randn(4, 32 * 32 * 3)
      val x = ns.randn(4, 3, 32, 32)
      val y = ns.randn(4, 1)

      gate ! Forward(x, y)

      val a = probe.expectMsgType[Forward](10 seconds)
      //println(a)

    }
  }

  /**
    * returns relative error
    */
  def relError(x: Tensor, y: Tensor): Double = {
    ns.max(ns.abs(x - y) / ns.maximum(ns.abs(x) + ns.abs(y), 1e-8)).squeeze()
  }

  /**
    * Evaluate a numeric gradient for a function that accepts an array and returns an array.
    */
  def evalNumericalGradientArray(f: (Tensor) => Tensor,
                                 x: Tensor,
                                 df: Tensor,
                                 h: Double = 1e-5): Tensor = {
    val grad = ns.zeros(x.shape)
    val it = ns.nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x.squeeze(ix)

      x(ix) := oldVal + h

      val pos = f(x)

      x(ix) := oldVal - h
      val neg = f(x)

      x(ix) := oldVal

      val g = ns.sum((pos - neg) * df) / (2.0 * h)
      grad(ix) := g
    }
    grad
  }

  /**
  a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    */
  def evalNumericalGradient(f: (Tensor) => Double,
                            x: Tensor,
                            h: Double = 0.00001): Tensor = {

    val grad = ns.zeros(x.shape)
    val it = ns.nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x.squeeze(ix)

      x(ix) := oldVal + h
      val pos = f(x)

      x(ix) := oldVal - h
      val neg = f(x)

      x(ix) := oldVal

      val g = (pos - neg) / (2.0 * h)
      grad(ix) := g
    }

    grad
  }

}
