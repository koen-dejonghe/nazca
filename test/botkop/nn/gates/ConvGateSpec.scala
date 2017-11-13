package botkop.nn.gates

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestKit}
import botkop.{numsca => ns}
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, Matchers, WordSpecLike}

class ConvGateSpec
    extends TestKit(
      ActorSystem(
        "ConvGateSpec",
        ConfigFactory.parseString("""
                                |akka {
                                |  loglevel = "DEBUG"
                                |}
                              """.stripMargin)
      ))
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

    "calculate the backward pass" in {

      val aPrev = ns.randn(10,4,4,3)
      val w = ns.randn(2,2,3,8)
      val b = ns.randn(1,1,1,8)
      val pad = 2
      val stride = 2

      val z = ConvGate.convForward(aPrev, w, b, stride, pad)

      val (dA, dW, db) = ConvGate.convBackward(z, aPrev, w, b, stride, pad)
      println("dA_mean = " + ns.mean(dA).squeeze())
      println("dW_mean = " + ns.mean(dW).squeeze())
      println("db_mean = " + ns.mean(db).squeeze())


    }
  }

}
