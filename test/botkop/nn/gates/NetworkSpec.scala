package botkop.nn.gates

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestKit}
import botkop.nn.costs.Softmax
import botkop.nn.optimizers.Nesterov
import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}
import play.api.libs.json.Json

class NetworkSpec
    extends TestKit(
      ActorSystem("NetworkSpec", ConfigFactory.parseString(NetworkSpec.config)))
    with ImplicitSender
    with WordSpecLike
    with Matchers
    with BeforeAndAfterAll
    with LazyLogging {

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A network builder" must {
    "write and read json" in {

      val t1 = ((Linear + BatchNorm + Relu + Dropout) * 2 + Linear)
        .withDimensions(32 * 32 * 3, 100, 50, 10)
        .withOptimizer(Nesterov)
        .withCostFunction(Softmax)
        .withRegularization(1e-3)
        .withLearningRate(0.8)
        .networkConfig

      val json = Json.prettyPrint(Json.toJson(t1))
      logger.info(json)

      val t2 = Json.fromJson[NetworkConfig](Json.parse(json)).get

      assert(t2 == t1)
    }

  }
}

object NetworkSpec {
  val config: String =
    """
      |akka {
      |  loglevel = "WARNING"
      |}
    """.stripMargin
}
