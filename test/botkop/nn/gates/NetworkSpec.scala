package botkop.nn.gates

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestActors, TestKit}
import com.typesafe.config.ConfigFactory
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}
import play.api.libs.json.Json

class NetworkSpec
    extends TestKit(
      ActorSystem("NetworkSpec", ConfigFactory.parseString(NetworkSpec.config)))
    with ImplicitSender
    with WordSpecLike
    with Matchers
    with BeforeAndAfterAll {

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A network" must {
    "write json" in {

      implicit val projectName: String = "test"

      val template: Network = ((Linear + BatchNorm + Relu) * 2)
        .withDimensions(32 * 32 * 3, 50, 10)

      println(Json.prettyPrint(Json.toJson(template)))

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
