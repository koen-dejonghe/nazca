package botkop.nn.templates

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestActors, TestKit}
import botkop.nn.gates.NetworkSpec
import com.typesafe.config.ConfigFactory
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}
import play.api.libs.json.Json

class NetworkTemplateSpec
    extends TestKit(ActorSystem("NetworkSpec", ConfigFactory.parseString("")))
    with ImplicitSender
    with WordSpecLike
    with Matchers
    with BeforeAndAfterAll {

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A network template" must {
    "write json" in {

      val optimizerParameters = AdamOptimizerParameters(0.1, 0.2, 0.3, 0.4, 0.5)
      val optimizer = AdamOptimizerTemplate(optimizerParameters)
      val l1parameters = LinearGateParameters(Array(3072, 100), 0.6, optimizer, 231)
      val l1 = LinearGateTemplate(l1parameters)
      val n = NetworkTemplate(List(l1))

      println(Json.prettyPrint(Json.toJson(n)))


    }
  }

}
