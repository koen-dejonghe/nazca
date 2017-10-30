package botkop.nn.templates

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestActors, TestKit}
import botkop.nn.gates.{NetworkSpec, SigmoidGate}
import com.typesafe.config.ConfigFactory
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}
import play.api.libs.json._

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

      val optimizer = AdamOptimizerTemplate(0.1, 0.2, 0.3, 0.4, 0.5)
      val l1 = LinearGateTemplate(List(3072, 100), 0.6, optimizer, 231)
      val l2 = ReluGateTemplate
      val l3 = OutputGateTemplate(Softmax)
      val n = NetworkTemplate(List(l1, l2, l3))

      val json = Json.prettyPrint(Json.toJson(n))
      println(json)

      Json.fromJson[NetworkTemplate](Json.parse(json)) match {
        case e: JsError =>
          println(e.errors)
          throw new Exception("error")
        case s: JsSuccess[NetworkTemplate] =>
          val nn: NetworkTemplate = s.get
          assert (nn == n)
      }

    }

    "construct from stubs" in {
      val nn: NetworkTemplate = (LinearGateTemplate + ReluGateTemplate) * 2 + OutputGateTemplate(Softmax)
    }
  }

}
