package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca.Tensor
import botkop.{numsca => ns}

class ConvGate(next: ActorRef, config: ConvConfig)
    extends Actor
    with ActorLogging {

  import config._

  val name: String = self.path.name
  log.debug(s"my name is $name")

  var w: Tensor = ns.randn(shape.toArray) * math.sqrt(2.0 / shape(1))
  var b: Tensor = ns.zeros(shape.head, 1)

  override def receive = {

    case Forward(x, y) =>
  }
}

object ConvGate {
  def props(next: ActorRef, config: ConvConfig) =
    Props(new ConvGate(next, config))
}

case class ConvConfig(shape: List[Int])
