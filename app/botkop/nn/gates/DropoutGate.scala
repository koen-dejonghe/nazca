package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import botkop.numsca
import botkop.numsca.Tensor
import play.api.libs.json.{Format, Json}

import scala.language.postfixOps

class DropoutGate(next: ActorRef, config: DropoutConfig)
    extends Actor
    with ActorLogging {

  import config._

  val name: String = self.path.name
  log.debug(s"my name is $name")

  override def receive: Receive = accept()

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {
    case Forward(x, y) =>
      val mask = (numsca.rand(x.shape) < p) / p
      val h = x * mask
      next ! Forward(h, y)
      context become accept(Some(sender(), mask))

    case Backward(dout) if cache isDefined =>
      val (prev, mask) = cache.get
      val dx = dout * mask
      prev ! Backward(dx)

    case predict: Predict =>
      next forward predict

    case eval: Eval =>
      next forward eval

  }
}

object DropoutGate {
  def props(next: ActorRef, config: DropoutConfig): Props =
    Props(new DropoutGate(next, config)).withDispatcher("gate-dispatcher")
}

case class DropoutConfig(p: Double = 0.5) extends GateConfig {
  override def materialize(next: Option[ActorRef], index: Int)(
      implicit system: ActorSystem,
      projectName: String): ActorRef = {
    system.actorOf(DropoutGate.props(next.get, this), Dropout.name(index))
  }
}
object DropoutConfig {
  implicit val f: Format[DropoutConfig] = Json.format
}
