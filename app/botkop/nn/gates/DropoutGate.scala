package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca
import botkop.numsca.Tensor

import scala.language.postfixOps

class DropoutGate(next: ActorRef, p: Double) extends Actor with ActorLogging {

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
      next ! eval

  }
}

object DropoutGate {
  def props(next: ActorRef, p: Double): Props =
    Props(new DropoutGate(next, p))
}
