package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca
import botkop.numsca.Tensor

import scala.language.postfixOps

class SigmoidGate(next: ActorRef) extends Actor with ActorLogging {

  val name: String = self.path.name
  log.debug(s"my name is $name")

  override def receive: Receive = accept()

  def activate(z: Tensor): Tensor = numsca.sigmoid(z)

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {

    case Forward(z, y) =>
      val a = activate(z)
      next ! Forward(a, y)
      context become accept(Some(sender(), a)) // !!! note passing the sigmoid in the cache

    case Backward(da) if cache isDefined =>
      val (prev, s) = cache.get
      val dz = da * s * (1 - s)
      prev ! Backward(dz)

    case Predict(x) =>
      val a = activate(x)
      next forward Predict(a)

    case Eval(source, id, x, y) =>
      val z = activate(x)
      next ! Eval(source, id, z, y)

  }
}

object SigmoidGate {
  def props(next: ActorRef) = Props(new SigmoidGate(next))
}
