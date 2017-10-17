package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca
import botkop.numsca.Tensor

import scala.language.postfixOps

class ReluGate(next: ActorRef) extends Actor with ActorLogging {

  val name: String = self.path.name
  log.debug(s"my name is $name")

  override def receive: Receive = accept()

  def activate(z: Tensor): Tensor = numsca.maximum(z, 0.0)

  def accept(cache: Option[(ActorRef, Tensor)] = None): Receive = {
    case Forward(z, y) =>
      val a = activate(z)
      next ! Forward(a, y)
      context become accept(Some(sender(), z))

    case Backward(da) if cache isDefined =>
      val (prev, z) = cache.get
      val dz = da * (z > 0.0)
      prev ! Backward(dz)

    case Predict(z) =>
      val a = activate(z)
      next forward Predict(a)

    case Eval(source, id, x, y) =>
      val z = activate(x)
      next ! Eval(source, id, z, y)

  }
}

object ReluGate {
  def props(next: ActorRef) = Props(new ReluGate(next))
}
