package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.numsca.Tensor

class OutputGate(costFunction: (Tensor, Tensor) => (Double, Tensor),
                 listener: ActorRef)
    extends Actor
    with ActorLogging {

  val name: String = self.path.name
  log.debug(s"my name is $name")

  override def receive: Receive = accept()

  def accept(i: Int = 0): Receive = {

    case Forward(al, y) =>
      val (cost, dal) = costFunction(al, y)
      sender() ! Backward(dal)

      listener ! CostLogEntry(i, cost)

      if (i % 1000 == 0) {
        log.debug(s"iteration: $i cost: $cost")
      }

      context become accept(i + 1)

    case Predict(x) =>
      listener ! x

  }
}

object OutputGate {
  def props(costFunction: (Tensor, Tensor) => (Double, Tensor),
            listener: ActorRef) =
    Props(new OutputGate(costFunction, listener))
}

case class CostLogEntry(iteration: Int, cost: Double)
