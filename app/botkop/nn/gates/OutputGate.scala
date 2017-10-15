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

      if (i % 1000 == 0) {
        listener ! CostLogEntry("train", i, cost)
      }
      context become accept(i + 1)

    case Eval(mode, id, x, y) =>
      val (cost, _) = costFunction(x, y)
      listener ! CostLogEntry(mode, id, cost)

    case Predict(x) =>
      listener ! x

  }
}

object OutputGate {
  def props(costFunction: (Tensor, Tensor) => (Double, Tensor),
            listener: ActorRef) =
    Props(new OutputGate(costFunction, listener))
}

