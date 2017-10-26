package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Publish
import botkop.numsca
import botkop.numsca.Tensor

class OutputGate(costFunction: (Tensor, Tensor) => (Double, Tensor))
    extends Actor
    with ActorLogging {

  val name: String = self.path.name
  log.debug(s"my name is $name")

  val mediator: ActorRef = DistributedPubSub(context.system).mediator

  override def receive: Receive = accept()

  def accept(i: Int = 0): Receive = {

    case Forward(al, y) =>
      val (cost, dal) = costFunction(al, y)
      sender() ! Backward(dal)

      if (i % 10 == 0) {
        mediator ! Publish("monitor", CostLogEntry("train", i, cost))
      }
      context become accept(i + 1)

    case Eval(source, id, x, y) =>
      val (cost, _) = costFunction(x, y)
      val acc = accuracy(x, y)
      sender ! EvalEntry(source, id, cost, acc)

    case p: Predict =>
      mediator ! Publish("predict", p)

  }

  def accuracy(x: Tensor, y: Tensor): Double = {
    val m = x.shape(1)
    val p = numsca.argmax(x, 0)
    numsca.sum(p == y) / m
  }

}

object OutputGate {
  def props(costFunction: (Tensor, Tensor) => (Double, Tensor)) =
    Props(new OutputGate(costFunction))
}
