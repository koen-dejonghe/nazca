package botkop.nn.gates

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Publish
import botkop.nn.costs.Cost
import botkop.numsca
import botkop.numsca.Tensor
import play.api.libs.json.{Format, Json}

// class OutputGate(costFunction: (Tensor, Tensor) => (Double, Tensor))
class OutputGate(config: OutputConfig) extends Actor with ActorLogging {

  import config._

  val name: String = self.path.name
  log.debug(s"my name is $name")

  val mediator: ActorRef = DistributedPubSub(context.system).mediator

  override def receive: Receive = accept()

  def accept(i: Int = 0): Receive = {

    case Forward(al, y) =>
      val (c, dal) = cost.costFunction(al, y)
      sender() ! Backward(dal)

      if (i % 10 == 0) {
        mediator ! Publish("monitor", CostLogEntry("train", i, c))
      }
      context become accept(i + 1)

    case Eval(source, id, x, y) =>
      val (c, _) = cost.costFunction(x, y)
      val acc = accuracy(x, y)
      sender ! EvalEntry(source, id, c, acc)

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
  def props(config: OutputConfig) = Props(new OutputGate(config: OutputConfig))
}

case class OutputConfig(cost: Cost) extends GateConfig {
  override def materialize(next: Option[ActorRef], index: Int)(
      implicit system: ActorSystem,
      projectName: String): ActorRef = {
    system.actorOf(OutputGate.props(this), Output.name(index))
  }
}
object OutputConfig {
  implicit val f: Format[OutputConfig] = Json.format
}
