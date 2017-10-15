package botkop

import akka.actor.{
  Actor,
  ActorLogging,
  ActorRef,
  ActorSystem,
  PoisonPill,
  Props
}
import botkop.data.Cifar10DataLoader
import botkop.nn.gates._
import botkop.nn.optimizers.Adam
import botkop.nn.costs._

class Monitor extends Actor with ActorLogging {

  implicit val system: ActorSystem = context.system

  val nn: Network =
    ((Linear + Relu + Dropout) * 2 + Linear)
      .withDimensions(32 * 32 * 3, 100, 50, 10)
      .withOptimizer(Adam(learningRate = 0.0001))
      .withCostFunction(softmaxCost)
      .withRegularization(1e-5)
      .build(self)

  val dataLoader: ActorRef = system.actorOf(
    Cifar10DataLoader.props("data/cifar-10/train", 16, nn.entryGate.get))

  override def receive: Receive = pauzed

  def pauzed: Receive = {
    case Start =>
      dataLoader ! NextBatch
      context become running
    case Quit =>
      nn.quit()
  }

  def running: Receive = {
    case Pause =>
      context become pauzed
    case Quit =>
      nn.quit()
    case Backward(_) =>
      dataLoader ! NextBatch
    case cl: CostLogEntry =>
      import cl._
      log.debug(s"mode: $mode id: $id cost: $cost")
    case z =>
      log.error(
        s"don't know how to handle message of type ${z.getClass.getCanonicalName}")
  }

}

object Monitor {
  def props() = Props(new Monitor())
}
