package botkop

import akka.actor._
import botkop.data.Cifar10DataLoader
import botkop.nn.gates._
import botkop.nn.optimizers.Adam
import botkop.nn.costs._
import botkop.nn.data.MiniBatchFetcher

class Monitor extends Actor with ActorLogging {

  implicit val system: ActorSystem = context.system

  val template: Network = ((Linear + Relu + Dropout) * 2 + Linear)
    .withDimensions(32 * 32 * 3, 100, 50, 10)
    .withOptimizer(Adam(learningRate = 0.0001))
    .withCostFunction(softmaxCost)
    .withRegularization(1e-5)

  val dataLoader = new Cifar10DataLoader("data/cifar-10/train", 16)

  override def receive: Receive = empty

  def empty: Receive = {
    case Start =>
      val nn = template.build(self)
      val mbf = system.actorOf(
        MiniBatchFetcher.props(dataLoader, nn.entryGate.get))
      mbf ! NextBatch
      context become running(nn, mbf)
  }

  def pauzed(nn: Network, dataLoader: ActorRef): Receive = {
    case Start =>
      dataLoader ! NextBatch
      context become running(nn, dataLoader)
    case Quit =>
      log.info("quitting")
      nn.quit()
      context become empty
  }

  def running(nn: Network, dataLoader: ActorRef): Receive = {
    case Pause =>
      log.info("pausing")
      context become pauzed(nn, dataLoader)
    case Quit =>
      log.info("quitting")
      nn.quit()
      context become empty
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
