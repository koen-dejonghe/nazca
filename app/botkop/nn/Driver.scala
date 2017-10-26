package botkop.nn

import akka.actor._
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.nn.costs._
import botkop.nn.data.loaders.Cifar10DataLoader
import botkop.nn.data.{Evaluator, MiniBatcher}
import botkop.nn.gates._
import botkop.nn.optimizers.{Adam, Nesterov}

import scala.concurrent.duration._
import scala.language.postfixOps

class Driver extends Actor with Timers with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  implicit val system: ActorSystem = context.system

  def optimizer = Adam(learningRate = 0.001)
  // def optimizer = Nesterov(learningRate = 0.3, learningRateDecay = 0.99)

  val template: Network = ((Linear + Relu) * 2)
  // .withDimensions(784, 50, 10)
    .withDimensions(32 * 32 * 3, 50, 10)
    .withOptimizer(optimizer)
    .withCostFunction(softmaxCost)
    // .withRegularization(1e-5)

  val miniBatchSize = 128
  val trainingDataLoader =
    new Cifar10DataLoader(mode = "train", miniBatchSize
       , take=Some(128)
    )
  val devEvalDataLoader =
    new Cifar10DataLoader(mode = "dev", miniBatchSize, take=Some(128))
  val trainEvalDataLoader =
    new Cifar10DataLoader(mode = "train", miniBatchSize, take = Some(128))

  // val trainingDataLoader =
  // new MnistDataLoader("data/mnist/mnist_train.csv.gz", 16)
  // val devEvalDataLoader =
  // new MnistDataLoader("data/mnist/mnist_test.csv.gz", 256)
  // val trainEvalDataLoader =
  // new MnistDataLoader("data/mnist/mnist_train.csv.gz", 256, take = Some(2048))

  timers.startPeriodicTimer(PersistTick, PersistTick, 30 seconds)

  override def receive: Receive = empty

  def empty: Receive = {
    case Start =>
      log.debug("starting...")
      val nn = template.build

      val miniBatcher =
        system.actorOf(MiniBatcher.props(trainingDataLoader, nn.entryGate.get))
      miniBatcher ! NextBatch

      val devEvaluator: ActorRef =
        system.actorOf(
          Evaluator.props("dev-eval", devEvalDataLoader, nn.entryGate.get))

      val trainEvaluator: ActorRef =
        system.actorOf(
          Evaluator.props("train-eval", trainEvalDataLoader, nn.entryGate.get))

      context become running(nn, miniBatcher)
  }

  def paused(nn: Network, miniBatcher: ActorRef): Receive = {
    case Start =>
      miniBatcher ! NextBatch
      context become running(nn, miniBatcher)
    case Quit =>
      log.info("quitting")
      nn.quit()
      context become empty
  }

  def running(nn: Network, miniBatcher: ActorRef): Receive = {
    case Pause =>
      log.info("pausing")
      context become paused(nn, miniBatcher)
    case Quit =>
      log.info("quitting")
      nn.quit()
      context become empty
    case Backward(_) =>
      miniBatcher ! NextBatch
    case PersistTick =>
      mediator ! Publish("control", Persist)
    case z =>
      log.error(
        s"don't know how to handle message of type ${z.getClass.getCanonicalName}")
  }

}

object Driver {
  def props() = Props(new Driver())
}

object PersistTick
