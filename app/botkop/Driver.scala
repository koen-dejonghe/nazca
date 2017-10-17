package botkop

import akka.actor._
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.data.{Cifar10DataLoader, MnistDataLoader}
import botkop.nn.costs._
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
  // def optimizer = Nesterov(learningRate = 0.1)

  val template: Network = (Linear + Relu + Linear)
    // .withDimensions(784, 50, 10)
    .withDimensions(32 * 32 * 3, 100, 10)
    .withOptimizer(optimizer)
    .withCostFunction(softmaxCost)
    .withRegularization(1e-5)

  val trainingDataLoader = new Cifar10DataLoader("data/cifar-10/train", 64)

  val devDataLoader = new Cifar10DataLoader("data/cifar-10/test", 1024)
  val trainEvalDataLoader = new Cifar10DataLoader("data/cifar-10/train", 1024)

  // val trainingDataLoader = new MnistDataLoader("data/mnist/mnist_train.csv.gz", 16)
  // val devDataLoader = new MnistDataLoader("data/mnist/mnist_test.csv.gz", 2048)

  val devEvaluator: ActorRef =
    system.actorOf(Evaluator.props("dev-eval", devDataLoader, 30 seconds))

  val trainEvaluator: ActorRef =
    system.actorOf(Evaluator.props("train-eval", trainEvalDataLoader, 30 seconds))

  timers.startPeriodicTimer(PersistTick, PersistTick, 30 seconds)

  override def receive: Receive = empty

  def empty: Receive = {
    case Start =>
      val nn = template.build
      val miniBatcher =
        system.actorOf(MiniBatcher.props(trainingDataLoader, nn.entryGate.get))
      miniBatcher ! NextBatch

      devEvaluator ! nn
      trainEvaluator ! nn

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
