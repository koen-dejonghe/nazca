package botkop.nn

import akka.actor._
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.nn.costs._
import botkop.nn.data.loaders._
import botkop.nn.data._
import botkop.nn.gates._
import botkop.nn.network._
import botkop.nn.optimizers._
import play.api.libs.json.Json

import scala.language.postfixOps

class Driver extends Actor with Timers with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  // implicit val system: ActorSystem = context.system

  //
  // def optimizer = AdamOptimizer(learningRate = 0.001, learningRateDecay = 0.95)
  // def optimizer = Nesterov(learningRate = 0.3, learningRateDecay = 0.99)

  implicit val projectName: String = "cifar10LBR2"

  val template: NetworkBuilder = ((Linear + Relu) * 2)
    .withDimensions(784, 50, 10)
    // .withDimensions(32 * 32 * 3, 50, 10)
    .withOptimizer(Nesterov)
    .withCostFunction(Softmax)
    // .withRegularization(1e-3)
    .withLearningRate(0.3)
    .withLearningRateDecay(1.0)

  log.debug(Json.prettyPrint(Json.toJson(template.networkConfig)))

  val miniBatchSize = 64

  // val trainingDataLoader =
  // new Cifar10DataLoader(mode = "train", miniBatchSize)
  // val devEvalDataLoader =
  // new Cifar10DataLoader(mode = "dev", miniBatchSize)
  // val trainEvalDataLoader =
  // new Cifar10DataLoader(mode = "train",
  // miniBatchSize,
  // take = Some(devEvalDataLoader.numSamples))

  val trainingDataLoader =
    new MnistDataLoader("data/mnist/mnist_train.csv.gz", miniBatchSize)
  val devEvalDataLoader =
    new MnistDataLoader("data/mnist/mnist_test.csv.gz", miniBatchSize)
  val trainEvalDataLoader =
    new MnistDataLoader("data/mnist/mnist_train.csv.gz",
                        miniBatchSize,
                        take = Some(devEvalDataLoader.numSamples))

  // timers.startPeriodicTimer(PersistTick, PersistTick, 30 seconds)

  override def receive: Receive = empty

  def empty: Receive = {
    case Start =>
      log.debug("starting...")
      val nn = template.build

      val miniBatcher =
        context.system.actorOf(MiniBatcher.props(trainingDataLoader, nn.entryGate))
      miniBatcher ! NextBatch

      val devEvaluator: ActorRef =
        context.actorOf(
          Evaluator.props("dev-eval", devEvalDataLoader, nn.entryGate))

      val trainEvaluator: ActorRef =
        context.actorOf(
          Evaluator.props("train-eval", trainEvalDataLoader, nn.entryGate))

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
