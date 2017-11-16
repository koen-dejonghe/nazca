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

import scala.concurrent.duration._

import scala.language.postfixOps

class Driver extends Actor with Timers with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  implicit val projectName: String = "mnist"

  val template: NetworkConfig = ((Linear + Relu) * 2)
    .withDimensions(784, 50, 10)
    .withOptimizer(Nesterov)
    .withCostFunction(Softmax)
    .withRegularization(1e-8)
    .withLearningRate(0.4)
    .withLearningRateDecay(0.99)
    .configure

  val miniBatchSize = 64

  val trainingDataLoader =
    new MnistDataLoader(mode = "train", miniBatchSize)
  val devEvalDataLoader =
    new MnistDataLoader(mode = "dev", miniBatchSize)
  val trainEvalDataLoader =
    new MnistDataLoader(mode = "train",
                        miniBatchSize,
                        take = Some(devEvalDataLoader.numSamples))

  /*
  implicit val projectName: String = "cifar10LBR4"
  val template: NetworkConfig = ((Linear + BatchNorm + Relu + Dropout) * 4)
    .withDimensions(32 * 32 * 3, 50, 50, 50, 10)
    .withOptimizer(Nesterov)
    .withCostFunction(Softmax)
    .withRegularization(1e-4)
    .withLearningRate(0.4)
    .withLearningRateDecay(0.99)
    .networkConfig

  val miniBatchSize = 64

  val trainingDataLoader =
    new Cifar10DataLoader(mode = "train", miniBatchSize)
  val devEvalDataLoader =
    new Cifar10DataLoader(mode = "dev", miniBatchSize)
  val trainEvalDataLoader =
    new Cifar10DataLoader(mode = "train",
                          miniBatchSize,
                          take = Some(devEvalDataLoader.numSamples))
   */

  timers.startPeriodicTimer(PersistTick, PersistTick, 30 seconds)

  override def receive: Receive = empty(template)

  def empty(template: NetworkConfig): Receive = {

    case CanvasMessage("driver", "ready") =>
      mediator ! Publish(
        "control",
        CanvasMessage("socket", Json.prettyPrint(Json.toJson(template))))

    case Start =>
      log.debug("starting...")
      val nn = template.materialize

      val miniBatcher =
        context.system.actorOf(
          MiniBatcher.props(trainingDataLoader, nn.entryGate))
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
      context become empty(nn.config)

    case CanvasMessage("driver", "ready") =>
      mediator ! Publish(
        "control",
        CanvasMessage("socket", Json.prettyPrint(Json.toJson(template))))

    case CanvasMessage("driver", msg) =>
      log.debug("deploying new network")
      val nc = Json.fromJson[NetworkConfig](Json.parse(msg)).get
      nn.quit()
      context become empty(nc)

  }

  def running(nn: Network, miniBatcher: ActorRef): Receive = {

    case Pause =>
      log.info("pausing")
      context become paused(nn, miniBatcher)

    case Quit =>
      log.info("quitting")
      nn.quit()
      context become empty(nn.config)

    case Backward(_) =>
      miniBatcher ! NextBatch

    case PersistTick =>
      mediator ! Publish("control", Persist)

    case z =>
      log.warning(
        s"don't know how to handle message of type ${z.getClass.getCanonicalName}")
  }

}

object Driver {
  def props() = Props(new Driver())
}

