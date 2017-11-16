package botkop.nn

import akka.actor.{Actor, ActorLogging, ActorRef, Props, Timers}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Subscribe
import botkop.nn.data.loaders.DataLoader
import botkop.nn.data.{Evaluator, MiniBatcher}
import botkop.nn.gates._
import botkop.nn.network.{Network, NetworkConfig}
import play.api.libs.json.{Format, Json}

import scala.concurrent.duration._
import scala.language.postfixOps

class NetDriver extends Actor with Timers with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  /**
    * initial state
    */
  override def receive: Receive = {
    case project: Project =>
      val dp = deploy(project)
      context become deployed(dp)
  }

  /**
    * deployed state
    */
  def deployed(deployedProject: DeployedProject): Receive = {

    case Start =>
      start(deployedProject)
      context become running(deployedProject)

    case Quit =>
      undeploy(deployedProject)
      context become receive
  }

  /**
    * running state
    */
  def running(deployedProject: DeployedProject): Receive = {

    case Backward(_) =>
      deployedProject.miniBatcher ! NextBatch

    case Pause =>
      context become paused(deployedProject)

    case Quit =>
      undeploy(deployedProject)
      context become receive
  }

  /**
    * paused state
    */
  def paused(deployedProject: DeployedProject): Receive = {

    case Start =>
      start(deployedProject)
      context become running(deployedProject)

    case Quit =>
      undeploy(deployedProject)
      context become receive

  }

  /**
    * deploy the project (project trigger)
    */
  def deploy(project: Project): DeployedProject = {

    import project._

    val nn = template.materialize(context, name)

    val trainingDataLoader =
      DataLoader.instance(dataSet, "train", miniBatchSize)

    val devEvalDataLoader =
      DataLoader.instance(dataSet, "dev", miniBatchSize)

    val trainEvalDataLoader =
      DataLoader.instance(dataSet,
                          "train",
                          miniBatchSize,
                          take = Some(devEvalDataLoader.numSamples))

    val miniBatcher =
      context.system.actorOf(
        MiniBatcher.props(trainingDataLoader, nn.entryGate))

    // create the evaluators
    context.actorOf(
      Evaluator.props("dev-eval", devEvalDataLoader, nn.entryGate))

    context.actorOf(
      Evaluator.props("train-eval", trainEvalDataLoader, nn.entryGate))

    DeployedProject(project, nn, miniBatcher)
  }

  /**
    * undeploy the project (quit trigger)
    */
  def undeploy(deployedProject: DeployedProject): Unit = {
    timers.cancelAll()
    deployedProject.network.quit()
  }

  /**
    * start trigger
    */
  def start(deployedProject: DeployedProject): Unit = {
    import deployedProject._
    import project.{persistenceFrequency => f}

    miniBatcher ! NextBatch
    if (f > 0)
      timers.startPeriodicTimer(PersistTick, PersistTick, f seconds)
  }

}

object NetDriver {
  def props() = Props(new NetDriver())
}

case class Project(name: String,
                   miniBatchSize: Int,
                   dataSet: String,
                   persistenceFrequency: Int,
                   template: NetworkConfig)

object Project {
  implicit val f: Format[Project] = Json.format
}

case class DeployedProject(project: Project,
                           network: Network,
                           miniBatcher: ActorRef)
