package botkop.nn.cluster

import akka.actor._
import botkop.nn.cluster.data.DataLoader
import botkop.nn.gates._
import akka.actor.Timers

import scala.concurrent.duration._
import scala.language.postfixOps

class Proxy(dataLoader: DataLoader, snapshotInterval: Long)
    extends Actor
    with ActorLogging
    with Timers {
  import Proxy._
  implicit val system: ActorSystem = context.system

  if (snapshotInterval > 0)
    timers.startPeriodicTimer(SnapshotTick,
                              SnapshotTick,
                              snapshotInterval millis)

  override def receive: Receive = notInitialized

  def initialized(invoker: ActorRef,
                  network: Network,
                  iteration: Int): Receive = {

    case Network =>
      log.error("network already initialized")

    case Quit =>
      log.info("stopping network")
      network.actors.foreach(a => a ! PoisonPill)
      context become notInitialized

    case SnapshotTick =>
      network.actors.foreach(a => a ! Persist)

    case Backward(_) =>
      sender() ! Forward(dataLoader.nextTrainingBatch())
      context become initialized(invoker, network, iteration + 1)

    case Start =>
      network.actors.head ! Forward(dataLoader.nextTrainingBatch())
      context become initialized(invoker, network, iteration + 1)

  }

  def notInitialized: Receive = {
    case n: Network =>
      log.info("initializing network")
      context become initialized(sender(), n.build(sender()), 0)
  }
}

object Proxy {
  case object SnapshotTick

  def props(dataLoader: DataLoader, snapshotInterval: Long) =
    Props(new Proxy(dataLoader, snapshotInterval))
}
