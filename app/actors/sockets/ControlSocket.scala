package actors.sockets

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.Publish
import botkop.nn.gates.{Pause, Quit, SetLearningRate, Start}

class ControlSocket(socket: ActorRef) extends Actor with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator

  override def receive: Receive = {
    case "start" =>
      mediator ! Publish("control", Start)
    case "quit"  =>
      mediator ! Publish("control", Quit)
    case "pause"  =>
      mediator ! Publish("control", Pause)

    case s: String =>
      log.info(s"received $s")
      if (s.matches("learning-rate=.*")) {
        try {
          val learningRate = s.split("=").last.toDouble
          log.info(s"publishing learning rate $learningRate")
          mediator ! Publish("control", SetLearningRate(learningRate))
        } catch {
          case t: Throwable =>
            log.error(s"unable to parse $s")
        }
      }
  }
}

object ControlSocket {
  def props(socket: ActorRef) =
    Props(new ControlSocket(socket))
}
