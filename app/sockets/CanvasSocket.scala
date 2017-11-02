package sockets

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import akka.cluster.pubsub.DistributedPubSub
import akka.cluster.pubsub.DistributedPubSubMediator.{Publish, Subscribe}
import botkop.nn.gates.CanvasMessage

class CanvasSocket(socket: ActorRef) extends Actor with ActorLogging {

  val mediator: ActorRef = DistributedPubSub(context.system).mediator
  mediator ! Subscribe("control", self)

  mediator ! Publish("control", "ready")

  override def receive: Receive = {
    // from socket
    case s: String =>
      mediator ! Publish("control", CanvasMessage("driver", s))
    // from mediator
    case CanvasMessage("socket", message) =>
      socket ! message
  }
}

object CanvasSocket {
  def props(socket: ActorRef) =
    Props(new CanvasSocket(socket))
}
