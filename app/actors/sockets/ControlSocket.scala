package actors.sockets

import akka.actor.{Actor, ActorRef, Props}
import botkop.nn.gates.{Pause, Quit, Start}

class ControlSocket(socket: ActorRef, monitor: ActorRef) extends Actor {
  override def receive: Receive = {
    case "start" => monitor ! Start
    case "quit"  => monitor ! Quit
    case "pause"  => monitor ! Pause
  }
}

object ControlSocket {
  def props(socket: ActorRef, monitor: ActorRef) =
    Props(new ControlSocket(socket, monitor))
}
