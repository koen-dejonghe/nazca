package controllers

import javax.inject._

import actors.sockets.ControlSocket
import actors.sockets.MonitorSocket
import akka.actor.{ActorRef, ActorSystem}
import akka.stream.Materializer
import botkop.Driver
import play.api.libs.streams.ActorFlow
import play.api.mvc._

@Singleton
class NetController @Inject()(cc: ControllerComponents)(
    implicit system: ActorSystem,
    mat: Materializer)
    extends AbstractController(cc) {

  val monitor: ActorRef = system.actorOf(Driver.props())

  def index = Action {
    Ok(views.html.index())
  }

  def controlSocket: WebSocket = WebSocket.accept[String, String] { request =>
    ActorFlow.actorRef { out =>
      ControlSocket.props(out)
    }
  }

  def monitorSocket: WebSocket = WebSocket.accept[String, String] { request =>
    ActorFlow.actorRef { out =>
      MonitorSocket.props(out)
    }
  }
}
