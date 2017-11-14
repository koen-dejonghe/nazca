package controllers

import javax.inject._

import akka.actor.{ActorRef, ActorSystem}
import akka.stream.Materializer
import botkop.nn.NetDriver
import play.api.libs.streams.ActorFlow
import play.api.mvc._
import sockets.{CanvasSocket, ControlSocket, MonitorSocket}

@Singleton
class NetController @Inject()(cc: ControllerComponents)(
    implicit system: ActorSystem,
    mat: Materializer)
    extends AbstractController(cc) {

  val monitor: ActorRef = system.actorOf(NetDriver.props(), "driver")

  def index = Action {
    Ok(views.html.index())
  }

  def controlSocket: WebSocket = WebSocket.accept[String, String] { _ =>
    ActorFlow.actorRef { out =>
      ControlSocket.props(out)
    }
  }

  def monitorSocket: WebSocket = WebSocket.accept[String, String] { _ =>
    ActorFlow.actorRef { out =>
      MonitorSocket.props(out)
    }
  }

  def canvasSocket: WebSocket = WebSocket.accept[String, String] { _ =>
    ActorFlow.actorRef { out =>
      CanvasSocket.props(out)
    }
  }

}
