package controllers

import javax.inject._

import actors.sockets.ControlSocket
import akka.actor.{ActorRef, ActorSystem}
import akka.stream.Materializer
import botkop.Monitor
import play.api.libs.streams.ActorFlow
import play.api.mvc._

/**
  * This controller creates an `Action` to handle HTTP requests to the
  * application's home page.
  */
@Singleton
class NetController @Inject()(cc: ControllerComponents)(
    implicit system: ActorSystem,
    mat: Materializer)
    extends AbstractController(cc) {

  val monitor: ActorRef = system.actorOf(Monitor.props())

  def index = Action {
    Ok(views.html.index())
  }

  def socket: WebSocket = WebSocket.accept[String, String] { request =>
    ActorFlow.actorRef { out =>
      ControlSocket.props(out, monitor)
    }
  }

}
