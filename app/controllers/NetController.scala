package controllers

import javax.inject._

import akka.actor.{ActorSelection, ActorSystem}
import botkop.nn.gates._
import botkop.nn.optimizers._
import botkop.nn.costs._
import play.api.mvc._

/**
  * This controller creates an `Action` to handle HTTP requests to the
  * application's home page.
  */
@Singleton
class NetController @Inject()(cc: ControllerComponents)(implicit system: ActorSystem)
    extends AbstractController(cc) {

  def index = Action {
    val proxy: ActorSelection =
      system.actorSelection("akka://NeuralSystem@127.0.0.1:25520/user/proxy")

    val nn: Network =
      ((Linear + Relu + Dropout) * 2 + Linear)
        .withDimensions(784, 100, 50, 10)
        .withOptimizer(Adam(learningRate = 0.0001))
        .withCostFunction(softmaxCost)
        .withRegularization(1e-5)

    proxy ! Quit

    Thread.sleep(2000)

    proxy ! nn

    Thread.sleep(2000)

    proxy ! Start

    Ok(views.html.index("Your new application is ready."))
  }

}
