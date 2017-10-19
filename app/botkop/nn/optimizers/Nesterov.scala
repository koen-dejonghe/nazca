package botkop.nn.optimizers

import botkop.numsca
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

import scala.language.postfixOps

case class Nesterov(var learningRate: Double,
                    learningRateDecay: Double = 0.95,
                    beta: Double = 0.9)
    extends Optimizer
    with LazyLogging {

  var vs = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    if (vs.isEmpty) {
      vs = parameters.map(numsca.zerosLike)
    }

    parameters.indices.foreach { i =>
      update(parameters(i), gradients(i), vs(i))
    }
  }

  def update(x: Tensor, dx: Tensor, v: Tensor): Unit = {
    val vPrev = v
    v *= beta
    v -= learningRate * dx
    x += (-beta * vPrev) + (1 + beta) * v
  }

  override def setLearningRate(lr: Double): Unit =
    this.learningRate = lr

  override def updateLearningRate(): Unit = {
    learningRate *= learningRateDecay
    logger.debug(s"learning rate is now $learningRate")
  }
}
