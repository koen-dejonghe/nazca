package botkop.nn.optimizers

import botkop.numsca
import botkop.numsca.Tensor

import scala.language.postfixOps

case class Nesterov(var learningRate: Double,
                    learningRateDecay: Double = 1.0 - 1e-8,
                    beta: Double = 0.9)
    extends Optimizer {

  var vs = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    if (vs.isEmpty) {
      vs = parameters.map(numsca.zerosLike)
    }

    learningRate *= learningRateDecay

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

  override def setLearningRate(lr: Double): Unit = this.learningRate = lr
}
