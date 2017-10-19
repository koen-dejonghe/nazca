package botkop.nn.optimizers

import botkop.numsca.Tensor

case class GradientDescent(var learningRate: Double,
                           learningRateDecay: Double = 0.95)
    extends Optimizer {

  override def update(parameters: List[Tensor], gradients: List[Tensor]): Unit =
    parameters.zip(gradients).foreach {
      case (z, dz) =>
        z -= dz * learningRate
    }

  override def setLearningRate(lr: Double): Unit =
    this.learningRate = lr

  override def updateLearningRate(): Unit =
    learningRate *= learningRateDecay
}
