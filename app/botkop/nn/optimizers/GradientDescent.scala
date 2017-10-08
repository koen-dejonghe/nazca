package botkop.nn.optimizers

import botkop.numsca.Tensor


case class GradientDescent(learningRate: Double) extends Optimizer {

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit =
    parameters.zip(gradients).foreach {
      case (z, dz) =>
        z -= dz * learningRate
    }
}
