package botkop.nn.optimizers

import botkop.numsca
import botkop.numsca.Tensor
import play.api.libs.json.{Format, Json}

case class MomentumOptimizer(var learningRate: Double,
                             beta: Double = 0.9,
                             learningRateDecay: Double = 0.95)
    extends Optimizer {

  var vs = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    if (vs.isEmpty) {
      vs = parameters.map(numsca.zerosLike)
    }

    parameters.zip(gradients).zipWithIndex.foreach {
      case ((z, dz), i) =>
        vs(i) *= beta
        vs(i) += (1 - beta) * dz
        z -= learningRate * vs(i)
    }

  }

  override def setLearningRate(lr: Double): Unit = this.learningRate = lr

  override def updateLearningRate(): Unit = learningRate *= learningRateDecay
}

object MomentumOptimizer {
  implicit val f: Format[MomentumOptimizer] = Json.format
}
