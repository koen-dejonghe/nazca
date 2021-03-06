package botkop.nn.optimizers

import botkop.numsca
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging
import play.api.libs.json.{Format, Json}

import scala.language.postfixOps

case class AdamOptimizer(var learningRate: Double,
                         beta1: Double = 0.9,
                         beta2: Double = 0.999,
                         epsilon: Double = 1e-8,
                         learningRateDecay: Double = 0.95)
    extends Optimizer
    with LazyLogging {

  var t = 1

  var vs = List.empty[Tensor]
  var ss = List.empty[Tensor]

  override def update(parameters: List[Tensor],
                      gradients: List[Tensor]): Unit = {

    // first time through
    // create the cache
    if (t == 1) {
      vs = parameters.map(numsca.zerosLike)
      ss = parameters.map(numsca.zerosLike)
    }

    t = t + 1

    parameters.indices.foreach { i =>
      update(parameters(i), gradients(i), vs(i), ss(i), t)
    }
  }

  def update(x: Tensor, dx: Tensor, m: Tensor, v: Tensor, t: Int): Unit = {
    m *= beta1
    m += (1 - beta1) * dx
    val mt = m / (1 - math.pow(beta1, t))

    v *= beta2
    v += (1 - beta2) * numsca.square(dx)
    val vt = v / (1 - math.pow(beta2, t))

    x -= learningRate * mt / (numsca.sqrt(vt) + epsilon)
  }

  override def setLearningRate(lr: Double): Unit = {
    logger.info(s"changing learning rate from $learningRate to $lr")
    learningRate = lr
  }

  override def updateLearningRate(): Unit = {
    learningRate *= learningRateDecay
    logger.info(s"learning rate is now $learningRate")
  }
}

object AdamOptimizer {
  implicit val f: Format[AdamOptimizer] = Json.format
}
