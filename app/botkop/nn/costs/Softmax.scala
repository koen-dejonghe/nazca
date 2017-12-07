package botkop.nn.costs

import botkop.{numsca => ns}
import ns.Tensor

case object Softmax extends Cost {

  override def costFunction(x: Tensor, y: Tensor): (Double, Tensor) = {
    val shiftedLogits = x - ns.max(x, axis = 1)
    val z = ns.sum(ns.exp(shiftedLogits), axis = 1)
    val logProbs = shiftedLogits - ns.log(z)
    val probs = ns.exp(logProbs)
    val n = x.shape(0)

    val loss = -ns.sum(logProbs(ns.arange(n), y)) / n

    val dx = probs
    dx(ns.arange(n), y) -= 1
    dx /= n

    (loss, dx)
  }

}
