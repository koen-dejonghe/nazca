package botkop.nn.costs

import botkop.numsca
import botkop.numsca.Tensor

case object Softmax extends Cost {
  override def costFunction(xt: Tensor, yt: Tensor): (Double, Tensor) = {

    val x = xt.T
    val y = yt.T

    val shiftedLogits = x - numsca.max(x, axis = 1)
    val z = numsca.sum(numsca.exp(shiftedLogits), axis = 1)
    val logProbs = shiftedLogits - numsca.log(z)
    val probs = numsca.exp(logProbs)
    val n = x.shape(0)
    val loss = -numsca.sum(logProbs(y)) / n

    val dx = probs
    dx.put(y, _ - 1)
    dx /= n

    (loss, dx.T)
  }
}
