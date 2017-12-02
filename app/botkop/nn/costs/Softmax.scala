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

  /*
  override def costFunction(xt: Tensor, yt: Tensor): (Double, Tensor) = {

    val x = xt.T
    val y = yt.T

    val shiftedLogits = x - ns.max(x, axis = 1)
    val z = ns.sum(ns.exp(shiftedLogits), axis = 1)
    val logProbs = shiftedLogits - ns.log(z)
    val probs = ns.exp(logProbs)
    val n = x.shape(0)

    println(logProbs.shape.toList)
    println(y.shape.toList)

    val loss = -ns.sum(logProbs(ns.arange(n).T, y)) / n

    val dx = probs
    // dx.put(y, _ - 1)
    dx.put(_ - 1)(ns.arange(n), y)
    dx /= n

    (loss, dx.T)
  }
  */
}
