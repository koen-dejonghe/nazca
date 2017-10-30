package botkop.nn.costs

import botkop.numsca
import botkop.numsca.Tensor

case object CrossEntropy extends Cost {
  override def costFunction(yHat: Tensor, y: Tensor): (Double, Tensor) = {
    val m = y.shape(1)
    val cost = (-y.dot(numsca.log(yHat).T) -
      (1 - y).dot(numsca.log(1 - yHat).T)) / m

    val dal = -(y / yHat - (1 - y) / (1 - yHat))
    (cost.squeeze(), dal)
  }
}
