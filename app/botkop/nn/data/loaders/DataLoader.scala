package botkop.nn.data.loaders

import botkop.numsca.Tensor

trait DataLoader extends Iterable[(Tensor, Tensor)] {
  def numSamples: Int
  def numBatches: Int
}

@SerialVersionUID(123L)
case class YX(y: Float, x: Array[Float]) extends Serializable

