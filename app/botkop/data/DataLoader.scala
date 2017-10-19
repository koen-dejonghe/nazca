package botkop.data

import botkop.numsca.Tensor

trait DataLoader extends Iterable[(Tensor, Tensor)] {
  def numSamples: Int
  def numBatches: Int
}
