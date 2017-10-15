package botkop.data

import botkop.numsca.Tensor

trait DataLoader {
  def nextBatch: (Tensor, Tensor)
}
