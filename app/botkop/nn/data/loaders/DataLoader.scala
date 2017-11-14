package botkop.nn.data.loaders

import botkop.numsca.Tensor

trait DataLoader extends Iterable[(Tensor, Tensor)] {
  def numSamples: Int
  def numBatches: Int
}

object DataLoader {
  def instance(dataSet: String,
                 mode: String,
                 miniBatchSize: Int,
                 take: Option[Int] = None): DataLoader =
    dataSet match {
      case "cifar-10" =>
        new Cifar10DataLoader(mode, miniBatchSize, take)
      case "mnist" =>
        new MnistDataLoader(mode, miniBatchSize, take)
    }
}

@SerialVersionUID(123L)
case class YX(y: Float, x: Array[Float]) extends Serializable
