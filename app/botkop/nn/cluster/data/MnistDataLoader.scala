package botkop.nn.cluster.data

import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

import scala.io.Source
import scala.util.Random

class MnistDataLoader(miniBatchSize: Int = 16,
                      seed: Long = 231,
                      take: Option[Int] = None)
    extends DataLoader
    with LazyLogging {

  Random.setSeed(seed)
  val (xTrain, yTrain) = loadData("data/mnist_train.csv.gz", take)
  val (xDev, yDev) = loadData("data/mnist_test.csv.gz", take)

  override def nextTrainingBatch(): (Tensor, Tensor) = nextBatch(xTrain, yTrain)

  override def nextDevBatch(): (Tensor, Tensor) = nextBatch(xDev, yDev)

  def nextBatch(x: Tensor, y: Tensor): (Tensor, Tensor) = {
    val m = x.shape(1)
    if (miniBatchSize >= m) {
      (x, y)
    } else {
      val samples = Seq.fill(miniBatchSize)(Random.nextInt(m))
      val xb = new Tensor(x.array.getColumns(samples: _*))
      val yb = new Tensor(y.array.getColumns(samples: _*))
      (xb, yb)
    }
  }

  def loadData(fname: String, take: Option[Int] = None): (Tensor, Tensor) = {

    logger.info(s"loading data from $fname: start")

    val lines = take match {
      case Some(n) =>
        Source
          .fromInputStream(gzis(fname))
          .getLines()
          .take(n)
      case None =>
        Source
          .fromInputStream(gzis(fname))
          .getLines()
    }

    val (xData, yData) = lines
      .foldLeft(List.empty[Float], List.empty[Float]) {
        case ((xs, ys), line) =>
          val tokens = line.split(",")
          val (y, x) =
            (tokens.head.toFloat, tokens.tail.map(_.toFloat / 255).toList)
          (x ::: xs, y :: ys)
      }

    val x = Tensor(xData.toArray).reshape(yData.length, 784).transpose
    val y = Tensor(yData.toArray).reshape(yData.length, 1).transpose

    logger.info(s"loading data from $fname: done")

    (x, y)

  }

}
