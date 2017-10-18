package botkop.data

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

import scala.io.Source
import scala.util.Random

class MnistDataLoader(file: String, miniBatchSize: Int, seed: Long = 231) extends DataLoader with LazyLogging {

  Random.setSeed(seed)

  val (x, y) = loadData(file)
  val m = x.shape(1)

  override def nextBatch: (Tensor, Tensor) = {
    val samples = Seq.fill(miniBatchSize)(Random.nextInt(m))
    val xb = new Tensor(x.array.getColumns(samples: _*))
    val yb = new Tensor(y.array.getColumns(samples: _*))
    (xb, yb)
  }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

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
