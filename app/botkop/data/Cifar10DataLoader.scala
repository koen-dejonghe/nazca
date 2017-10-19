package botkop.data

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

import scala.language.postfixOps
import scala.util.Random

class Cifar10DataLoader(folder: String,
                        miniBatchSize: Int,
                        take: Option[Int] = None,
                        seed: Long = 231)
    extends DataLoader
    with LazyLogging {

  val labels = List(
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
  )

  val n: Int = 32 * 32 * 3

  val fileList: List[(Float, File)] = getListOfFiles(folder)

  override val numSamples: Int = fileList.length
  override val numBatches: Int =
    (numSamples / miniBatchSize) +
      (if (numSamples % miniBatchSize == 0) 0 else 1)

  override def iterator: Iterator[(Tensor, Tensor)] =
    new Random(seed)
      .shuffle(fileList)
      .take(take.getOrElse(numSamples))
      .sliding(miniBatchSize, miniBatchSize)
      .map { sampleFiles =>
        val xData = sampleFiles map (_._2) flatMap readFile toArray

        /*
        val xData = sampleFiles.foldLeft(List.empty[Float]) {
          case (xs, f) =>
            val x = readFile(f._2).toList
            x ::: xs
        } toArray
         */

        val yData = sampleFiles map (_._1) toArray

        val batchSize = sampleFiles.length

        (Tensor(xData).reshape(batchSize, n).transpose,
         Tensor(yData).reshape(batchSize, 1).transpose)
      }

  def readFile(file: File): Seq[Float] = {
    val image: BufferedImage = ImageIO.read(file)

    val w = image.getWidth
    val h = image.getHeight
    val div: Float = 255

    val lol = for {
      i <- 0 until h
      j <- 0 until w
    } yield {
      val pixel = image.getRGB(i, j)
      val red = ((pixel >> 16) & 0xff) / div
      val green = ((pixel >> 8) & 0xff) / div
      val blue = (pixel & 0xff) / div

      Seq(red, green, blue)
    }
    lol flatten
  }

  def getListOfFiles(dir: String): List[(Float, File)] = {
    val d = new File(dir)
    d.listFiles
      .filter(_.isFile)
      .toList
      .map { f =>
        val List(seq, cat) =
          f.getName.replaceFirst("\\.png", "").split("_").toList
        (seq.toInt, cat, f)
      }
      .sortBy(_._1)
      .map {
        case (i, cat, file) =>
          (labels.indexOf(cat).toFloat, file)
      }
  }

}
