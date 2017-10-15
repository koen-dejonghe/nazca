package botkop.data

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import botkop.nn.gates.{Forward, NextBatch}
import botkop.numsca.Tensor

import scala.language.postfixOps
import scala.util.Random

class Cifar10DataLoader(folder: String,
                        miniBatchSize: Int,
                        // entryGate: ActorRef,
                        seed: Long = 231)
    extends DataLoader
    // with Actor
    // with ActorLogging {
{
  Random.setSeed(seed)

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

  val files: List[(Float, File)] = getListOfFiles(folder)
  val m: Int = files.length
  val n: Int = {
    val image = ImageIO.read(files.head._2)
    image.getHeight * image.getWidth * 3
  }

  def nextBatch: (Tensor, Tensor) = {
    val sampleFiles = Seq.fill(miniBatchSize)(Random.nextInt(m)) map files
    val xData = sampleFiles map(_._2) flatMap readFile toArray
    val yData = sampleFiles map(_._1) toArray

    (Tensor(xData).reshape(miniBatchSize, n).transpose,
     Tensor(yData).reshape(miniBatchSize, 1).transpose)
  }

  def readFile(file: File): Seq[Float] = {
    val image: BufferedImage = ImageIO.read(file)

    val w = image.getWidth
    val h = image.getHeight
    val div: Float = 255

    val lol = for {
      i <- 0 until h
      j <- 0 until w
    }  yield {
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

  /*
  override def receive: Receive = accept(nextBatch)

  def accept(batch: (Tensor, Tensor)): Receive = {
    case NextBatch =>
      entryGate forward Forward(batch)
      context become accept(nextBatch)
  }
  */

}

/*
object Cifar10DataLoader {
  def props(folder: String,
            miniBatchSize: Int,
            entryGate: ActorRef,
            seed: Long = 231) =
    Props(new Cifar10DataLoader(folder, miniBatchSize, entryGate, seed))
}
*/
