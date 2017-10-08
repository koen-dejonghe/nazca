package botkop.nn.cluster.data

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

import scala.reflect.ClassTag
import scala.util.Random

trait DataLoader extends LazyLogging {

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

  def reservoirSample[T: ClassTag](input: Iterator[T], k: Int): Array[T] = {
    val reservoir = new Array[T](k)
    // Put the first k elements in the reservoir.
    var i = 0
    while (i < k && input.hasNext) {
      val item = input.next()
      reservoir(i) = item
      i += 1
    }

    if (i < k) {
      // If input size < k, trim the array size
      reservoir.take(i)
    } else {
      // If input size > k, continue the sampling process.
      while (input.hasNext) {
        val item = input.next
        val replacementIndex = Random.nextInt(i)
        if (replacementIndex < k) {
          reservoir(replacementIndex) = item
        }
        i += 1
      }
      reservoir
    }
  }

  def nextTrainingBatch(): (Tensor, Tensor)
  def nextDevBatch(): (Tensor, Tensor)
}
