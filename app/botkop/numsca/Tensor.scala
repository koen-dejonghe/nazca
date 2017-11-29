package botkop.numsca

import botkop.numsca
import org.nd4j.linalg.api.iter.NdIndexIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.JavaConverters._
import scala.language.{implicitConversions, postfixOps}
import scala.reflect.runtime.universe._

class Tensor(val array: INDArray, val isBoolean: Boolean = false)
    extends Serializable {

  def data: Array[Double] = array.dup.data.asDouble

  def copy(): Tensor = new Tensor(array.dup())

  def shape: Array[Int] = array.shape()
  def reshape(newShape: Array[Int]) = new Tensor(array.reshape(newShape: _*))
  def reshape(newShape: Int*) = new Tensor(array.reshape(newShape: _*))
  def shapeLike(t: Tensor): Tensor = reshape(t.shape)

  def transpose() = new Tensor(array.transpose())
  def T: Tensor = transpose()
  def transpose(axes: Array[Int]): Tensor = {
    require(axes.sorted sameElements shape.indices, "invalid axes")
    val newShape = axes.map(a => shape(a))
    reshape(newShape)
  }
  def transpose(axes: Int*): Tensor = transpose(axes.toArray)

  def round: Tensor =
    Tensor(data.map(math.round(_).toDouble)).reshape(this.shape)

  def dot(other: Tensor) = new Tensor(array mmul other.array)

  def unary_- : Tensor = new Tensor(array mul -1)
  def +(d: Double): Tensor = new Tensor(array add d)
  def -(d: Double): Tensor = new Tensor(array sub d)
  def *(d: Double): Tensor = new Tensor(array mul d)
  def **(d: Double): Tensor = power(this, d)
  def /(d: Double): Tensor = new Tensor(array div d)
  def %(d: Double): Tensor = new Tensor(array fmod d)

  def +=(d: Double): Unit = array addi d
  def -=(d: Double): Unit = array subi d
  def *=(d: Double): Unit = array muli d
  def **=(d: Double): Unit = array.assign(Transforms.pow(array, d))
  def /=(d: Double): Unit = array divi d
  def %=(d: Double): Unit = array fmodi d

  def >(d: Double): Tensor = new Tensor(array gt d, true)
  def >=(d: Double): Tensor = new Tensor(array gte d, true)
  def <(d: Double): Tensor = new Tensor(array lt d, true)
  def <=(d: Double): Tensor = new Tensor(array lte d, true)
  def ==(d: Double): Tensor = new Tensor(array eq d, true)
  def !=(d: Double): Tensor = new Tensor(array neq d, true)

  // todo when product of dim of other array = 1 then extract number iso broadcasting
  // def +(other: Tensor): Tensor = new Tensor(array add bc(other))
  // def -(other: Tensor): Tensor = new Tensor(array sub bc(other))
  // def *(other: Tensor): Tensor = new Tensor(array mul bc(other))
  // def /(other: Tensor): Tensor = new Tensor(array div bc(other))
  // def %(other: Tensor): Tensor = new Tensor(array fmod bc(other))
  // def >(other: Tensor): Tensor = new Tensor(array gt bc(other), true)
  // def <(other: Tensor): Tensor = new Tensor(array lt bc(other), true)
  // def ==(other: Tensor): Tensor = new Tensor(array eq bc(other), true)
  // def !=(other: Tensor): Tensor = new Tensor(array neq bc(other), true)

  def +(other: Tensor): Tensor = Ops.add(this, other)
  def -(other: Tensor): Tensor = Ops.sub(this, other)
  def *(other: Tensor): Tensor = Ops.mul(this, other)
  def /(other: Tensor): Tensor = Ops.div(this, other)
  def %(other: Tensor): Tensor = Ops.mod(this, other)
  def >(other: Tensor): Tensor = Ops.gt(this, other)
  def <(other: Tensor): Tensor = Ops.lt(this, other)
  def ==(other: Tensor): Tensor = Ops.eq(this, other)
  def !=(other: Tensor): Tensor = Ops.neq(this, other)

  def &&(other: Tensor): Tensor = {
    require(this.isBoolean && other.isBoolean)
    new Tensor(this.array.mul(other.array), true)
  }

  def ||(other: Tensor): Tensor = {
    require(this.isBoolean && other.isBoolean)
    new Tensor(Transforms.max(this.array.add(other.array), 1.0), true)
  }

  /**
    * broadcast argument tensor with shape of this tensor
    */
  private def bc(t: Tensor): INDArray = t.array.broadcast(shape: _*)

  def +=(t: Tensor): Unit = array addi bc(t)
  def -=(t: Tensor): Unit = array subi bc(t)
  def *=(t: Tensor): Unit = array muli bc(t)
  def /=(t: Tensor): Unit = array divi bc(t)
  def %=(t: Tensor): Unit = array fmodi bc(t)

  def :=(t: Tensor): Unit = array assign t.array
  def :=(d: Double): Unit = array assign d

  def maximum(other: Tensor): Tensor = Ops.max(this, other)
  def maximum(d: Double): Tensor = new Tensor(Transforms.max(this.array, d))
  def minimum(other: Tensor): Tensor = Ops.min(this, other)
  def minimum(d: Double): Tensor = new Tensor(Transforms.min(this.array, d))

  def slice(i: Int): Tensor = new Tensor(array.slice(i))
  def slice(i: Int, dim: Int): Tensor = new Tensor(array.slice(i, dim))

  def squeeze(): Double = {
    require(shape.product == 1)
    array.getDouble(0)
  }
  def squeeze(index: Int*): Double = array.getDouble(index: _*)
  def squeeze(index: Array[Int]): Double = squeeze(index: _*)

  def apply(index: Int*): Tensor = {
    val ix = index.map(NDArrayIndex.point)
    new Tensor(array.get(ix: _*))
  }
  def apply(index: Array[Int]): Tensor = apply(index: _*)

  def apply(ranges: NumscaRange*)(
      implicit tag: TypeTag[NumscaRange]): Tensor = {

    def handleNegIndex(i: Int, shapeIndex: Int) =
      if (i < 0) shape(shapeIndex) + i else i

    val indexes: Seq[INDArrayIndex] = ranges.zipWithIndex.map {
      case (nr, i) =>
        nr.to match {
          case None if nr.from == 0 =>
            NDArrayIndex.all()
          case None =>
            NDArrayIndex.interval(handleNegIndex(nr.from, i), shape(i))
          case Some(n) =>
            NDArrayIndex.interval(handleNegIndex(nr.from, i),
                                  handleNegIndex(n, i))
        }
    }
    new Tensor(array.get(indexes: _*))
  }

  /**
    * Slice by tensor
    * Note this does not return a view, but a new copy of the data!
    */
  /*
  def apply(t: Tensor): Tensor = {
    val indexes = indexBy(t)
    val newData = indexes.map(array getFloat)
    val s = Tensor(newData)
    if (t.isBoolean) s else s.reshape(t.shape)
  }
   */

  def apply(selection: Tensor*)(implicit dummy: Int = 0): Tensor = {
    val indexes = selectIndexes(selection)
    val newData = indexes.map(array getFloat)
    Tensor(newData)
  }

  private def selectIndexes(selection: Seq[Tensor]): Array[Array[Int]] = {
    if (selection.length == 1 && selection.head.isBoolean) {
      indexByBooleanTensor(selection.head)
    } else {
      multiIndex(selection)
    }
  }

  private def multiIndex(selection: Seq[Tensor]): Array[Array[Int]] = {
    val rank = selection.head.shape(1)
    require(selection.forall(s => s.shape.head == 1 && s.shape(1) == rank))

    (0 until rank).map { r =>
      selection.map(s => s.array.getInt(0, r)).toArray
    }.toArray
  }

  private def indexByBooleanTensor(t: Tensor): Array[Array[Int]] = {
    require(t.isBoolean)
    require(t sameShape this)

    new NdIndexIterator(t.shape: _*).asScala.filterNot { ii: Array[Int] =>
      t.array.getFloat(ii) == 0
    } toArray
  }

  /*
  private def indexByTensor(t: Tensor): Array[Array[Int]] = {
    require(shape.length == t.shape.length)
    require(t.shape.last == 1)
    require(shape.init sameElements t.shape.init)

    numsca.nditer(t).toArray.map { ii: Array[Int] =>
      val v = t.array.getInt(ii: _*)
      ii.init :+ v
    }
  }

  private def indexBy(t: Tensor): Array[Array[Int]] =
    if (t.isBoolean) indexByBooleanTensor(t) else indexByTensor(t)
   */

  // def put(index: Int*)(d: Double): Unit = put(index.toArray, d)

  def put(index: Array[Int], d: Double): Unit =
    array.put(NDArrayIndex.indexesFor(index: _*), d)

  /*
  def put(t: Tensor, d: Double): Unit =
    indexBy(t).foreach(ix => array.putScalar(ix, d))

  def put(t: Tensor, f: (Double) => Double): Unit =
    indexBy(t).foreach(ix => array.putScalar(ix, f(array.getFloat(ix))))

  def put(t: Tensor, f: (Array[Int], Double) => Double): Unit =
    indexBy(t).foreach(ix => array.putScalar(ix, f(ix, array.getFloat(ix))))
   */

  def put(d: Double)(selection: Tensor*): Unit =
    selectIndexes(selection).foreach(ix => array.putScalar(ix, d))

  def put(f: (Double) => Double)(selection: Tensor*): Unit =
    selectIndexes(selection).foreach { ix =>
      array.putScalar(ix, f(array.getFloat(ix)))
    }

  def put(f: (Array[Int], Double) => Double)(selection: Tensor*): Unit =
    selectIndexes(selection).foreach { ix =>
      array.putScalar(ix, f(ix, array.getFloat(ix)))
    }

  def sameShape(other: Tensor): Boolean = shape sameElements other.shape
  def sameElements(other: Tensor): Boolean = data sameElements other.data

  def rank: Int = array.rank()

  override def toString: String = array.toString

  /* also returns a new array, so useless
  def apply(t: Tensor): Tensor = {
    val ixs: Array[INDArray] = indexBy(t) map { ix: Array[Int] =>
      val points = ix map NDArrayIndex.point
      array.get(points: _*)
    }
    new Tensor(Nd4j.toFlattened(ixs.toSeq.asJava)).reshape(t.shape)
  }
 */

}

object Tensor {

  def apply(data: Array[Double]): Tensor = {
    val array = Nd4j.create(data)
    new Tensor(array)
  }

  def apply(data: Array[Float]): Tensor = {
    val array = Nd4j.create(data)
    new Tensor(array)
  }

  def apply(data: Double*): Tensor = Tensor(data.toArray)

}
