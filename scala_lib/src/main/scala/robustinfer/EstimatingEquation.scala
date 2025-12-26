package robustinfer

import breeze.linalg._
import org.apache.spark.sql.Dataset

case class Obs(
  i: String, // cluster ID
  x: Array[Double], // covariates
  y: Double, // outcome
  timeIndex: Option[Int] = None, // optional time index
  z: Option[Double] = None // optional treatment indicator
)
case class EESummary(coef: DenseVector[Double], variance: DenseMatrix[Double])

abstract class EstimatingEquation {
  def fit(
    df: Dataset[Obs],
    maxIter: Int,
    tol: Double,
    verbose: Boolean
  ): Boolean
  def result(): EESummary
}
