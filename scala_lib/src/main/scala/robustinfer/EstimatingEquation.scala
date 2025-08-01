package robustinfer

import breeze.linalg._
import org.apache.spark.sql.Dataset

case class Obs(
  i: String,                   // cluster ID
  x: Array[Double],           // covariates
  y: Double,                  // outcome
  timeIndex: Option[Int] = None,      // optional time index
  z: Option[Double] = None       // optional treatment indicator
)
case class EESummary(beta: DenseVector[Double], variance: DenseMatrix[Double])

abstract class EstimatingEquation {
  def fit(df: Dataset[Obs], maxIter: Int = 10, tol: Double = 1e-6, verbose: Boolean = true): Unit
  def summary(): EESummary
}