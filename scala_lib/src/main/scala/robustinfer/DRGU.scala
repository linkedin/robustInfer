package robustinfer

import breeze.linalg.{DenseVector, DenseMatrix, norm}
import breeze.linalg.operators._
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.commons.math3.distribution.NormalDistribution
import robustinfer.UGEEUtils._

class DRGU extends EstimatingEquation with Serializable {
  private var theta: Map[String, DenseVector[Double]] = _
  private var variance: DenseMatrix[Double] = _
  private var df: Dataset[Obs] = _
  private var p: Int = _ // number of covariates
  private var t: Int = _ // number of observations in the first cluster

  // Override the required interface method
  def fit(
    df: Dataset[Obs],
    maxIter: Int,
    tol: Double,
    verbose: Boolean
  ): Boolean =
    fit(df, maxIter, tol, lambda = 0.0001, dampingOnly = false, verbose)

  def fit(
    data: Dataset[Obs],
    maxIter: Int,
    tol: Double,
    lambda: Double,
    dampingOnly: Boolean,
    verbose: Boolean
  ): Boolean = {
    import data.sparkSession.implicits._
    // Initialize variables
    initialize(data)

    val pairFeatureDS = sampleAllPairs(df).map(toPairFeatures)

    var diff = 10.0
    var iteration = 0

    while (diff > tol && iteration < maxIter) {
      val step = computeStep(pairFeatureDS, theta, lambda, dampingOnly)
      val thetaUpdated = updateTheta(theta, step)
      diff = norm(step, 2) // L2 norm of the step

      if (verbose) {
        println(s"Iteration $iteration: diff = $diff")
      }

      theta = thetaUpdated
      iteration += 1
    }

    val finalStep = computeStepWithSig(pairFeatureDS, theta, lambda, dampingOnly)
    theta = updateTheta(theta, finalStep._1)
    variance = finalStep._2

    val converged = norm(finalStep._1, 2) < tol
    if (!converged) {
      println(s"DRGU did not converge after $maxIter iterations")
    }

    if (verbose) {
      println(s"Final step norm: ${norm(finalStep._1, 2)}")
      println(s"Final parameter estimates:\n$theta")
      println(s"Final variance estimate:\n$variance")
    }

    converged
  }

  def fitMiniBatch(
    data: Dataset[Obs],
    k: Int = 10, // Partners per observation for parameter estimation
    maxEpochs: Int = 50, // Training epochs
    pairsPerBatch: Int = 10000, // Batch size for processing
    ema: Double = 0.0, // EMA smoothing for B/U
    lambda: Double = 1e-4, // L2 regularization
    s_variance: Int = -1, // Anchors for variance estimation (default: same as n)
    m_variance: Int = 20, // Partners per anchor for variance estimation
    learningRate: Double = 1.0, // Learning rate for parameter updates
    momentum: Double = 0.0, // Momentum coefficient (0.0 = no momentum, 0.1 = light momentum)
    tol: Double = 1e-6, // Convergence tolerance
    verbose: Boolean = true
  ): Boolean = {
    import data.sparkSession.implicits._

    // Initialize (reuse existing initialization)
    initialize(data)

    val d = theta("beta").length + theta("gamma").length + 1 // total params
    var B_running: DenseVector[Double] = DenseVector.zeros[Double](d * d)
    var U_running: DenseVector[Double] = DenseVector.zeros[Double](d)

    // Momentum variables
    var velocity: DenseVector[Double] = DenseVector.zeros[Double](d)

    var diff = 10.0
    var epoch = 0
    val Penalty = DenseMatrix.eye[Double](d)
    Penalty(0, 0) = 0.0 // don't penalize delta

    while (diff > tol && epoch < maxEpochs) {
      // Step 1: Generate k-partners pairs using partition-local sampling
      val pairs = sampleKPartnersWithinPartitions(df, k, seed = epoch)

      // Step 2: Process in batches
      val batchResults = pairs.rdd
        .mapPartitions { iter =>
          iter.grouped(pairsPerBatch).map(batch => computeBatchBU(batch.toArray, theta))
        }
        .collect()

      // Step 2: Aggregate batch results
      val batchAggregated = aggregateBatchStats(batchResults)
      val B_batch = batchAggregated._1
      val U_batch = batchAggregated._2

      // Step 2: Apply EMA smoothing
      B_running = applyEMA(B_running, B_batch, ema)
      U_running = applyEMA(U_running, U_batch, ema)

      // Existing Fisher step computation
      val B_matrix = new DenseMatrix(d, d, B_running.data)
      val J = -B_matrix

      val thetavector = DenseVector.vertcat(
        theta("delta"),
        theta("beta"),
        theta("gamma")
      )

      val step =
        generalizedInverse(J - lambda * Penalty) * (-U_running + lambda * Penalty * thetavector)

      // Apply momentum with learning rate: velocity = momentum * velocity + learningRate * step
      velocity = momentum * velocity + learningRate * step
      theta = updateTheta(theta, velocity)

      diff = norm(velocity, 2)

      if (verbose) {
        println(s"Epoch $epoch: diff norm = $diff")
      }

      epoch += 1
    }

    // Monte Carlo variance estimation using anchor-based sampling
    val n = df.count().toInt
    val s_total = if (s_variance == -1) n else s_variance // Default: same as sample size
    variance = computeMonteCarloVariance(
      df,
      theta,
      s_total = s_total,
      m = m_variance,
      lambda = lambda,
      penalty = Some(Penalty),
      seed = epoch
    )

    val converged = diff <= tol
    if (!converged) {
      println(s"DRGU mini-batch did not converge after $maxEpochs epochs")
    }

    if (verbose) {
      println(s"Final step norm: $diff")
      println(s"Final parameter estimates:\n$theta")
      println(s"Final variance estimate:\n$variance")
    }

    converged
  }

  def result(): EESummary = {
    val coef = DenseVector.vertcat(
      theta("delta") + 0.5,
      theta("beta"),
      theta("gamma")
    )
    EESummary(coef, variance)
  }

  def summary(): DataFrame = {
    if (theta == null || variance == null) {
      throw new IllegalStateException("Model has not been fitted yet.")
    }

    // calculate summary statistics
    // Extract SparkSession from df
    val spark = df.sparkSession
    import spark.implicits._ // Use stable identifier for implicits

    val delta = theta("delta")(0)
    val beta = theta("beta")
    val gamma = theta("gamma")

    // create column for name
    val names =
      Seq("delta") ++ (0 until beta.length).map(i => s"beta_$i") ++ (0 until gamma.length).map(i =>
        s"gamma_$i"
      )
    val values = Seq(delta + 0.5) ++ beta.toArray ++ gamma.toArray
    val Var = variance / df.select("i").distinct().count().toDouble // scale variance by number of clusters

    // add standard error from diagonal of variance matrix
    val stdErrors = (0 until Var.rows).map(i => math.sqrt(Var(i, i)))

    // add z scores, handling division by zero
    val valuesDiffH0 = Seq(delta) ++ beta.toArray ++ gamma.toArray
    val zScores = valuesDiffH0.zip(stdErrors).map {
      case (value, stdError) =>
        if (stdError != 0.0) value / stdError else Double.NaN
    }
    val normalDist = new NormalDistribution()
    val pValues = zScores.map(zScore => 2 * (1 - normalDist.cumulativeProbability(math.abs(zScore))))
    // create summary DataFrame
    import spark.implicits._
    import org.apache.spark.sql.functions._
    val summaryData = names
      .zip(values)
      .zip(stdErrors)
      .zip(zScores)
      .zip(pValues)
      .map {
        case ((((name, value), stdError), zScore), pValue) =>
          (name, value, stdError, zScore, pValue)
      }
      .toDF("parameter", "estimate", "std_error", "z_score", "p_value")

    summaryData
  }

  private def initialize(data: Dataset[Obs], checkClusterSize: Boolean = true): Unit = {
    df = data
    p = data.first().x.length

    if (checkClusterSize) {
      // Check if all clusters have the same size
      val clusterSizes = df.groupBy("i").count()
      val uniqueSizes = clusterSizes.select("count").distinct().collect().map(_.getLong(0)).toSeq
      if (uniqueSizes.length > 1) {
        throw new IllegalArgumentException("All clusters must have the same size")
      }
      t = uniqueSizes.head.toInt
    } else {
      // number of observations in the first cluster (reserve for when checks drag down performance)
      val firstClusterId = df.select("i").limit(1).collect().head.getString(0)
      t = df.filter(_.i == firstClusterId).count().toInt
    }

    if (t > 1) {
      throw new IllegalArgumentException(s"cluster size: $t > 1 is not supported yet")
    }

    theta = Map(
      "delta" -> DenseVector(0.0), // initial value for delta (shifted, so delta = 0.0 under null hypothesis)
      "beta" -> DenseVector.zeros[Double](p + 1), // where p is dim(x)
      "gamma" -> DenseVector.zeros[Double](2 * p + 1)
    )

    variance = DenseMatrix.eye[Double](3 * p + 3) // variance matrix for delta, beta, gamma
  }

}
