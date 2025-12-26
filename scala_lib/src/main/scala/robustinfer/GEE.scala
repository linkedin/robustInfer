package robustinfer

import org.apache.spark.sql.Dataset
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Rand
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import breeze.stats.distributions.RandBasis

object GEEUtils {
  // Compute U_i and B_i for one cluster
  def computeClusterStats(
    cluster: Seq[Obs],
    beta: DenseVector[Double],
    R: DenseMatrix[Double],
    eps: Double,
    family: DistributionFamily = Binomial,
    kappa: Double = 0.0,
    phi: Double = 1.0
  ): (DenseVector[Double], DenseMatrix[Double]) = {
    val X_i = DenseMatrix(cluster.map(_.x): _*)
    val Y_i = DenseVector(cluster.map(_.y): _*)

    val mu_i = family match {
      case Binomial                                => sigmoid(X_i * beta)
      case Gaussian                                => X_i * beta
      case Poisson | NegativeBinomial | Tweedie(_) => breeze.numerics.exp(X_i * beta)
    }

    val A_diag = family match {
      case Binomial         => mu_i.map(m => math.max(eps, m * (1.0 - m)))
      case Gaussian         => DenseVector.ones[Double](mu_i.length)
      case Poisson          => mu_i.map(m => math.max(eps, m))
      case NegativeBinomial => mu_i.map(m => math.max(eps, m + kappa * m * m))
      case Tweedie(p)       => mu_i.map(m => math.max(eps, phi * math.pow(m, p)))
    }

    val A_i = diag(A_diag)
    val A_sqrt = diag(A_diag.map(math.sqrt))
    val m_i = Y_i.length

    // val R = DenseMatrix.tabulate(m_i, m_i)((j, k) => if (j == k) 1.0 else rho)
    val V_i = A_sqrt * R * A_sqrt
    val V_i_inv = pinv(V_i)

    // compute D_i = A_i * X_i for canonical link functions, otherwise D_i = diag(d_mu/d_eta) * X_i
    val D_i = family match {
      case Tweedie(_) | NegativeBinomial => diag(mu_i) * X_i
      case _                             => A_i * X_i
    }
    val resid = Y_i - mu_i
    val U_i = D_i.t * V_i_inv * resid
    val B_i = D_i.t * V_i_inv * D_i
    (U_i, B_i)
  }
}

class GEE(
  corStruct: CorrelationStructure = Independent,
  family: DistributionFamily = Binomial)
    extends EstimatingEquation
    with Serializable {
  private var beta: DenseVector[Double] = _
  private var variance: DenseMatrix[Double] = _
  private var R: DenseMatrix[Double] = _
  private var p: Int = _ // number of covariates
  private var t: Int = _ // number of observations in the first cluster
  private var df: Dataset[Obs] = _

  // Dispersion parameters for NB and Tweedie
  private var kappa: Double = 0.0 // for Negative Binomial (0 = Poisson variance)
  private var phi: Double = 1.0 // for Tweedie

  // Default algo params
  private var warmupStepSize: Int = 5 // Update R and dispersion parameters every warmupStepSize iterations
  private var warmupRounds: Int = 2 // Warm-up rounds

  def fit(
    data: Dataset[Obs],
    warmupStepSize: Int,
    warmupRounds: Int,
    maxIter: Int,
    tol: Double,
    verbose: Boolean
  ): Boolean = {
    this.warmupStepSize = warmupStepSize
    this.warmupRounds = warmupRounds
    fit(data, maxIter, tol, verbose)
  }

  def fit(
    data: Dataset[Obs],
    maxIter: Int = 10,
    tol: Double = 1e-6,
    verbose: Boolean = true
  ): Boolean = {
    // intialized variables
    initialize(data)

    val eps = 1e-6
    var iter = 0
    var converged = false

    // Determine if we need to estimate dispersion parameters
    val needsDispersionEstimation = family match {
      case NegativeBinomial => true
      case Tweedie(_)       => true
      case _                => false
    }

    if (corStruct != Independent || needsDispersionEstimation) {

      // Warm-up iterations to estimate R and dispersion parameters
      while (iter < warmupStepSize * warmupRounds && !converged) {
        // Outer loop: Update R and dispersion parameters
        if (iter % warmupStepSize == 0) {
          if (corStruct != Independent) {
            if (verbose) println(s"Updating R at warm-up iteration $iter")
            estimateR() // Update R using the current beta
          }
          if (needsDispersionEstimation) {
            family match {
              case NegativeBinomial =>
                estimateKappa()
                if (verbose) println(s"Updated kappa = $kappa at warm-up iteration $iter")
              case Tweedie(p)       =>
                estimatePhi(p)
                if (verbose) println(s"Updated phi = $phi at warm-up iteration $iter")
              case _                => // do nothing
            }
          }
        }
        // Inner loop: Update beta using the current R and dispersion parameters
        converged = updateBeta(R, eps, tol)
        if (verbose) println(s"Updated beta = $beta at warm-up iteration $iter")
        iter += 1
      }

      // Final update of R and dispersion parameters after warm-up
      if (corStruct != Independent) {
        estimateR()
      }
      if (needsDispersionEstimation) {
        family match {
          case NegativeBinomial =>
            estimateKappa()
            if (verbose) println(s"Final warm-up kappa = $kappa")
          case Tweedie(p)       =>
            estimatePhi(p)
            if (verbose) println(s"Final warm-up phi = $phi")
          case _                => // do nothing
        }
      }
      if (verbose) println(s"Warm-up iterations completed: $iter, converged: $converged")
    }

    // Main iterations with updated R and dispersion parameters
    iter = 0
    converged = false
    while (iter < maxIter && !converged) {
      // Update beta using the current R and dispersion parameters
      converged = updateBeta(R, eps, tol, verbose = verbose)
      iter += 1
    }
    if (verbose) println(s"Main iterations completed: $iter, converged: $converged")

    if (!converged) {
      println(s"GEE did not converge after $maxIter iterations")
    }

    // Estimate robust variance
    val UBS = computeAggregatedStatsForVar(beta, R, eps)
    val invB = pinv(UBS._2)
    val delta = invB * UBS._1
    beta = beta + delta
    variance = invB * UBS._3 * invB.t

    converged
  }

  def result(): EESummary = {
    if (beta == null || variance == null) {
      throw new IllegalStateException("Model has not been fitted yet.")
    }
    EESummary(beta, variance)
  }

  def summary(): DataFrame = {
    if (beta == null || variance == null) {
      throw new IllegalStateException("Model has not been fitted yet.")
    }
    implicit val randBasis: RandBasis = RandBasis.mt0 // Default Mersenne Twister
    // Extract SparkSession from df
    val spark = df.sparkSession
    import spark.implicits._ // Use stable identifier for implicits

    // Compute standard errors
    val se = (0 until beta.length).map(i => math.sqrt(variance(i, i)))

    // Compute z-scores
    val zScores = beta.toArray.zip(se).map { case (coef, se) => coef / se }

    // Compute p-values
    val pValues =
      zScores.map(z => 2 * (1 - breeze.stats.distributions.Gaussian(0, 1).cdf(math.abs(z))))

    // Generate names
    val names = Seq("intercept") ++ (1 until beta.length).map(i => s"beta$i")

    // Create a DataFrame
    val result = names.zip(beta.toArray).zip(se).zip(zScores).zip(pValues).map {
      case ((((name, coef), se), z), p) =>
        (name, coef, se, z, p)
    }

    result.toDF("parameter", "estimate", "std_error", "z_score", "p_value")
  }

  /**
   * Get the estimated working correlation matrix.
   *
   * @return A copy of the estimated correlation matrix R
   * @throws IllegalStateException if the model has not been fitted yet
   */
  def getCorrelationMatrix(): DenseMatrix[Double] = {
    if (R == null) {
      throw new IllegalStateException("Model has not been fitted yet.")
    }
    R.copy // Return copy to prevent external modification
  }

  /**
   * Get the correlation structure used in the model.
   *
   * @return The correlation structure (Independent, Exchangeable, AR, or Unstructured)
   */
  def getCorrelationStructure(): CorrelationStructure = corStruct

  /**
   * Get the estimated working correlation matrix as a DataFrame for easy display.
   *
   * @param format "long" for long format (time_i, time_j, correlation),
   *               "square" for square/wide format (matrix-like display)
   * @return DataFrame with correlation matrix
   * @throws IllegalStateException if the model has not been fitted yet
   * @throws IllegalArgumentException if format is not "long" or "square"
   */
  def correlationSummary(format: String = "wide"): DataFrame = {
    if (R == null) {
      throw new IllegalStateException("Model has not been fitted yet.")
    }

    val spark = df.sparkSession
    import spark.implicits._

    format.toLowerCase match {
      case "long" =>
        // Long format: (time_i, time_j, correlation)
        val correlations = for {
          i <- 0 until R.rows
          j <- 0 until R.cols
        } yield (i, j, R(i, j))

        correlations
          .toDF("time_i", "time_j", "correlation")
          .orderBy("time_i", "time_j")

      case "square" | "wide" =>
        // Square format: matrix-like with columns for each time point
        val rows = (0 until R.rows).map { i =>
          val rowData = (0 until R.cols).map(j => R(i, j)).toSeq
          Seq(i.asInstanceOf[Any]) ++ rowData
        }

        // Create column names: time_point, time_0, time_1, time_2, ...
        val colNames = "time_point" +: (0 until R.cols).map(j => s"time_$j")

        // Convert to DataFrame
        import scala.collection.JavaConverters._
        spark.createDataFrame(
          rows.map(row => org.apache.spark.sql.Row(row: _*)).asJava,
          org.apache.spark.sql.types.StructType(
            colNames.zipWithIndex.map {
              case (name, idx) =>
                if (idx == 0)
                  org.apache.spark.sql.types
                    .StructField(name, org.apache.spark.sql.types.IntegerType, false)
                else
                  org.apache.spark.sql.types
                    .StructField(name, org.apache.spark.sql.types.DoubleType, false)
            }
          )
        )

      case _ =>
        throw new IllegalArgumentException(s"Invalid format: $format. Must be 'long' or 'square'.")
    }
  }

  private def initialize(data: Dataset[Obs], checkClusterSize: Boolean = false): Unit = {
    import data.sparkSession.implicits._ // Required for encoder
    if (data.isEmpty) {
      throw new IllegalArgumentException("Input dataset cannot be empty")
    }
    df = data.map(obs => obs.copy(x = Array(1.0) ++ obs.x)) // Add intercept term to covariates

    p = df.take(1).head.x.length

    if (checkClusterSize) {
      // Check if all clusters have the same size
      val clusterSizes = df.groupBy("i").count()
      val uniqueSizes = clusterSizes
        .select("count")
        .distinct()
        .collect()
        .map(row =>
          row.get(0) match {
            case i: Int  => i.toLong
            case l: Long => l
            case _       => throw new IllegalStateException("Unexpected type for count")
          }
        )
        .toSeq
      if (uniqueSizes.length > 1) {
        throw new IllegalArgumentException("All clusters must have the same size")
      }
      t = uniqueSizes.head.toInt
    } else {
      // number of observations in the first cluster (reserve for when checks drag down performance)
      val firstClusterId = df.select("i").limit(1).collect().head.getString(0)
      t = df.filter(_.i == firstClusterId).count().toInt
    }

    beta = DenseVector.zeros[Double](p)
    R = DenseMatrix.eye[Double](t)
    variance = DenseMatrix.eye[Double](p)
  }

  private def updateBeta(
    R: DenseMatrix[Double],
    eps: Double,
    tol: Double,
    verbose: Boolean = false
  ): Boolean = {
    val UB = computeAggregatedStats(beta, R, eps)
    val delta = pinv(UB._2) * UB._1
    beta = beta + delta
    if (verbose) {
      println(s"Iteration: $beta, ||delta|| = ${norm(delta)}")
    }
    val converged = norm(delta) < tol
    converged
  }

  private def estimateR(): Unit = {
    // check if beta and R are initialized
    if (beta == null) {
      throw new IllegalStateException("Beta must be initialized before estimating R")
    }
    // Compute the empirical correlation matrix R
    val covMatByCluster = df.rdd
      .groupBy(_.i)
      .map {
        case (_, obsSeq) =>
          val cluster = obsSeq.toSeq
          val X = DenseMatrix(cluster.map(_.x): _*)
          val Y = DenseVector(cluster.map(_.y): _*)

          val mu = family match {
            case Binomial         => sigmoid(X * beta)
            case Gaussian         => X * beta
            case Poisson          => breeze.numerics.exp(X * beta)
            case NegativeBinomial => breeze.numerics.exp(X * beta)
            case Tweedie(_)       => breeze.numerics.exp(X * beta)
          }
          val resi = Y - mu
          val covMat = resi * resi.t
          covMat.toArray
      }
    val aggCov = covMatByCluster.reduce((a, b) => a.zip(b).map { case (x, y) => x + y })
    val nClusters = covMatByCluster.count()
    val avgCovMat = new DenseMatrix(t, t, aggCov.map(_ / nClusters))

    val stddevs = (0 until t).map(i => math.sqrt(avgCovMat(i, i)))

    val corrMat = DenseMatrix.tabulate(t, t) {
      case (i, j) =>
        avgCovMat(i, j) / (stddevs(i) * stddevs(j))
    }

    // Estimate R based on corStruct
    R = corStruct match {
      case Independent =>
        DenseMatrix.eye[Double](t) // Identity matrix for Independent

      case Exchangeable =>
        val rhoHat_exchangeable = {
          val offDiags = for {
            i <- 0 until t
            j <- 0 until t if i != j
          } yield corrMat(i, j)
          offDiags.sum / offDiags.size
        }
        DenseMatrix.tabulate(t, t)((i, j) => if (i == j) 1.0 else rhoHat_exchangeable)

      case AR =>
        val rhoHat_ar1 = {
          val lags = for (i <- 0 until t - 1) yield corrMat(i, i + 1)
          lags.sum / lags.size
        }
        DenseMatrix.tabulate(t, t)((i, j) => math.pow(rhoHat_ar1, math.abs(i - j)))

      case Unstructured =>
        corrMat
    }
  }

  private def estimateKappa(): Unit = {
    // Estimate kappa for Negative Binomial using Pearson residuals
    // kappa is estimated from: Var(Y) = mu + kappa * mu^2
    // Using method of moments: kappa = sum(mu^2 * ((Y - mu)^2 - mu)) / sum(mu^4)
    if (beta == null) {
      throw new IllegalStateException("Beta must be initialized before estimating kappa")
    }

    val stats = df.rdd
      .groupBy(_.i)
      .flatMap {
        case (_, obsSeq) =>
          val cluster = obsSeq.toSeq
          val X = DenseMatrix(cluster.map(_.x): _*)
          val Y = DenseVector(cluster.map(_.y): _*)
          val mu = breeze.numerics.exp(X * beta)

          // Compute weighted components for kappa estimation
          cluster.indices.map { j =>
            val resid_sq = math.pow(Y(j) - mu(j), 2)
            val mu_sq = mu(j) * mu(j)
            val numerator = mu_sq * (resid_sq - mu(j)) // mu^2 * ((Y - mu)^2 - mu)
            val denominator = mu_sq * mu_sq // mu^4
            (numerator, denominator)
          }
      }
      .reduce {
        case ((num1, denom1), (num2, denom2)) =>
          (num1 + num2, denom1 + denom2)
      }

    val kappaHat = math.max(0.0, stats._1 / stats._2) // kappa must be non-negative
    kappa = kappaHat
  }

  private def estimatePhi(p: Double): Unit = {
    // Estimate phi for Tweedie distribution
    // phi is estimated from: Var(Y) = phi * mu^p
    // Using method of moments: phi = mean((Y - mu)^2 / mu^p)
    if (beta == null) {
      throw new IllegalStateException("Beta must be initialized before estimating phi")
    }

    val stats = df.rdd
      .groupBy(_.i)
      .flatMap {
        case (_, obsSeq) =>
          val cluster = obsSeq.toSeq
          val X = DenseMatrix(cluster.map(_.x): _*)
          val Y = DenseVector(cluster.map(_.y): _*)
          val mu = breeze.numerics.exp(X * beta)

          // Compute (Y - mu)^2 / mu^p for each observation
          cluster.indices.map { j =>
            val resid_sq = math.pow(Y(j) - mu(j), 2)
            val mu_p = math.pow(mu(j), p)
            (resid_sq / mu_p, 1.0) // ratio and count
          }
      }
      .reduce {
        case ((sum1, count1), (sum2, count2)) =>
          (sum1 + sum2, count1 + count2)
      }

    val phiHat = math.max(1e-6, stats._1 / stats._2) // phi = mean of ratios, must be positive
    phi = phiHat
  }

  private def computeAggregatedStats(
    beta: DenseVector[Double],
    R: DenseMatrix[Double],
    eps: Double
  ): (DenseVector[Double], DenseMatrix[Double]) = {
    val statsRdd = df.rdd
      .groupBy(_.i)
      .map {
        case (_, cluster) =>
          val aggUB = GEEUtils.computeClusterStats(cluster.toSeq, beta, R, eps, family, kappa, phi)
          (aggUB._1.toArray, aggUB._2.toArray)
      }

    val aggUBSum = statsRdd.reduce {
      case ((u1, b1), (u2, b2)) =>
        val u = u1.zip(u2).map { case (a, b) => a + b }
        val b = b1.zip(b2).map { case (a, b) => a + b }
        (u, b)
    }

    val U = new DenseVector(aggUBSum._1)
    val B = new DenseMatrix(p, p, aggUBSum._2)
    (U, B)
  }

  private def computeAggregatedStatsForVar(
    beta: DenseVector[Double],
    R: DenseMatrix[Double],
    eps: Double
  ): (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    val statsRdd = df.rdd
      .groupBy(_.i)
      .map {
        case (_, cluster) =>
          val aggUB = GEEUtils.computeClusterStats(cluster.toSeq, beta, R, eps, family, kappa, phi)
          val uMat = aggUB._1 * aggUB._1.t
          (aggUB._1.toArray, aggUB._2.toArray, uMat.toArray)
      }

    val aggUBSSum = statsRdd.reduce {
      case ((u1, b1, s1), (u2, b2, s2)) =>
        val u = u1.zip(u2).map { case (a, b) => a + b }
        val b = b1.zip(b2).map { case (a, b) => a + b }
        val s = s1.zip(s2).map { case (a, b) => a + b }
        (u, b, s)
    }

    val U = new DenseVector(aggUBSSum._1)
    val B = new DenseMatrix(p, p, aggUBSSum._2)
    val S = new DenseMatrix(p, p, aggUBSSum._3)
    (U, B, S)
  }
}
