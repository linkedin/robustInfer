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
    family: DistributionFamily = Binomial
  ): (DenseVector[Double], DenseMatrix[Double]) = {
    val X_i = DenseMatrix(cluster.map(_.x): _*)
    val Y_i = DenseVector(cluster.map(_.y): _*)

    val mu_i = family match {
      case Binomial => sigmoid(X_i * beta)
      case Gaussian => X_i * beta
      case Poisson  => breeze.numerics.exp(X_i * beta)
    }

    val A_diag = family match {
      case Binomial => mu_i.map(m => math.max(eps, m * (1.0 - m)))
      case Gaussian => DenseVector.ones[Double](mu_i.length)
      case Poisson  => mu_i.map(m => math.max(eps, m))
    }

    val A_i = diag(A_diag)
    val A_sqrt = diag(A_diag.map(math.sqrt))
    val m_i = Y_i.length

    // val R = DenseMatrix.tabulate(m_i, m_i)((j, k) => if (j == k) 1.0 else rho)
    val V_i = A_sqrt * R * A_sqrt
    val V_i_inv = pinv(V_i)

    val D_i = A_i * X_i
    val resid = Y_i - mu_i
    val U_i = D_i.t * V_i_inv * resid
    val B_i = D_i.t * V_i_inv * D_i
    (U_i, B_i)
  }
}

class GEE(
  corStruct: CorrelationStructure = Independent,
  family: DistributionFamily = Binomial) extends EstimatingEquation with Serializable {
  private var beta: DenseVector[Double] = _
  private var variance: DenseMatrix[Double] = _
  private var R: DenseMatrix[Double] = _
  private var p: Int = _ // number of covariates
  private var t: Int = _ // number of observations in the first cluster
  private var df: Dataset[Obs] = _

  def fit(data: Dataset[Obs], maxIter: Int = 10, tol: Double = 1e-6, verbose: Boolean = true): Unit = {
    // intialized variables
    initialize(data)

    val eps = 1e-6
    var iter = 0
    var converged = false
    if (corStruct != Independent) {
      val stepR = 5 // Update R every iteration, can be adjusted
    
      // Warm-up iterations to estimate R
      while (iter < stepR*2 && !converged) {
        // Outer loop: Update R
        if (iter % stepR == 0) {
          if (verbose) println(s"Updating R at warm-up iteration $iter")
          estimateR() // Update R using the current beta
        }
        // Inner loop: Update beta using the current R
        converged = updateBeta(R, eps, tol)
        iter += 1
      }
      estimateR()
      if (verbose) println(s"Warm-up iterations completed: $iter, converged: $converged")
    }

    // Main iterations with updated R
    iter = 0
    converged = false
    while (iter < maxIter && !converged) {
      // Update beta using the current R
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
  }

  def summary(): EESummary = {
    if (beta == null || variance == null) {
      throw new IllegalStateException("Model has not been fitted yet.")
    }
    EESummary(beta, variance)
  }

  def dfSummary(): DataFrame = {
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
    val pValues = zScores.map(z => 2 * (1 - breeze.stats.distributions.Gaussian(0, 1).cdf(math.abs(z))))

    // Generate names
    val names = Seq("intercept") ++ (1 until beta.length).map(i => s"beta$i")

    // Create a DataFrame
    val result = names.zip(beta.toArray).zip(se).zip(zScores).zip(pValues).map {
      case ((((name, coef), se), z), p) =>
        (name, coef, se, z, p)
    }

    result.toDF("names", "coef", "se", "z", "p-value")
  }

  private def initialize(data: Dataset[Obs], checkClusterSize: Boolean = true): Unit = {
    import data.sparkSession.implicits._  // Required for encoder
    if (data.isEmpty) {
      throw new IllegalArgumentException("Input dataset cannot be empty")
    }
    df = data.map(obs => obs.copy(x = Array(1.0) ++ obs.x)) // Add intercept term to covariates

    p = df.take(1).head.x.length

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

    beta = DenseVector.zeros[Double](p)
    R = DenseMatrix.eye[Double](t) 
    variance = DenseMatrix.eye[Double](p)
  }

  private def updateBeta(
    R: DenseMatrix[Double], 
    eps: Double, 
    tol: Double, verbose: Boolean = false): Boolean = {
    val UB = computeAggregatedStats(beta, R, eps)
    val delta = pinv(UB._2) * UB._1
    beta = beta + delta
    if (verbose) {
      println(s"Iteration: ${beta}, ||delta|| = ${norm(delta)}")
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
      .map { case (_, obsSeq) =>
        val cluster = obsSeq.toSeq
        val X = DenseMatrix(cluster.map(_.x): _*)
        val Y = DenseVector(cluster.map(_.y): _*)
        
        val mu = family match {
          case Binomial => sigmoid(X * beta)
          case Gaussian => X * beta
          case Poisson  => breeze.numerics.exp(X * beta)
        }
        val resi = Y - mu
        val covMat = resi * resi.t
        covMat.toArray
      }
    val aggCov = covMatByCluster.reduce((a, b) => a.zip(b).map { case (x, y) => x + y })
    val nClusters = covMatByCluster.count()
    val avgCovMat = new DenseMatrix(t, t, aggCov.map(_ / nClusters))

    val stddevs = (0 until t).map(i => math.sqrt(avgCovMat(i, i)))

    val corrMat = DenseMatrix.tabulate(t, t) { case (i, j) =>
      avgCovMat(i, j) / (stddevs(i) * stddevs(j))
    }

    // Estimate R based on corStruct
    corStruct match {
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
        DenseMatrix.tabulate(t, t) { (i, j) =>
          if (i == j) 1.0 else rhoHat_exchangeable
        }

      case AR =>
        val rhoHat_ar1 = {
          val lags = for (i <- 0 until t - 1) yield corrMat(i, i + 1)
          lags.sum / lags.size
        }
        DenseMatrix.tabulate(t, t) { (i, j) =>
          math.pow(rhoHat_ar1, math.abs(i - j))
        }

      case Unstructured =>
        corrMat
    }
  }

  private def computeAggregatedStats(
    beta: DenseVector[Double],
    R: DenseMatrix[Double],
    eps: Double,
  ): (DenseVector[Double], DenseMatrix[Double]) = {
    val statsRdd = df.rdd
      .groupBy(_.i)
      .map { case (_, cluster) =>
        val aggUB = GEEUtils.computeClusterStats(cluster.toSeq, beta, R, eps, family)
        (aggUB._1.toArray, aggUB._2.toArray)
      }

    val aggUBSum = statsRdd.reduce { case ((u1, b1), (u2, b2)) =>
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
    eps: Double,
  ): (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
    val statsRdd = df.rdd
      .groupBy(_.i)
      .map { case (_, cluster) =>
        val aggUB = GEEUtils.computeClusterStats(cluster.toSeq, beta, R, eps, family)
        val uMat = aggUB._1 * aggUB._1.t
        (aggUB._1.toArray, aggUB._2.toArray, uMat.toArray)
      }

    val aggUBSSum = statsRdd.reduce { case ((u1, b1, s1), (u2, b2, s2)) =>
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
