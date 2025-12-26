package robustinfer

import org.apache.spark.sql.Dataset
import breeze.linalg._
import breeze.linalg.eigSym.EigSym
import breeze.numerics._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import scala.util.Random

case class ObsPair(left: Obs, right: Obs)

case class PairFeatures(
  Wt_i: Array[Double],
  Wt_j: Array[Double],
  Xg_ij: Array[Double],
  Xg_ji: Array[Double],
  yi: Double,
  yj: Double,
  zi: Double,
  zj: Double,
  weight: Double = 1.0, // Horvitz-Thompson weight for unbiased U-statistics
  anchorId: Option[Int] = None // Anchor identifier for variance estimation
) extends Serializable

object UGEEUtils {

  /**
   * Deterministic hash-based partitioning for balanced data distribution.
   * Uses xxhash64 to assign observations to partitions based on cluster ID.
   * This ensures reproducible partitioning and avoids data shuffling during sampling.
   */
  def assignDeterministicPartition(
    ds: Dataset[Obs],
    idColumn: String = "i", // cluster ID or any stable ID
    numPartitions: Int
  ): Dataset[(Obs, Int)] = {
    import ds.sparkSession.implicits._
    import org.apache.spark.sql.functions._

    val withBucket = ds.withColumn("bucket", (xxhash64(col(idColumn)) % numPartitions).cast("int"))
    val repartitioned = withBucket.repartition($"bucket")
    repartitioned
      .select(
        struct($"i", $"x", $"y", $"timeIndex", $"z").as("obs"),
        $"bucket"
      )
      .as[(Obs, Int)]
  }

  /**
   * K-partners sampling within partitions with Horvitz-Thompson weights.
   * For each observation i, samples k distinct partners j != i within the same partition.
   * Applies proper HT weighting for unbiased U-statistic estimation.
   */
  def sampleKPartnersWithinPartitions(
    df: Dataset[Obs],
    k: Int,
    numPartitions: Int = -1, // Auto-determine if -1
    seed: Long = 42
  ): Dataset[PairFeatures] = {
    import df.sparkSession.implicits._

    // Auto-determine partitions: ~1000 observations per partition
    val totalCount = df.count()
    val partitions =
      if (numPartitions > 0) numPartitions
      else math.max(1, (totalCount / 1000).toInt)

    // Use deterministic partitioning
    val repartitioned = assignDeterministicPartition(df, "i", partitions).cache()

    // Sample k partners for each observation within partitions
    val pairs = repartitioned
      .groupByKey { case (_, bucket) => bucket }
      .flatMapGroups {
        case (bucketId, iter) =>
          val rng = new Random(seed ^ bucketId)
          val items = iter.map(_._1).toIndexedSeq
          val n_local = items.length

          if (n_local < 2) Iterator.empty
          else {
            // Compute HT weight: accounts for partition-local sampling
            // w_ij = 2 * (n_local - 1) / k for within-partition pairs
            // Factor of 2 because we only keep one direction (i < j)
            val htWeight = 2.0 * (n_local - 1) / k.toDouble

            // For each observation i, sample k partners
            (0 until n_local).flatMap { i =>
              val candidates = (0 until n_local).filter(_ != i)
              val k_actual = math.min(k, candidates.length)
              val partners = rng.shuffle(candidates.toVector).take(k_actual)

              partners.flatMap { j =>
                // Only keep one direction to avoid double counting
                if (i < j) {
                  val pf = toPairFeatures(ObsPair(items(i), items(j)))
                  Some(pf.copy(weight = htWeight))
                } else None
              }
            }
          }
      }

    // Clean up cached repartitioned data
    repartitioned.unpersist()
    pairs
  }

  /**
   * Anchor-based sampling within partitions for variance estimation.
   * Samples s_total anchors across all partitions, with m partners per anchor.
   * Each pair is tagged with anchorId for within-anchor and between-anchor variance computation.
   */
  def sampleAnchorsWithinPartitions(
    df: Dataset[Obs],
    s_total: Int, // total anchors across all partitions
    m: Int, // partners per anchor
    numPartitions: Int = -1,
    seed: Long = 42
  ): Dataset[PairFeatures] = {
    import df.sparkSession.implicits._

    // Auto-determine partitions: aim for ~10 anchors per partition
    val partitions =
      if (numPartitions > 0) numPartitions
      else math.max(1, s_total / 10)

    // Use deterministic partitioning
    val repartitioned = assignDeterministicPartition(df, "i", partitions).cache()
    val s_per_partition = math.max(1, s_total / partitions)

    val pairs = repartitioned
      .groupByKey { case (_, bucket) => bucket }
      .flatMapGroups {
        case (bucketId, iter) =>
          val rng = new Random(seed ^ bucketId)
          val items = iter.map(_._1).toIndexedSeq
          val n_local = items.length

          if (n_local < 2) Iterator.empty
          else {
            // Sample s_per_partition anchors from this partition
            val anchorIndices = rng.shuffle(items.indices.toVector).take(s_per_partition)

            anchorIndices.zipWithIndex.flatMap {
              case (anchorIdx, localAnchorId) =>
                // Global anchor ID: unique across all partitions
                val globalAnchorId = bucketId * s_per_partition + localAnchorId

                // Sample m partners for this anchor
                val candidates = items.indices.filter(_ != anchorIdx)
                val m_actual = math.min(m, candidates.length)
                val partners = rng.shuffle(candidates.toVector).take(m_actual)

                partners.map { partnerIdx =>
                  val pf = toPairFeatures(ObsPair(items(anchorIdx), items(partnerIdx)))
                  pf.copy(
                    weight = 1.0, // No HT weighting for anchor-based (design-based variance)
                    anchorId = Some(globalAnchorId)
                  )
                }
            }
          }
      }

    // Clean up cached repartitioned data
    repartitioned.unpersist()
    pairs
  }

  def sampleAllPairs(obsDS: Dataset[Obs]): Dataset[ObsPair] = {
    val spark = obsDS.sparkSession
    import spark.implicits._

    // Add index to each observation
    val indexed: RDD[(Obs, Long)] = obsDS.rdd.zipWithIndex()

    // Cartesian join and filter only (i < j) to avoid duplicates
    val allPairs: RDD[ObsPair] = indexed
      .cartesian(indexed)
      .filter { case ((_, i), (_, j)) => i < j }
      .map { case ((obs1, _), (obs2, _)) => ObsPair(obs1, obs2) }

    // Convert back to Dataset
    spark.createDataset(allPairs)
  }

  def toPairFeatures(pair: ObsPair): PairFeatures = {
    val Obs(_, x1, y1, _, Some(z1)) = pair.left
    val Obs(_, x2, y2, _, Some(z2)) = pair.right

    val Wt_i = Array(1.0) ++ x1
    val Wt_j = Array(1.0) ++ x2
    val Xg_ij = Array(1.0) ++ x1 ++ x2
    val Xg_ji = Array(1.0) ++ x2 ++ x1

    PairFeatures(Wt_i, Wt_j, Xg_ij, Xg_ji, y1, y2, z1, z2)
  }

  def safeSig(x: Double): Double = {
    val clipped = math.max(-15.0, math.min(15.0, x))
    1.0 / (1.0 + math.exp(-clipped))
  }

  def computeHFDFDHforPair(
    p: PairFeatures,
    theta: Map[String, DenseVector[Double]]
  ): (DenseVector[Double], DenseVector[Double], DenseVector[Double], DenseVector[Double]) = {
    val delta = theta("delta")(0)
    val beta = theta("beta")
    val gamma = theta("gamma")

    val Wt_i = DenseVector(p.Wt_i)
    val Wt_j = DenseVector(p.Wt_j)
    val Xg_ij = DenseVector(p.Xg_ij)
    val Xg_ji = DenseVector(p.Xg_ji)

    // ---- sigmoid preds ----
    val pi_i = safeSig(Wt_i dot beta)
    val pi_j = safeSig(Wt_j dot beta)
    val g_ij = safeSig(Xg_ij dot gamma)
    val g_ji = safeSig(Xg_ji dot gamma)

    val I_ij = if (p.yi >= p.yj) 1.0 else 0.0
    val I_ji = 1.0 - I_ij

    // ---- compute h ----
    val A = p.zi * (1 - p.zj) / (2 * pi_i * (1 - pi_j))
    val B = p.zj * (1 - p.zi) / (2 * pi_j * (1 - pi_i))
    val num1 = A * (I_ij - g_ij)
    val num2 = B * (I_ji - g_ji)
    val h1 = num1 + num2 + 0.5 * (g_ij + g_ji) - 0.5 // shifted, so delta = 0.0 under null hypothesis
    val h2 = 0.5 * (p.zi + p.zj)
    val h3 = 0.5 * (p.zi * (1 - p.zj) * I_ij + p.zj * (1 - p.zi) * I_ji)
    val h = DenseVector(h1, h2, h3)

    // ---- compute f ----
    val f1 = delta // shifted delta = 0.0 under null hypothesis
    val f2 = 0.5 * (pi_i + pi_j)
    val f3 = 0.5 * (pi_i * (1 - pi_j) * g_ij + pi_j * (1 - pi_i) * g_ji)
    val f = DenseVector(f1, f2, f3)

    // ---- gradient df ----
    // df1
    val df1Delta = 1.0

    // df2
    val dPiI = pi_i * (1 - pi_i) * Wt_i
    val dPiJ = pi_j * (1 - pi_j) * Wt_j
    val df2Beta = 0.5 * (dPiI + dPiJ)

    // df3
    val df3Beta = 0.5 * (
      ((1 - pi_j) * pi_i * (1 - pi_i) * Wt_i - pi_i * pi_j * (1 - pi_j) * Wt_j) * g_ij +
        ((1 - pi_i) * pi_j * (1 - pi_j) * Wt_j - pi_j * pi_i * (1 - pi_i) * Wt_i) * g_ji
    )
    val df3Gamma = 0.5 * (
      pi_i * (1 - pi_j) * g_ij * (1 - g_ij) * Xg_ij +
        pi_j * (1 - pi_i) * g_ji * (1 - g_ji) * Xg_ji
    )

    val pb = beta.length
    val qg = gamma.length
    val df = DenseMatrix.vertcat(
      DenseVector(1.0 +: Array.fill(pb + qg)(0.0): _*).toDenseMatrix,
      DenseVector.vertcat(DenseVector(0.0), df2Beta, DenseVector.zeros[Double](qg)).toDenseMatrix,
      DenseVector.vertcat(DenseVector(0.0), df3Beta, df3Gamma).toDenseMatrix
    )

    // ---- gradient dh ----
    // dh1 / dβ
    val dh1Beta = -(I_ij - g_ij) * A * ((1.0 - pi_i) * Wt_i - pi_j * Wt_j) -
      (I_ji - g_ji) * B * ((1.0 - pi_j) * Wt_j - pi_i * Wt_i)
    // dh1 / dγ
    val dh1Gamma = -A * (I_ij - g_ij) * g_ij * (1 - g_ij) * Xg_ij -
      B * (I_ji - g_ji) * g_ji * (1 - g_ji) * Xg_ji +
      0.5 * (g_ij * (1 - g_ij) * Xg_ij + g_ji * (1 - g_ji) * Xg_ji)
    // dh2, dh3 are zero
    val dh2Beta = DenseVector.zeros[Double](pb)
    val dh2Gamma = DenseVector.zeros[Double](qg)
    val dh3Beta = DenseVector.zeros[Double](pb)
    val dh3Gamma = DenseVector.zeros[Double](qg)

    // assemble dh matrix
    val dh = DenseMatrix.vertcat(
      DenseVector.vertcat(DenseVector(0.0), dh1Beta, dh1Gamma).toDenseMatrix,
      DenseVector.vertcat(DenseVector(0.0), dh2Beta, dh2Gamma).toDenseMatrix,
      DenseVector.vertcat(DenseVector(0.0), dh3Beta, dh3Gamma).toDenseMatrix
    )

    // return (h, f, df, dh)
    (h, f, df.toDenseVector, dh.toDenseVector)
  }

  def computeBUforPair(
    pf: PairFeatures,
    theta: Map[String, DenseVector[Double]]
  ): (DenseVector[Double], DenseVector[Double]) = {
    // 1) get h, f, D, M
    val (h, f, dfVec, dhVec) = computeHFDFDHforPair(pf, theta)
    val d = dfVec.length / 3
    val D = new DenseMatrix(3, d, dfVec.data) // 3×d
    val M = new DenseMatrix(3, d, dhVec.data) // 3×d
    val Vinv = DenseMatrix.eye[Double](3) // 3x3, identity matrix for simplicity

    // 2) G = D^T * V⁻¹  (d×3)
    val G = D.t * Vinv

    // 3) B_i = G * (D - M)   (d×d)
    val B_i = G * (D - M)

    // 4) u_i = G * (h - f)   (d)
    val u_i = G * (h - f)

    (B_i.toDenseVector, u_i)
  }

  def updateTheta(
    theta: Map[String, DenseVector[Double]],
    step: DenseVector[Double]
  ): Map[String, DenseVector[Double]] = {
    var idx = 0
    // delta (scalar)
    val newDelta = theta("delta") + step(idx to idx)
    idx += 1
    // beta
    val bLen = theta("beta").length
    val newBeta = theta("beta") + step(idx until idx + bLen)
    idx += bLen
    // gamma
    val gLen = theta("gamma").length
    val newGamma = theta("gamma") + step(idx until idx + gLen)

    Map(
      "delta" -> newDelta,
      "beta" -> newBeta,
      "gamma" -> newGamma
    )
  }

  def generalizedInverse(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    val svd.SVD(u, s, vt) = svd(matrix)
    val sInv = DenseMatrix.zeros[Double](s.length, s.length)

    // Invert non-zero singular values
    for (i <- 0 until s.length)
      if (s(i) > 1e-10) { // Threshold to avoid division by zero
        sInv(i, i) = 1.0 / s(i)
      }

    // Compute the pseudoinverse
    vt.t * sInv * u.t
  }

  def computeStep(
    pairFeatureDS: Dataset[PairFeatures],
    theta: Map[String, DenseVector[Double]],
    lambda: Double,
    dampingOnly: Boolean = false
  ): DenseVector[Double] = {
    val d = theta("beta").length + theta("gamma").length + 1 // total params

    val BURDD: RDD[(DenseVector[Double], DenseVector[Double])] =
      pairFeatureDS.rdd.mapPartitions { iter =>
        iter.grouped(200).flatMap(batch => batch.map(p => computeBUforPair(p, theta)))
      }

    val BUsum = BURDD.aggregate(
      (DenseVector.zeros[Double](d * d), DenseVector.zeros[Double](d), 0)
    )(
      (acc, value) => {
        val BAcc = acc._1
        val UAcc = acc._2
        val c = acc._3
        val B = value._1
        val U = value._2
        (BAcc + B, UAcc + U, c + 1)
      },
      (acc1, acc2) => {
        val B1 = acc1._1
        val U1 = acc1._2
        val c1 = acc1._3
        val B2 = acc2._1
        val U2 = acc2._2
        val c2 = acc2._3
        (B1 + B2, U1 + U2, c1 + c2)
      }
    )

    val Btot = new DenseMatrix(d, d, BUsum._1.data) // d×d
    val Utot = BUsum._2 // d
    val countBU = BUsum._3

    val Bmean = Btot / countBU.toDouble
    val Umean = Utot / countBU.toDouble

    // Fisher scoring update

    val J = -Bmean
    // Assuming penalty term is quatratic form of thetavector * Penalty * thetavector
    val Penalty = DenseMatrix.eye[Double](d) // Assuming simple penalty of Identity matrix of size d
    // don't penalize delta
    // TODO: make this configurable
    Penalty(0, 0) = 0.0
    val thetavector = DenseVector.vertcat(
      theta("delta"),
      theta("beta"),
      theta("gamma")
    )
    if (dampingOnly) {
      val step = generalizedInverse(J - lambda * Penalty) * (-Umean)
      return step
    }
    val step = generalizedInverse(J - lambda * Penalty) * (-Umean + lambda * Penalty * thetavector)

    step
  }

  def computeStepWithSig(
    pairFeatureDS: Dataset[PairFeatures],
    theta: Map[String, DenseVector[Double]],
    lambda: Double = 0.0,
    dampingOnly: Boolean = false
  ): (DenseVector[Double], DenseMatrix[Double]) = {
    val d = theta("beta").length + theta("gamma").length + 1 // total params

    val BURDD: RDD[(DenseVector[Double], DenseVector[Double])] =
      pairFeatureDS.rdd.mapPartitions { iter =>
        iter.grouped(200).flatMap(batch => batch.map(p => computeBUforPair(p, theta)))
      }

    val BUsumWithSig = BURDD.aggregate(
      (
        DenseVector.zeros[Double](d * d),
        DenseVector.zeros[Double](d),
        DenseVector.zeros[Double](d * d),
        0
      )
    )(
      (acc, value) => {
        val BAcc = acc._1
        val UAcc = acc._2
        val SAcc = acc._3
        val c = acc._4
        val B = value._1
        val U = value._2
        val S = (U * U.t).toDenseVector
        (BAcc + B, UAcc + U, SAcc + S, c + 1)
      },
      (acc1, acc2) => {
        val B1 = acc1._1
        val U1 = acc1._2
        val S1 = acc1._3
        val c1 = acc1._4
        val B2 = acc2._1
        val U2 = acc2._2
        val S2 = acc2._3
        val c2 = acc2._4
        (B1 + B2, U1 + U2, S1 + S2, c1 + c2)
      }
    )

    val Btot = new DenseMatrix(d, d, BUsumWithSig._1.data) // d×d
    val Stot = new DenseMatrix(d, d, BUsumWithSig._3.data) // d×d
    val Utot = BUsumWithSig._2 // d
    val countBU = BUsumWithSig._4

    val Bmean = Btot / countBU.toDouble
    val Umean = Utot / countBU.toDouble
    val Smean = Stot / countBU.toDouble

    // Fisher scoring update
    val J = -Bmean
    // Assuming penalty term is quatratic form of thetavector * Penalty * thetavector
    val Penalty = DenseMatrix.eye[Double](d) // Assuming simple penalty of Identity matrix of size d
    // don't penalize delta
    // TODO: make this configurable
    Penalty(0, 0) = 0.0
    val thetavector = DenseVector.vertcat(
      theta("delta"),
      theta("beta"),
      theta("gamma")
    )
    if (dampingOnly) {
      val step = generalizedInverse(J - lambda * Penalty) * (-Umean)
      val Var = 4.0 * generalizedInverse(Bmean).t * Smean * generalizedInverse(Bmean)
      return (step, Var)
    }
    val step = generalizedInverse(J - lambda * Penalty) * (-Umean + lambda * Penalty * thetavector)
    // variance calculation
    val Var = 4.0 * generalizedInverse(Bmean + lambda * Penalty).t * Smean * generalizedInverse(
      Bmean + lambda * Penalty
    )

    (step, Var)
  }

  def generateData(
    numClusters: Int,
    numObsPerCluster: Int,
    p: Int,
    etaTrue: Array[Double],
    betaTrue: Array[Double]
  )(implicit spark: SparkSession
  ): Dataset[Obs] = {
    import spark.implicits._
    val random = new Random()
    val data = (1 to numClusters).flatMap { clusterId =>
      (1 to numObsPerCluster).map { obsId =>
        val w = Array.fill(p)(random.nextDouble()) // p covariates
        val Wt = 1.0 +: w // Add intercept
        val piTrue = 1.0 / (1.0 + math.exp(-(Wt zip etaTrue).map { case (w, e) => w * e }.sum))
        val z = if (random.nextDouble() < piTrue) 1.0 else 0.0
        val X = 1.0 +: z +: w
        val error = random.nextGaussian()
        val y = (X zip betaTrue).map { case (x, b) => x * b }.sum + error

        Obs(
          i = s"c$clusterId",
          x = w,
          y = y,
          timeIndex = Some(obsId),
          z = Some(z)
        )
      }
    }

    data.toDS()
  }

  /**
   * Process a batch of pairs to compute B and U vectors (simplified, no HT weights).
   * Returns vectors for consistency with computeBUforPair - convert to matrices when needed.
   * This function processes multiple pairs at once for efficiency.
   */
  def computeBatchBU(
    batch: Array[PairFeatures],
    theta: Map[String, DenseVector[Double]]
  ): (DenseVector[Double], DenseVector[Double]) = {
    if (batch.isEmpty) {
      val d = theta("beta").length + theta("gamma").length + 1
      return (DenseVector.zeros[Double](d * d), DenseVector.zeros[Double](d))
    }

    // Process each pair and collect B_i, u_i
    val results = batch.map(pf => computeBUforPair(pf, theta))
    val B_vecs = results.map(_._1)
    val u_vecs = results.map(_._2)

    // Compute simple averages directly on vectors (more efficient)
    val B_avg_vec = B_vecs.reduce(_ + _) / batch.length.toDouble
    val u_avg = u_vecs.reduce(_ + _) / batch.length.toDouble

    (B_avg_vec, u_avg)
  }

  /**
   * Aggregate B and U vectors across multiple batches.
   * Simple averaging across batches (no complex weighting).
   * Returns vectors for consistency - convert to matrices when needed for operations.
   */
  def aggregateBatchStats(
    batchResults: Array[(DenseVector[Double], DenseVector[Double])]
  ): (DenseVector[Double], DenseVector[Double]) = {
    if (batchResults.isEmpty) {
      throw new IllegalArgumentException("Cannot aggregate empty batch results")
    }

    val B_vecs = batchResults.map(_._1)
    val u_vecs = batchResults.map(_._2)

    // Simple averaging across all batches
    val B_aggregated = B_vecs.reduce(_ + _) / batchResults.length.toDouble
    val u_aggregated = u_vecs.reduce(_ + _) / batchResults.length.toDouble

    (B_aggregated, u_aggregated)
  }

  /**
   * Apply exponential moving average for numerical stability.
   * Helps smooth out mini-batch noise during optimization.
   * Special case: alpha = 0 means no smoothing (use newBatch directly)
   */
  def applyEMA(
    current: DenseMatrix[Double],
    newBatch: DenseMatrix[Double],
    alpha: Double
  ): DenseMatrix[Double] = {
    require(
      alpha >= 0.0 && alpha < 1.0,
      "EMA alpha must be in [0,1) - alpha=0 means no smoothing, alpha=1 would ignore new batches"
    )
    alpha * current + (1.0 - alpha) * newBatch
  }

  /**
   * Apply exponential moving average for vectors.
   * Special case: alpha = 0 means no smoothing (use newBatch directly)
   */
  def applyEMA(
    current: DenseVector[Double],
    newBatch: DenseVector[Double],
    alpha: Double
  ): DenseVector[Double] = {
    require(
      alpha >= 0.0 && alpha < 1.0,
      "EMA alpha must be in [0,1) - alpha=0 means no smoothing, alpha=1 would ignore new batches"
    )
    alpha * current + (1.0 - alpha) * newBatch
  }

  /**
   * Compute anchor-based Monte Carlo variance estimation.
   *
   * Implements Algorithm 1 from the paper using anchor-based sampling
   * for unbiased U-statistic variance estimation.
   *
   * @param data Dataset of observations
   * @param theta Current parameter estimates
   * @param s_total Total number of anchors across all partitions
   * @param m Number of partners per anchor
   * @param alpha Debiasing parameter in [0,1] (0 = no debiasing)
   * @param seed Random seed for reproducible sampling
   * @return Variance matrix for sandwich estimator
   */
  def computeMonteCarloVariance(
    data: Dataset[Obs],
    theta: Map[String, DenseVector[Double]],
    s_total: Int = 100,
    m: Int = 10,
    alpha: Double = 0.0,
    lambda: Double = 1e-4, // L2 regularization
    penalty: Option[DenseMatrix[Double]] = None,
    numPartitions: Int = -1,
    seed: Int = 42
  ): DenseMatrix[Double] = {

    // Step 1: Generate anchor pairs using partition-local sampling
    val pairs =
      sampleAnchorsWithinPartitions(data, s_total, m, seed = seed, numPartitions = numPartitions)

    // Step 2: Compute per-anchor statistics
    val anchorStatsRDD = pairs.rdd.mapPartitions { iter =>
      // Group by anchor ID within each partition
      val pairsByAnchor = iter
        .filter(_.anchorId.isDefined)
        .toArray
        .groupBy(_.anchorId.get)

      // Compute statistics for each anchor
      pairsByAnchor.map {
        case (anchorId, anchorPairs) =>
          computeAnchorStatistics(anchorPairs, theta)
      }.toIterator
    }

    // Step 3: Get intermediate variance components
    val n = data.count().toInt
    val (sum_u_outer, sum_sigma_within, sum_b_avg) =
      computeVarianceComponents(anchorStatsRDD, alpha, m, n, s_total, lambda)

    // Step 4: Compute final variance using the common function
    computeFinalVariance(
      sum_u_outer,
      sum_sigma_within,
      sum_b_avg,
      s_total,
      alpha,
      m,
      n,
      lambda,
      penalty
    )
  }

  /**
   * Compute statistics for a single anchor: mean U-statistic, within-anchor covariance, and average B vector.
   *
   * @param anchorPairs All pairs for this anchor (length = m)
   * @param theta Current parameter estimates
   * @return (u_bar_i, Sigma_i_within, B_avg_i) - anchor mean, within-anchor covariance, and average B vector
   */
  def computeAnchorStatistics(
    anchorPairs: Array[PairFeatures],
    theta: Map[String, DenseVector[Double]]
  ): (DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
    if (anchorPairs.isEmpty) {
      val d = theta("beta").length + theta("gamma").length + 1
      return (
        DenseVector.zeros[Double](d),
        DenseMatrix.zeros[Double](d, d),
        DenseVector.zeros[Double](d * d)
      )
    }

    // Compute B and U statistics for all pairs with this anchor
    val buResults = anchorPairs.map(pf => computeBUforPair(pf, theta))
    val bVectors = buResults.map(_._1)
    val uVectors = buResults.map(_._2)
    val m = uVectors.length

    // Anchor mean: u_bar_i = (1/m) * sum(u_ij)
    val u_bar = uVectors.reduce(_ + _) / m.toDouble

    // Average B vector for this anchor: B_avg_i = (1/m) * sum(B_ij)
    val b_avg_vec = bVectors.reduce(_ + _) / m.toDouble

    // Within-anchor covariance: Sigma_i^within = (1/(m-1)) * sum((u_ij - u_bar_i)(u_ij - u_bar_i)^T)
    val sigma_within = if (m > 1) {
      val centeredVectors = uVectors.map(_ - u_bar)
      val covMatrix = centeredVectors.map(u => u * u.t).reduce(_ + _) / (m - 1).toDouble
      covMatrix
    } else {
      // If m = 1, within-anchor covariance is zero
      DenseMatrix.zeros[Double](u_bar.length, u_bar.length)
    }

    (u_bar, sigma_within, b_avg_vec)
  }

  /**
   * Auto-determine whether to use distributed or local aggregation based on number of anchors.
   *
   * @param anchorStatsRDD RDD of (u_bar_i, Sigma_i_within, B_avg_i) for each anchor
   * @param alpha Debiasing parameter
   * @param m Partners per anchor
   * @param n Total sample size
   * @param expectedAnchors Expected number of anchors (for threshold decision)
   * @param distributedThreshold Threshold above which to use distributed computation (default: 1000)
   * @return Final variance matrix
   */
  def computeVarianceComponents(
    anchorStatsRDD: RDD[(DenseVector[Double], DenseMatrix[Double], DenseVector[Double])],
    alpha: Double,
    m: Int,
    n: Int,
    expectedAnchors: Int,
    lambda: Double = 1e-4,
    distributedThreshold: Int = 1000
  ): (DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double]) =
    if (expectedAnchors <= distributedThreshold) {
      // Small number of anchors: collect to driver and use array-based computation
      val anchorStats = anchorStatsRDD.collect()
      computeVarianceComponentsLocal(anchorStats, alpha, m, n, expectedAnchors, lambda)
    } else {
      // Large number of anchors: use distributed computation
      computeVarianceComponentsDistributed(anchorStatsRDD, alpha, m, n, expectedAnchors, lambda)
    }

  /**
   * Aggregate anchor statistics into final variance estimate (distributed version).
   *
   * @param anchorStatsRDD RDD of (u_bar_i, Sigma_i_within, B_avg_i) for each anchor
   * @param alpha Debiasing parameter
   * @param m Partners per anchor
   * @param n Total sample size
   * @return Final variance matrix
   */
  def computeVarianceComponentsDistributed(
    anchorStatsRDD: RDD[(DenseVector[Double], DenseMatrix[Double], DenseVector[Double])],
    alpha: Double,
    m: Int,
    n: Int,
    s_total: Int,
    lambda: Double = 1e-4
  ): (DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double]) = {

    // Get dimension from first element (if available)
    val firstAnchor = anchorStatsRDD.first()
    val d = firstAnchor._1.length

    // Distributed aggregation using reduce operations (no need to count, we know s_total)
    val aggregated = anchorStatsRDD.aggregate(
      // Zero value: (sum_u_bar_outer, sum_sigma_within, sum_b_avg)
      (
        DenseMatrix.zeros[Double](d, d),
        DenseMatrix.zeros[Double](d, d),
        DenseVector.zeros[Double](d * d)
      )
    )(
      // Sequence operation: add each anchor's contribution
      (acc, anchor) => {
        val (u_bar, sigma_within, b_avg) = anchor
        val (sum_u_outer, sum_sigma_within, sum_b_avg) = acc
        (
          sum_u_outer + (u_bar * u_bar.t), // Between-anchor variance component
          sum_sigma_within + sigma_within, // Within-anchor variance component
          sum_b_avg + b_avg // B matrix component
        )
      },
      // Combine operation: merge partial results
      (acc1, acc2) =>
        (
          acc1._1 + acc2._1, // sum_u_outer
          acc1._2 + acc2._2, // sum_sigma_within
          acc1._3 + acc2._3 // sum_b_avg
        )
    )

    val (sum_u_outer, sum_sigma_within, sum_b_avg) = aggregated

    if (s_total == 0) {
      throw new IllegalArgumentException(
        "Cannot compute variance with zero anchors. Check anchor sampling parameters."
      )
    }

    // Return intermediate components instead of final variance
    (sum_u_outer, sum_sigma_within, sum_b_avg)
  }

  /**
   * Aggregate anchor statistics into final variance estimate (local/array version).
   * Optimized for small numbers of anchors.
   *
   * @param anchorStats Array of (u_bar_i, Sigma_i_within, B_avg_i) for each anchor
   * @param alpha Debiasing parameter
   * @param m Partners per anchor
   * @param n Total sample size
   * @return Final variance matrix
   */
  def computeVarianceComponentsLocal(
    anchorStats: Array[(DenseVector[Double], DenseMatrix[Double], DenseVector[Double])],
    alpha: Double,
    m: Int,
    n: Int,
    s_total: Int,
    lambda: Double = 1e-4
  ): (DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double]) = {
    if (s_total == 0) {
      throw new IllegalArgumentException(
        "Cannot compute variance with zero anchors. Check anchor sampling parameters."
      )
    }

    // Aggregate components using array operations
    val sum_u_outer = anchorStats
      .map { case (u_bar, _, _) => u_bar * u_bar.t }
      .reduce(_ + _)

    val sum_sigma_within = anchorStats
      .map { case (_, sigma_i, _) => sigma_i }
      .reduce(_ + _)

    val sum_b_avg = anchorStats
      .map { case (_, _, b_avg) => b_avg }
      .reduce(_ + _)

    // Return intermediate components instead of final variance
    (sum_u_outer, sum_sigma_within, sum_b_avg)
  }

  /**
   * Common variance computation logic shared between distributed and local versions.
   *
   * @param sum_u_outer Sum of u_bar * u_bar.t across all anchors
   * @param sum_sigma_within Sum of within-anchor covariances
   * @param sum_b_avg Sum of B vectors across all anchors
   * @param s Number of anchors
   * @param alpha Debiasing parameter
   * @param m Partners per anchor
   * @param n Total sample size
   * @return Final variance matrix
   */
  def computeFinalVariance(
    sum_u_outer: DenseMatrix[Double],
    sum_sigma_within: DenseMatrix[Double],
    sum_b_avg: DenseVector[Double],
    s: Int,
    alpha: Double,
    m: Int,
    n: Int,
    lambda: Double = 1e-4,
    penalty: Option[DenseMatrix[Double]] = None
  ): DenseMatrix[Double] = {
    val d = sum_b_avg.length / sum_u_outer.rows // Infer dimension

    // Compute final variance components
    val sigma_between = sum_u_outer / s.toDouble
    val sigma_within = sum_sigma_within / s.toDouble

    // Debiasing factor: alpha * ((1/m) - (1/(n-1)))
    val debias_factor = alpha * ((1.0 / m) - (1.0 / (n - 1)))

    // Debiased variance estimate: Sigma_hat = Sigma_between - debias_factor * Sigma_within
    val sigma_hat = sigma_between - (sigma_within * debias_factor)

    // Ensure positive semidefinite only when alpha > 0 (when debiasing is applied)
    val sigma_hat_final = if (alpha > 0.0) {
      ensurePositiveSemidefinite(sigma_hat)
    } else {
      sigma_hat
    }

    // Average B vector across all anchors: B_avg = (1/s) * sum(B_avg_i)
    val B_avg_vec = sum_b_avg / s.toDouble

    // Convert to matrix only for final inversion
    val penaltyMatrix = penalty.getOrElse(DenseMatrix.eye[Double](d))
    val B_matrix = new DenseMatrix(d, d, B_avg_vec.data) + lambda * penaltyMatrix

    // Use pseudo-inverse for B matrix (handles singularity gracefully)
    val B_inv = generalizedInverse(B_matrix)

    // Sandwich variance: 4 * B^(-1) * Sigma_hat * B^(-1)^T (unscaled, like original DRGU)
    // Scaling by n_clusters happens in summary() method
    val variance = B_inv * sigma_hat_final * B_inv.t * 4.0

    // Final safety check: ensure no zero diagonal elements (can happen with edge cases)
    val result = variance.copy
    for (i <- 0 until math.min(result.rows, result.cols))
      if (result(i, i) <= 0.0) {
        result(i, i) = 1e-6 // Minimal reasonable variance
      }

    result
  }

  /**
   * Ensure a matrix is positive semidefinite by clipping negative eigenvalues.
   * Uses proper eigenvalue decomposition and reconstruction.
   */
  def ensurePositiveSemidefinite(matrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    // Use symmetric eigenvalue decomposition for symmetric matrices
    val EigSym(eigenValues, eigenVectors) = eigSym(matrix)

    // Clip negative eigenvalues to zero (standard approach)
    val clippedEigenValues = eigenValues.map(ev => math.max(ev, 0.0))

    // Reconstruct the matrix: V * diag(clipped_eigenvalues) * V^T
    val clippedMatrix = eigenVectors * diag(clippedEigenValues) * eigenVectors.t

    clippedMatrix
  }

}
