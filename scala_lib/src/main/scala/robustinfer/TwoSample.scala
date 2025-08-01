package robustinfer

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.DoubleType
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.spark.rdd.RDD

object TwoSample {

  def zeroTrimmedU(
    xRdd: RDD[Double],
    yRdd: RDD[Double],
    alpha: Double = 0.05,
    scale: Boolean = true, 
    tieCorrection: Boolean = false
  ): (Double, Double, Double, (Double, Double)) = {
    // 1) Basic counts & checks
    val n0 = xRdd.count.toDouble
    val n1 = yRdd.count.toDouble
    require(n0 > 0 && n1 > 0, "Both RDDs must be non-empty")
    require(xRdd.filter(_ < 0).isEmpty(), "All x must be ≥ 0")
    require(yRdd.filter(_ < 0).isEmpty(), "All y must be ≥ 0")

    // 2) Proportions of non-zeros
    val xPlus = xRdd.filter(_ > 0)
    val yPlus = yRdd.filter(_ > 0)
    val pHat0 = xPlus.count / n0
    val pHat1 = yPlus.count / n1
    val pHat = math.max(pHat0, pHat1)

    // 3) Truncate zeros
    val nPrime0 = math.round(n0 * pHat)
    val nPrime1 = math.round(n1 * pHat)
    val nPlus0 = xPlus.count().toDouble
    val nPlus1 = yPlus.count().toDouble
    val pad0 = Seq.fill((nPrime0 - nPlus0).toInt)(0.0)
    val pad1 = Seq.fill((nPrime1 - nPlus1).toInt)(0.0)

    val xTrun = xRdd.sparkContext.parallelize(pad0) union xPlus
    val yTrun = yRdd.sparkContext.parallelize(pad1) union yPlus

    // 4) Compute descending‐ordinal ranks
    val tagged: RDD[(Double, Boolean)] =
      yTrun.map(v => (v, true)) union xTrun.map(v => (v, false))

    val ranks: RDD[((Double, Boolean), Double)] = computeAverageRanks(tagged)

    val R1: Double =
      ranks.filter { case ((_, isY), _) => isY }
        .map { case (_, idx) => idx.toDouble }
        .sum()

    // 5) Wilcoxon-style statistic
    val wPrime = - (R1 - nPrime1 * (nPrime0 + nPrime1 + 1) / 2.0)

    // 6) Variance components
    val varComp1 = (n1 * n0 * n1 * n0 / 4.0) * (pHat * pHat) * (
      (pHat0 * (1 - pHat0) / n0) + (pHat1 * (1 - pHat1) / n1)
    )
    val correction = if (tieCorrection) {
      // Count ties on positives, and compute correction factor
      val valueCounts = tagged.filter(_._1 > 0).map(_._1).countByValue()
      val tiesFactor = valueCounts.values.filter(_ > 1).map(t => t * (t * t - 1)).sum.toDouble
      val nPlus = nPlus0 + nPlus1
      1.0 - tiesFactor / (nPlus * (nPlus * nPlus - 1))
    } else {
      1.0
    }
    val varComp2 = (nPlus0 * nPlus1 * (nPlus0 + nPlus1)) / 12.0 * correction
    val varW = varComp1 + varComp2

    // 7) Z and p-value
    val z = wPrime / math.sqrt(varW)
    val pValue = 2 * (1 - normalCDF(math.abs(z)))
    val zAlpha = normalQuantile(1 - alpha / 2)
    val confidenceInterval = (wPrime - zAlpha * math.sqrt(varW), wPrime + zAlpha * math.sqrt(varW))

    // 8) Scale the statistic to P(X' < Y')
    if (scale) {
      val locationFactor = (nPrime1 * nPrime0 ) * 0.5
      val scaleFactor = 1.0 * nPrime1  * nPrime0 
      val wPrimeScaled = (wPrime + locationFactor)/scaleFactor
      val confidenceIntervalScaled = (
        (confidenceInterval._1 + locationFactor) / scaleFactor,
        (confidenceInterval._2 + locationFactor) / scaleFactor
      )
      return (z, pValue, wPrimeScaled, confidenceIntervalScaled)
    }

    (z, pValue, wPrime, confidenceInterval)
  }

  def computeAverageRanks(
    values: RDD[(Double, Boolean)],  // (value, isFromTreatment)
    descending: Boolean = true
  ): RDD[((Double, Boolean), Double)] = {
  
    // Step 1: Sort values (descending or ascending)
    val sorted = if (descending)
      values.sortBy({ case (v, _) => -v })
    else
      values.sortBy({ case (v, _) => v })

    // Step 2: Assign provisional ranks (starting at 1)
    val ranked = sorted.zipWithIndex().map {
      case ((v, isY), idx) => (v, (idx + 1L).toDouble, isY)
    }

    // Step 3: Group by value to compute average rank for ties
    val avgRanksByValue: RDD[(Double, Double)] = ranked
      .map { case (v, rank, _) => (v, rank) }
      .groupByKey()
      .mapValues(ranks => ranks.sum / ranks.size)

    // Step 4: Join average ranks back to original (value, isY) key
    val withAvgRanks = ranked
        .map { case (v, _, isY) => (v, isY) } // key = v, value = isY
        .join(avgRanksByValue)               // join on v
        .map { case (v, (isY, avgRank)) => ((v, isY), avgRank) }

    withAvgRanks
  }

  def mwU(
    xRdd: RDD[Double],
    yRdd: RDD[Double],
    alpha: Double = 0.05,
    scale: Boolean = true, 
    tieCorrection: Boolean = false
  ): (Double, Double, Double, (Double, Double)) = {
    // This function performs the Mann-Whitney U test on two RDDs of doubles.
    // 1) Basic counts & checks
    val n0 = xRdd.count().toDouble
    val n1 = yRdd.count().toDouble
    require(n0 > 0 && n1 > 0, "Both RDDs must be non-empty")

    // 2) Compute descending-ordinal ranks with avg ranks for ties
    val tagged = yRdd.map(v => (v, 1)) union xRdd.map(v => (v, 0))

    val sortedWithIdx = tagged
      .sortBy({ case (v, _) => -v })
      .zipWithIndex()
      .map { case ((value, label), idx) => (value, (label, idx + 1)) }

    val grouped = sortedWithIdx
      .map { case (v, (label, rank)) => (v, (label, rank.toDouble)) }
      .groupByKey()
      .flatMap { case (_, entries) =>
        val (labels, ranks) = entries.unzip
        val avgRank = ranks.sum / ranks.size
        labels.map(label => (label, avgRank))
      }

    // 3) Wilcoxon-style statistic
    val R1 = grouped.filter(_._1 == 1).map(_._2).sum()
    val w = - (R1 - n1 * (n0 + n1 + 1) / 2.0)

    // 4) Tie-adjusted variance
    val totalN = n0 + n1
    val correction = if (tieCorrection) {
      // Count ties
      val valueCounts = tagged.map(_._1).countByValue()
      val ties = valueCounts.values.filter(_ > 1).map(t => t * (t * t - 1)).sum
      1.0 - ties / (totalN * (totalN * totalN - 1))
    } else {
      1.0
    }
    val varW = n0 * n1 * (totalN + 1) / 12.0 * correction

    // 5) Z and p-value
    val z = w / math.sqrt(varW)
    val pValue = 2 * (1 - normalCDF(math.abs(z)))
    val zAlpha = normalQuantile(1 - alpha / 2)
    val confidenceInterval = (w - zAlpha * math.sqrt(varW), w + zAlpha * math.sqrt(varW))

    // 6) Scale the statistic to P(X' < Y')
    if (scale) {
      val locationFactor = (n1 * n0) / 2.0
      val scaleFactor = n1 * n0
      val wScaled = (w + locationFactor) / scaleFactor
      val confidenceIntervalScaled = (
        (confidenceInterval._1 + locationFactor) / scaleFactor,
        (confidenceInterval._2 + locationFactor) / scaleFactor
      )
      return (z, pValue, wScaled, confidenceIntervalScaled)
    }
    
    (z, pValue, w, confidenceInterval)
  }

  def tTest(
    xRdd: RDD[Double],
    yRdd: RDD[Double],
    alpha: Double = 0.05
  ): (Double, Double, Double, (Double, Double)) = {
    // This function performs a two-sample t-test on two RDDs of doubles.
    // 1) Basic counts & checks
    val n0 = xRdd.count.toDouble
    val n1 = yRdd.count.toDouble
    require(n0 > 0 && n1 > 0, "Both RDDs must be non-empty")

    // 2) Calculate means, variances, and counts for each group
    val mean0 = xRdd.mean()
    val mean1 = yRdd.mean()
    val var0 = xRdd.variance()
    val var1 = yRdd.variance()

    // 3) Perform the t-test
    val stdErrorDifference = math.sqrt(var0 / n0 + var1 / n1)
    val z = (mean1 - mean0) / stdErrorDifference

    // 4) Calculate the p-value using the normal distribution CDF
    val pValue = 2 * (1 - normalCDF(math.abs(z)))

    // 5) Calculate the 95% confidence interval for the mean difference
    val meanDifference = mean1 - mean0
    val zAlpha = normalQuantile(1 - alpha / 2)
    val confidenceInterval = (meanDifference - zAlpha * stdErrorDifference, meanDifference + zAlpha * stdErrorDifference)

    (z, pValue, meanDifference, confidenceInterval)
  }

  def zeroTrimmedUDf(data: DataFrame, groupCol: String, valueCol: String,
    controlStr: String, treatmentStr: String, alpha: Double): (Double, Double, Double, (Double, Double)) = {
    // This test basically test P(X < Y) = 0.5, where X is a random variable from control group and Y is a random variable from treatment group
    // Filter and select the relevant data
    val filteredData = data
      .withColumn(valueCol, col(valueCol).cast(DoubleType))
      .filter(col(groupCol).isin(controlStr, treatmentStr)).cache()

    // Ensure that the value column is non-negative
    require(filteredData.filter(col(valueCol) < 0).count() == 0,
      s"All values in column '$valueCol' must be non-negative for zeroTrimmedU.")

    // Calculate counts, percentage, and other statistics for each group
    val summary = filteredData.groupBy(groupCol).agg(
      sum(when(col(valueCol) > 0, 1.0).otherwise(0.0)).as("positiveCount"),
      mean(when(col(valueCol) > 0, 1.0).otherwise(0.0)).as("theta"),
      count(valueCol).alias("count")).cache()
    
    val n0Plus = summary.filter(col(groupCol) === controlStr).first().getDouble(1)
    val p0Hat = summary.filter(col(groupCol) === controlStr).first().getDouble(2)
    val n0 = summary.filter(col(groupCol) === controlStr).first().getLong(3)

    val n1Plus = summary.filter(col(groupCol) === treatmentStr).first().getDouble(1)
    val p1Hat = summary.filter(col(groupCol) === treatmentStr).first().getDouble(2)
    val n1 = summary.filter(col(groupCol) === treatmentStr).first().getLong(3)

    summary.unpersist()

    val pHat = if (p0Hat > p1Hat) p0Hat else p1Hat
    val samplingGrpStr = if (p0Hat > p1Hat) treatmentStr else controlStr
    val samplingSize = math.round(math.abs(p0Hat - p1Hat) * (if (p0Hat > p1Hat) n1 else n0)).toInt
    val zeroData = filteredData.filter(col(groupCol) === samplingGrpStr).filter(col(valueCol) === 0).limit(samplingSize)
    val positiveData = filteredData.filter(col(valueCol) > 0)
    val trimmedData = positiveData.union(zeroData)
    trimmedData.cache()

    val rankedData = trimmedData.withColumn("rank", row_number().over(Window.orderBy(desc(valueCol))))
      .withColumn("rankD", col("rank").cast(DoubleType))
    val r1 = rankedData.filter(col(groupCol) === treatmentStr).agg(sum("rankD")).first().getDouble(0)
    val n0Prime = trimmedData.filter(col(groupCol) === controlStr).count().toDouble
    val n1Prime = trimmedData.filter(col(groupCol) === treatmentStr).count().toDouble
    trimmedData.unpersist()
    filteredData.unpersist()

    val wPrime = - r1 + n1Prime * (n1Prime + n0Prime + 1) / 2

    val varComp1 = math.pow(n0, 2) * math.pow(n1, 2) / 4 *
      math.pow(pHat, 2) *
      ((p0Hat * (1 - p0Hat)) / n0 + (p1Hat * (1 - p1Hat)) / n1)
    val varComp2 = n1Plus * n0Plus * (n1Plus + n0Plus) / 12
    val varW = varComp1 + varComp2

    val z = wPrime / math.sqrt(varW)

    // Calculate the p-value using the normal distribution CDF
    val pValue = 2 * (1 - normalCDF(math.abs(z)))
    val zAlpha = normalQuantile(1 - alpha / 2)
    val confidenceInterval = (wPrime - zAlpha * math.sqrt(varW), wPrime + zAlpha * math.sqrt(varW))

    (z, pValue, wPrime, confidenceInterval)
  }

  def tTestDf(data: DataFrame, groupCol: String, valueCol: String,
    controlStr: String, treatmentStr: String, alpha: Double): (Double, Double, Double, (Double, Double)) = {
    // Filter and select the relevant data
    val filteredData = data
      .withColumn(valueCol, col(valueCol).cast(DoubleType))
      .filter(col(groupCol).isin(controlStr, treatmentStr)).cache()

    // Calculate means, variances, and counts for each group
    val summary = filteredData.groupBy(groupCol).agg(
      mean(valueCol).alias("mean"),
      variance(valueCol).alias("variance"),
      count(valueCol).alias("count")
    ).cache()

    // Extract mean, variance, and count for control and treatment
    val controlMean = summary.filter(col(groupCol) === controlStr).first().getDouble(1)
    val controlVariance = summary.filter(col(groupCol) === controlStr).first().getDouble(2)
    val controlCount = summary.filter(col(groupCol) === controlStr).first().getLong(3)

    val treatmentMean = summary.filter(col(groupCol) === treatmentStr).first().getDouble(1)
    val treatmentVariance = summary.filter(col(groupCol) === treatmentStr).first().getDouble(2)
    val treatmentCount = summary.filter(col(groupCol) === treatmentStr).first().getLong(3)

    summary.unpersist()
    filteredData.unpersist()

    // Perform the t-test
    val stdErrorDifference = math.sqrt(controlVariance/ controlCount + treatmentVariance / treatmentCount)
    val t = (treatmentMean - controlMean) / stdErrorDifference

    // Calculate the p-value using the normal distribution CDF
    val pValue = 2 * (1 - normalCDF(math.abs(t)))

    // Calculate the 95% confidence interval for the mean difference
    val meanDifference = treatmentMean - controlMean
    val zAlpha = normalQuantile(1 - alpha / 2)
    val confidenceInterval = (meanDifference - zAlpha * stdErrorDifference, meanDifference + zAlpha * stdErrorDifference)

    (t, pValue, meanDifference, confidenceInterval)
  }

  // Custom implementation of the normal distribution cumulative distribution function (CDF)
  def normalCDF(t: Double): Double = {
    val standardNormal = new NormalDistribution(0, 1)
    standardNormal.cumulativeProbability(Math.abs(t))
  }
  // Custom implementation of the normal distribution quantile function (inverse CDF)
  def normalQuantile(p: Double): Double = {
    val standardNormal = new NormalDistribution(0, 1)
    standardNormal.inverseCumulativeProbability(p)
  }
}
