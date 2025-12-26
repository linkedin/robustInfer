package robustinfer

sealed trait DistributionFamily
case object Binomial extends DistributionFamily
case object Gaussian extends DistributionFamily
case object Poisson extends DistributionFamily
case object NegativeBinomial extends DistributionFamily
case class Tweedie(p: Double) extends DistributionFamily // p is pre-specified, phi will be estimated
