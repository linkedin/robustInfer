package robustinfer

sealed trait DistributionFamily
case object Binomial extends DistributionFamily
case object Gaussian extends DistributionFamily
case object Poisson extends DistributionFamily
