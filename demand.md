
## Elasticity-Adjusted AR(1) Process

Let $D_t$ be the gas demand at time $t$. The process combines price elasticity with an autoregressive component:

$$D_t = \mu_t + \rho(D_{t-1} - \mu_t) + \varepsilon_t\$$

where:
- $\varepsilon_t \sim N(0, \sigma^2)$ is the random shock
- $\sigma = \nu \cdot \mu_t$ where $\nu$ is the volatility ratio and $\mu_t$ is the mean demand at time t

## Price Elasticity Component

The time-varying mean $\mu_t$ adjusts according to:

$$\mu_t = \mu_0 \cdot \left(\frac{P_{t}}{P_{ref}}\right)^{\eta}$$

where:
- $\mu_0$ is the base mean demand (before elasticity adjustment)
- $P_{t-1}$ is the gas price in the previous period
- $P_{ref}$ is the reference price at which demand equals the base mean
- $\eta$ is the price elasticity coefficient (typically negative)

## Parameters
- $\rho \in [-0.99, 0.99]$: AR(1) persistence coefficient
- $\eta < 0$: Price elasticity (e.g., $\eta = -0.5$ means 1% price increase reduces mean demand by 0.5%)
- $\nu > 0$: Volatility ratio (standard deviation as fraction of base mean)

This formulation allows the expected demand to vary with price through the elasticity mechanism, while the AR(1) component captures persistence of deviations from the expected demand.
