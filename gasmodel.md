### Recommended base-fee update rule (USDC-based Avalanche subnet)

\[
\boxed{%
B_{t+1}\;=\;\operatorname{clip}\!\Bigl(
B_t\;\Bigl[\,1+k\Bigl(\frac{\hat G_t}{G_{\text{target}}}-1\Bigr)\Bigr],
\;B_{\min},\,B_{\max}\Bigr)}
\]

**Where**

| symbol | meaning | suggested value |
|--------|---------|-----------------|
| \(B_t\) | base-fee (USDC / gas) in block \(t\) | dynamic |
| \(B_{t+1}\) | base-fee written into the header of block \(t{+}1\) | — |
| \(\hat G_t\) | smoothed gas-used in the last **\(W\) seconds** | see below |
| \(G_{\text{target}}\) | throughput target in the same \(W\)-s window | e.g. \(50\%\) of capacity |
| \(k\) | per-update learning rate | **0.125 (12.5 %)** |
| \(B_{\min}\) | fee floor (anti-spam) | e.g. \(1.0\times10^{-8}\;\text{USDC/gas}\) |
| \(B_{\max}\) | emergency fee ceiling | set very high, e.g. \(1\times10^{-3}\;\text{USDC/gas}\) |
| \(\operatorname{clip}(x,a,b)\) | bounds \(x\) into \([a,b]\) | — |

---

#### Smoothing the measured load  
Use a rolling window or EMA so that one spiky block cannot whipsaw the fee:

*Rolling-window average (10 s)*  
\[
\hat G_t \;=\; \frac1W \sum_{i=0}^{W/\Delta-1} G_{t-i}
\]
with \(W = 10\) s and average block interval \(\Delta\).

*Equivalent exponential moving average*  
\[
\hat G_t \;=\; (1-\alpha)\,\hat G_{t-1} + \alpha\,G_t,
\qquad
\alpha = \frac{\Delta}{W}.
\]

---

#### Typical parameter instantiation

* Hardware can safely execute **2 M gas s⁻¹**.  
* Target utilisation \(u_{\text{target}} = 0.5\).  
  \[
  G_{\text{target}} = u_{\text{target}}\times(2\,\text{M gas s⁻¹})\times W
                      \;=\; 1\,\text{M gas s⁻¹}\times10\text{ s}
                      \;=\; 10\;{\rm Mgas}.
  \]

With these numbers the base-fee

* **stays at \(B_{\min}\)** whenever the 10-s average load is ≤10 M gas,  
* **rises 12.5 %** for every extra 10 M gas (i.e. if load = 20 M gas, \(B_{t+1}=1.125\,B_t\)),  
* **falls 12.5 %** symmetrically when load is half the target.

All USDC collected as \(B_t\!\times\!G_t\) is **sent to a treasury/burn address**; optional priority-tips are handled normally but can likewise be burned to neutralise validator incentives.

This single closed-form rule reproduces Ethereum’s 1559 dynamics, tempered by a time-window average and bounded by \((B_{\min},B_{\max})\) for the price-stability guarantees needed on a payment-focused Avalanche subnet.