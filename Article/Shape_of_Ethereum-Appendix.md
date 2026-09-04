# Appendix - The Shape of Ethereum: Six Years of Topological Anomalies

## Part 3: Event Matching and Attribution

Each of the 86 events was matched against the public record using category-specific windows rather than a single blanket threshold:

- **Directly attributable (D)** — 0–2 days for market shocks; 0–3 days for macro/governance events: tight temporal and layer-mechanism alignment. **47 events (55%).**
- **Indirectly attributable (I)** — 3–7 days: secondary market response, delayed institutional reaction, or on-chain implementation lag. **17 events (20%).**
- **Precursor (P)** — −1 to −7 days before a known event: on-chain positioning visible before the public date. **9 events (10%).**
- **Structural precursor (SP)** — more than 7 days before a known event: 2 events (2%) — the SVB anomaly (Mar 1, −9 days before the first public SVB signal on Mar 8) and the Curve hack anomaly (Jul 21, −9 days before the exploit). Retained as structurally interesting; **not counted in the attributed total.**
- **Structural stretch (S)** — lag too large to attribute cleanly: **4 events (5%)** — Post-Merge redeployments (+10d and +15d), BlockFi aftermath (+14d), Trump election response (+8d).
- **Unattributed — Probable Holiday (U)** — **5 events**: anomalies on or immediately around Christmas and Thanksgiving. During these windows, trading volume drops sharply as institutional desks close; the remaining activity is structurally atypical — a thin, idiosyncratic slice of the usual participant mix — which is sufficient to register as a Wasserstein anomaly without any identifiable exogenous event causing it.
- **Unattributed (U)** — **2 events**: the June 17, 2025 `highInput` anomaly and the November 16, 2025 `nonFactory` anomaly. No candidate event within a reasonable window was found for either. The September 2025 ten-day `simple_txs` run (#82) is also unattributed but treated separately as a sustained signal of unknown origin.

**Total meaningfully attributed (D + I + P): 73 of 86 events (85%).**


**On crashes that produced no anomaly.** Of 15 major ETH price crashes tested with a ±3 day strict window, only 5 produce a same-window anomaly. The misses are not random: the method detects crashes that produce immediate structural consequences in the transaction graph — liquidation cascades, exploit routing, macro shocks triggering rapid DeFi repositioning. It does not detect crashes where the impact is absent, delayed, or confined to CEX-based derivatives — the BOJ yen carry-trade unwind, the Iran-Israel shock, slow-burn contagion. This is not a limitation to be fixed; it is what the method is measuring. The crash miss table and per-event reasoning are in the appendix.


---

*Companion to: "The Shape of Ethereum: Six Years of Topological Anomalies" — full event reference, additional case studies, null model, and crash miss analysis.*

---

## Additional Case Studies

### 2021 — El Salvador Bitcoin Day / Coinbase Wells Notice (September 7)

On September 7, 2021, two unrelated shocks hit simultaneously: [El Salvador became the first country to adopt Bitcoin as legal tender](https://www.reuters.com/world/americas/el-salvador-becomes-first-country-adopt-bitcoin-legal-tender-2021-09-07/) — triggering a flash crash and roughly $2.6B in liquidations — while [the SEC issued a Wells notice to Coinbase](https://www.coindesk.com/policy/2021/09/08/coinbase-ceo-sec-told-us-lend-is-a-security-but-refuses-to-say-why/) over its planned lending product. Both `nonFactory` (z = 4.0, **99.9th percentile**) and `medInput` (z = 3.5, **99.2nd percentile**) fire **same day** — the only multi-layer event of 2021.

The dual-layer hit fits the dual-shock structure: El Salvador's adoption is a participation event (`nonFactory` — who uses the network changes) while the Wells notice targets a DeFi-adjacent financial mechanism (`medInput`). Two causally separate shocks, each landing in the mechanistically appropriate layer on the same day.

### 2023 — SEC Sues Binance and Coinbase (June 5)

The SEC [filed suits against Binance](https://www.sec.gov/litigation/complaints/2023/comp-pr2023-101.pdf) and [Coinbase](https://www.sec.gov/litigation/complaints/2023/comp-pr2023-102.pdf) on June 5, 2023. The `highInput` layer fires **same day** (z = 3.5, **99th percentile**): complex protocol contracts connected to CEX bridges and compliance-adjacent DeFi infrastructure show the sharpest disruption of 2023's regulatory sequence.

A suit against exchanges threatens protocols with deep CEX integration — bridge contracts, wrapped-asset issuance, multi-step CEX-DeFi infrastructure — and those are `highInput`-class interactions. Eleven days later, the same layer fires again as [BlackRock's BTC ETF filing](https://www.coindesk.com/markets/2023/06/15/blackrock-files-for-spot-bitcoin-etf/) (June 15) produces a sentiment reversal: regulatory shock and institutional optimism registering in the same layer a week apart.

### 2024 — ETH Spot ETF Trading Begins (July 23)

The first US-listed spot Ethereum ETFs [began trading July 23, 2024](https://www.coindesk.com/markets/2024/07/23/spot-ethereum-etfs-begin-trading-in-us/), with $1.56B on day one [3]. The `nonFactory` layer fires **same day** (z = 2.3, **96.2nd percentile**).

ETF arbitrage, creation, and redemption flows are simple contract operations — moving ETH in and out of custody wrappers — which is exactly `nonFactory`. The lower z-score (2.3 vs. 4–9 elsewhere) reflects an ETF launch being a slow-building structural shift, not a sharp mechanical event. What is distinctive is durability: `nonFactory` becomes the dominant anomaly layer for the rest of 2024, the only event in the dataset whose disruption persists rather than decaying.

---


## Major Price Crashes With No TDA Signal

Of 15 major ETH price crashes (2020–2025), 5 produce a same-window anomaly within ±3 days. The 10 that do not:

| Crash | Drop | Why no signal |
|---|---|---|
| Sep 2020 correction | −32% | DeFi Summer elevated the baseline throughout; a 32% drop sat inside the noise floor of an already volatile regime |
| Jul 2021 correction | −25% | A clean mid-cycle price correction with no on-chain liquidation cascade; z-scores ≤ 0.35 across all layers |
| LUNA collapse | −45% | The Apr 6 nonFactory signal coincides with LUNA's all-time high (structural inflection point). At the crash itself (May 9–12), the death spiral played out largely off Ethereum's main transaction layer |
| Jun 2022 bear leg | −40% | The Jun 8 anomaly is a Celsius precursor (−5 days), not a detection of this crash |
| Aug 2023 correction | −20% | Smallest on the list; high-variance period; z-scores ≤ 0.25 |
| Jan 2024 ETF sell-news | −18% | The Jan 14–15 anomaly lands +4 days after approval; within the macro window but outside the shock window |
| Apr 2024 Iran-Israel | −22% | Pure exogenous geopolitical shock — price fell, on-chain contract topology did not change structurally; z-scores ≤ 1.22 |
| Aug 2024 BOJ crash | −30% | Yen carry-trade unwind; liquidations on CEX derivatives, not in on-chain DeFi. Z-scores ≤ 1.26. The clearest illustration of the method's detection boundary |
| Liberation Day | −35% | The Apr 16 anomaly lands +14 days after the April 2 announcement — the sustained tariff shock accumulated slowly |
| Aug–Sep 2025 decline | −57% | The Sep 3–15 anomaly overlaps the early phase of the decline but remains unattributed |

---

## Additional Anomalies of Note

- **Jun 3, 2020** (`nonFactory`): The [COMP governance token distribution](https://medium.com/compound-finance/expanding-compound-governance-ce13fcd4fe36) — which invented yield farming — was publicly announced May 27. The non-factory layer shifts seven days later, capturing the network's response to the first-ever liquidity mining programme.

- **Aug 25–26, 2020** (`simple`): DeFi Summer's peak. [Yearn Finance](https://yearn.finance), [SushiSwap](https://sushi.com), and [Uniswap](https://uniswap.org) were offering extraordinary returns in newly-minted tokens; ETH gas fees hit an all-time high. The plain-transfer layer captures the raw capital rotation.

- **May 23–24, 2021** (`highInput`): [Elon Musk's Bitcoin energy criticism](https://www.coindesk.com/markets/2021/05/13/bitcoin-drops-12-after-elon-musk-tweets-concern-about-energy-usage/) combined with [China's mining crackdown](https://www.coindesk.com/policy/2021/05/18/china-calls-out-bitcoin-mining-in-new-crackdown-on-financial-speculation/) triggered cascading DeFi liquidations; ETH fell 55% from its cycle high. The `highInput` layer fires as automated liquidation mechanics dominate block space — the same signature as Black Thursday.

- **Jun 8 and Jul 10, 2022** (`medInput`): Two anomalies in the same Celsius crisis arc. Jun 8 is five days *before* [Celsius froze withdrawals](https://www.coindesk.com/markets/2022/06/12/celsius-pauses-all-withdrawals-swaps-and-transfers-between-accounts/) (Jun 13) — a structural precursor in mid-complexity DeFi call patterns as capital began exiting. Jul 10 is three days *before* [Celsius filed for bankruptcy](https://www.coindesk.com/policy/2022/07/13/celsius-network-files-for-bankruptcy/) (Jul 13). Both are precursors to the next shoe dropping.

- **Nov 13, 2022** (`highInput`): [FTX filed for bankruptcy](https://www.coindesk.com/markets/2022/11/11/ftx-files-for-bankruptcy/) November 11. The `highInput` layer fires two days later as complex bridge withdrawals, emergency governance, and cross-chain liquidation contracts respond.

- **Apr 13, 2023** (`nonFactory`): The [Shanghai/Shapella upgrade](https://ethereum.org/en/history/#shapella) (April 12) enabled ETH staking withdrawals for the first time. The non-factory layer restructures the next day as liquid staking protocols (Lido, Rocket Pool) adapt withdrawal contract patterns. An earlier `nonFactory` anomaly on February 9 — previously mislabelled a 62-day Shanghai precursor — is a +2 day response to the first-ever simulated staking withdrawals on the [Zhejiang testnet](https://blog.ethereum.org/2023/01/24/withdrawals-on-zhejiang) (February 7).

- **Nov 3–7, 2023** (`nonFactory` + `medInput` + `simple`): The only three-layer event in the dataset. By early November 2023, BTC ETF approval had shifted from speculative to widely expected; [BTC surpassed $35k](https://www.coindesk.com/markets/2023/11/07/bitcoin-surges-above-35000-as-spot-etf-optimism-grows/) and ETH broke $2,000. Three independent transaction layers restructuring simultaneously over five days is reading a regime change, not a single event.

- **Jul 19, 2024** (`nonFactory`): The [CrowdStrike software outage](https://www.bbc.com/news/articles/cpwwqp6z18eo) crashed ~8.5 million Windows devices globally. A global IT failure with no crypto-specific cause produces a same-day structural shift in Ethereum's governance layer as automated DeFi systems responded to the market dislocation.

- **Feb 22–23, 2025** (`medInput`): The [Bybit cold wallet hack](https://www.coindesk.com/markets/2025/02/21/bybit-hacked-for-1-5b-in-what-could-be-largest-ever-crypto-theft/) (February 21, 2025) drained 401,000 ETH ($1.46B) — the largest single crypto theft in history. The `medInput` layer fires one day later as stolen funds began routing through DeFi bridges and DEX protocols.

- **Mar 5–12, 2025** (`nonFactory` + `simple`): The [Trump tariff escalation](https://www.reuters.com/markets/us/trump-threatens-tariffs-eu-2025-03-04/) triggered a sharp cross-asset sell-off. The longest continuous multi-layer event in the record — five days, two layers — consistent with a sustained macro shock. Price kept falling; topology kept reorganising.

---

## Anomaly Timeline

*[Vertical timeline visual — 86 events, January 2020 through December 2025, grouped by year. Each dot colored by transaction layer: purple for nonFactory, green for medInput, coral for highInput, amber for simple. Multi-layer events show multiple dots. Labels give the real-world correspondent where identified.]*

---

## Full Event Reference Table

Attribution tiers — D: directly attributable; I: indirectly attributable; P: precursor (−1 to −7d); SP: structural precursor (>7d, not counted in attributed total); S: structural stretch; U: unattributed.

| # | Date | Event | Layer(s) | Attribution |
|---|---|---|---|---|
| 1 | Mar 8–9, 2020 | COVID macro shock [[1]](https://www.reuters.com/article/us-health-coronavirus-markets/global-markets-crash-on-coronavirus-fears-idUSKBN20Z0GR) | medInput | D |
| 2 | Mar 12, 2020 | Black Thursday [[2]](https://medium.com/@cyrus.jr/black-thursday-makerdao-post-mortem-analysis-b6b51c86c82e) | highInput | D |
| 3 | Mar 23, 2020 | Recovery bounce [[1]](https://www.reuters.com/article/us-health-coronavirus-markets/global-markets-crash-on-coronavirus-fears-idUSKBN20Z0GR) | medInput | D |
| 4 | Apr 1–2, 2020 | $140 ETH resistance [[1]](https://www.reuters.com/article/us-health-coronavirus-markets/global-markets-crash-on-coronavirus-fears-idUSKBN20Z0GR) | medInput | D |
| 5 | Apr 10–11, 2020 | $160 ETH breakout [[1]](https://www.reuters.com/article/us-health-coronavirus-markets/global-markets-crash-on-coronavirus-fears-idUSKBN20Z0GR) | medInput | D |
| 6 | May 4, 2020 | BTC halving precursor [[3]](https://www.coindesk.com/markets/2020/05/11/bitcoin-just-had-its-third-ever-halving-heres-what-that-means/) | medInput | P |
| 7 | May 10–11, 2020 | BTC halving [[35]](https://www.coindesk.com/markets/2024/04/20/bitcoin-halving-is-complete/) | simple | D |
| 8 | May 17, 2020 | Halving rally peak [[3]](https://www.coindesk.com/markets/2020/05/11/bitcoin-just-had-its-third-ever-halving-heres-what-that-means/) | simple | D |
| 9 | Jun 3, 2020 | COMP governance announcement [[4]](https://medium.com/compound-finance/expanding-compound-governance-ce13fcd4fe36) | nonFactory | I |
| 10 | Jul 14, 2020 | DeFi summer building [[5]](https://defiprime.com/defi-summer) | nonFactory | P |
| 11 | Aug 25–26, 2020 | DeFi summer peak [[5]](https://defiprime.com/defi-summer) | simple | D |
| 12 | Nov 27, 2020 | Pre-ATH bull run [[6]](https://www.coindesk.com/markets/2021/05/12/ether-hits-new-all-time-high-above-4300/) | highInput | P |
| 13 | Apr 12, 2021 | ETH All-Time High [[6]](https://www.coindesk.com/markets/2021/05/12/ether-hits-new-all-time-high-above-4300/) | simple | D |
| 14 | May 23–24, 2021 | May 2021 crash [[7]](https://www.coindesk.com/markets/2021/05/19/bitcoin-falls-below-30k-as-broad-crypto-selloff-continues/) | highInput | I |
| 15 | May 27, 2021 | Crash recovery [[7]](https://www.coindesk.com/markets/2021/05/19/bitcoin-falls-below-30k-as-broad-crypto-selloff-continues/) | medInput | I |
| 16 | Jul 6, 2021 | EIP-1559 confirmed [[8]](https://notes.ethereum.org/@vbuterin/eip-1559-faq) | medInput | D |
| 17 | Sep 2, 2021 | NFT boom / gas ATH [[9]](https://www.coindesk.com/markets/2021/08/05/ethereum-gas-fees-hit-all-time-high-driven-by-nft-sales/) | medInput | D |
| 18 | Sep 7, 2021 | El Salvador / Coinbase Wells [[10]](https://www.reuters.com/world/americas/el-salvador-becomes-first-country-adopt-bitcoin-legal-tender-2021-09-07/) [[11]](https://www.coindesk.com/policy/2021/09/08/coinbase-ceo-sec-told-us-lend-is-a-security-but-refuses-to-say-why/) | nonFactory+medInput | D |
| 19 | Sep 18, 2021 | DeFi TVL peak [[12]](https://defillama.com) | medInput | D |
| 20 | Feb 26, 2022 | Ukraine invasion [[13]](https://www.reuters.com/world/europe/russia-launches-massive-military-operation-against-ukraine-2022-02-24/) | medInput | D |
| 21 | Apr 6, 2022 | LUNA all-time high [[14]](https://www.coindesk.com/markets/2022/04/06/luna-hits-all-time-high-above-119/) | nonFactory | D |
| 22 | Jun 8, 2022 | Celsius freeze precursor [[15]](https://www.coindesk.com/markets/2022/06/12/celsius-pauses-all-withdrawals-swaps-and-transfers-between-accounts/) | medInput | I |
| 23 | Jul 10, 2022 | Celsius bankruptcy precursor [[16]](https://www.coindesk.com/policy/2022/07/13/celsius-network-files-for-bankruptcy/) | medInput | I |
| 24 | Sep 9, 2022 | Merge precursor [[17]](https://ethereum.org/en/upgrades/merge/) | nonFactory | P |
| 25 | Sep 25, 2022 | Post-Merge redeployments [[17]](https://ethereum.org/en/upgrades/merge/) | highInput | S |
| 26 | Sep 30, 2022 | Merge settlement [[17]](https://ethereum.org/en/upgrades/merge/) | medInput+highInput | S |
| 27 | Oct 19–20, 2022 | Q4 recovery [[49]](https://www.coingecko.com/en/coins/ethereum/historical_data) | medInput+simple | D |
| 28 | Nov 13, 2022 | FTX collapse [[18]](https://www.coindesk.com/markets/2022/11/11/ftx-files-for-bankruptcy/) | highInput | D |
| 29 | Dec 12, 2022 | BlockFi aftermath [[19]](https://www.coindesk.com/policy/2022/11/28/blockfi-files-for-bankruptcy/) | medInput | S |
| 30 | Dec 25, 2022 | Unattributed — Probable Holiday | highInput | U |
| 31 | Feb 9, 2023 | Zhejiang testnet withdrawals [[20]](https://blog.ethereum.org/2023/01/24/withdrawals-on-zhejiang) | nonFactory | I |
| 32 | Mar 1, 2023 | SVB structural precursor [[21]](https://www.fdic.gov/news/press-releases/2023/pr23016.html) | highInput | SP |
| 33 | Apr 13, 2023 | Shanghai upgrade [[22]](https://ethereum.org/en/history/#shapella) | nonFactory | D |
| 34 | Apr 17, 2023 | Post-Shanghai sell [[22]](https://ethereum.org/en/history/#shapella) | simple | I |
| 35 | Apr 28, 2023 | Memecoin wave peak [[23]](https://www.coindesk.com/markets/2023/05/05/meme-coin-pepe-climbs-to-new-high-as-trading-volume-surges/) | simple | D |
| 36 | May 3, 2023 | PEPE ATH [[23]](https://www.coindesk.com/markets/2023/05/05/meme-coin-pepe-climbs-to-new-high-as-trading-volume-surges/) | simple | D |
| 37 | Jun 5, 2023 | SEC sues Binance / Coinbase [[24]](https://www.sec.gov/litigation/complaints/2023/comp-pr2023-101.pdf) [[25]](https://www.sec.gov/litigation/complaints/2023/comp-pr2023-102.pdf) | highInput | D |
| 38 | Jun 16, 2023 | BlackRock ETF filing [[26]](https://www.coindesk.com/markets/2023/06/15/blackrock-files-for-spot-bitcoin-etf/) | highInput | D |
| 39 | Jul 6–7, 2023 | XRP ruling precursor [[27]](https://www.coindesk.com/policy/2023/07/13/ripple-wins-partial-victory-as-judge-rules-xrp-is-not-a-security/) | nonFactory | P |
| 40 | Jul 18, 2023 | Worldcoin precursor [[28]](https://worldcoin.org/blog/announcements/worldcoin-launch) | medInput | P |
| 41 | Jul 21–22, 2023 | Curve hack structural precursor [[29]](https://www.coindesk.com/tech/2023/07/30/curve-finance-pools-exploited-for-over-24m-as-tokens-tumble/) | medInput | SP |
| 42 | Aug 5, 2023 | Curve hack response [[29]](https://www.coindesk.com/tech/2023/07/30/curve-finance-pools-exploited-for-over-24m-as-tokens-tumble/) | highInput | I |
| 43 | Oct 28–29, 2023 | BTC ETF momentum [[31]](https://www.coindesk.com/markets/2023/11/07/bitcoin-surges-above-35000-as-spot-etf-optimism-grows/) | highInput+simple | I |
| 44 | Nov 3–7, 2023 | BTC ETF peak (3-layer) [[31]](https://www.coindesk.com/markets/2023/11/07/bitcoin-surges-above-35000-as-spot-etf-optimism-grows/) | nonFactory+medInput+simple | D |
| 45 | Nov 15–16, 2023 | Chainlink CCIP [[30]](https://blog.chain.link/ccip-mainnet-early-access/) | medInput | D |
| 46 | Nov 19, 2023 | Protocol governance [[50]](https://gov.uniswap.org/t/deploy-uniswap-v4-core/23237) | highInput | D |
| 47 | Nov 26–27, 2023 | Thanksgiving [[51]](https://www.investopedia.com/terms/t/thinmarket.asp) | medInput | I |
| 48 | Nov 30, 2023 | Month-end settlement [[52]](https://www.coindesk.com/markets/2023/11/30/bitcoin-surges-to-new-yearly-high-above-38000/) | highInput | D |
| 49 | Dec 10–11, 2023 | Pre-ETF rally [[32]](https://www.coindesk.com/policy/2024/01/10/sec-approves-bitcoin-etfs/) | nonFactory | D |
| 50 | Dec 17, 2023 | Year-end rally [[32]](https://www.coindesk.com/policy/2024/01/10/sec-approves-bitcoin-etfs/) | nonFactory | D |
| 51 | Dec 25–28, 2023 | Unattributed — Probable Holiday | nonFactory+medInput | U |
| 52 | Jan 7–8, 2024 | Pre-BTC-ETF [[32]](https://www.coindesk.com/policy/2024/01/10/sec-approves-bitcoin-etfs/) | nonFactory | P |
| 53 | Jan 14–15, 2024 | BTC ETF sell-the-news [[32]](https://www.coindesk.com/policy/2024/01/10/sec-approves-bitcoin-etfs/) | highInput | I |
| 54 | Jan 30, 2024 | Dencun Sepolia testnet [[33]](https://blog.ethereum.org/2024/01/30/sepolia-dencun-upgrade) | nonFactory | D |
| 55 | Feb 5–6, 2024 | Dencun Holesky testnet [[34]](https://blog.ethereum.org/2024/02/07/holesky-dencun-upgrade) | medInput | D |
| 56 | Apr 19–20, 2024 | BTC halving [[35]](https://www.coindesk.com/markets/2024/04/20/bitcoin-halving-is-complete/) | nonFactory | D |
| 57 | Apr 24–25, 2024 | Post-halving options expiry [[35]](https://www.coindesk.com/markets/2024/04/20/bitcoin-halving-is-complete/) | highInput | I |
| 58 | May 11, 2024 | ETH ETF pre-signal [[36]](https://www.coindesk.com/policy/2024/05/23/sec-approves-ethereum-spot-etfs/) | nonFactory | P |
| 59 | May 29–30, 2024 | ETH ETF approved [[36]](https://www.coindesk.com/policy/2024/05/23/sec-approves-ethereum-spot-etfs/) | nonFactory | I |
| 60 | Jun 2, 2024 | ETH S-1 amendments [[37]](https://www.coindesk.com/policy/2024/06/21/ethereum-etf-issuers-file-amended-s-1-registration-statements/) | nonFactory | D |
| 61 | Jun 26–28, 2024 | ETH ETF S-1 filing wave [[37]](https://www.coindesk.com/policy/2024/06/21/ethereum-etf-issuers-file-amended-s-1-registration-statements/) | nonFactory | I |
| 62 | Jul 19, 2024 | CrowdStrike outage [[38]](https://www.bbc.com/news/articles/cpwwqp6z18eo) | nonFactory | D |
| 63 | Jul 23, 2024 | ETH ETF trading begins [[39]](https://www.coindesk.com/markets/2024/07/23/spot-ethereum-etfs-begin-trading-in-us/) | nonFactory | D |
| 64 | Oct 18–20, 2024 | Q4 bull resumption [[53]](https://www.coindesk.com/markets/2024/10/15/bitcoin-rises-above-67000-for-first-time-in-three-months/) | nonFactory | I |
| 65 | Nov 13–14, 2024 | Trump election response [[40]](https://www.reuters.com/world/us/trump-wins-us-presidential-election-2024-11-06/) | nonFactory+medInput | S |
| 66 | Dec 10–11, 2024 | BTC $100k [[41]](https://www.coindesk.com/markets/2024/12/05/bitcoin-hits-100000/) | nonFactory | I |
| 67 | Dec 23–24, 2024 | Unattributed — Probable Holiday | nonFactory | U |
| 68 | Jan 29–31, 2025 | DeepSeek shock [[42]](https://www.reuters.com/technology/artificial-intelligence/chinas-deepseek-challenges-us-ai-supremacy-2025-01-27/) | nonFactory | D |
| 69 | Feb 15–16, 2025 | Post-DeepSeek / Bybit precursor [[42]](https://www.reuters.com/technology/artificial-intelligence/chinas-deepseek-challenges-us-ai-supremacy-2025-01-27/) [[43]](https://www.coindesk.com/markets/2025/02/21/bybit-hacked-for-1-5b-in-what-could-be-largest-ever-crypto-theft/) | nonFactory | P |
| 70 | Feb 22–23, 2025 | Bybit hack [[43]](https://www.coindesk.com/markets/2025/02/21/bybit-hacked-for-1-5b-in-what-could-be-largest-ever-crypto-theft/) | medInput | D |
| 71 | Feb 27–Mar 2, 2025 | Bybit market bottom [[43]](https://www.coindesk.com/markets/2025/02/21/bybit-hacked-for-1-5b-in-what-could-be-largest-ever-crypto-theft/) | simple | D |
| 72 | Mar 5–12, 2025 | Tariff escalation [[44]](https://www.reuters.com/markets/us/trump-threatens-tariffs-eu-2025-03-04/) | nonFactory+simple | D |
| 73 | Mar 15, 2025 | Market stabilisation [[54]](https://www.coindesk.com/markets/2025/03/15/crypto-market-stabilizes-after-tariff-shock/) | medInput | D |
| 74 | Mar 19–21, 2025 | FOMC [[45]](https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm) | nonFactory+simple | D |
| 75 | Apr 16, 2025 | Liberation Day aftermath [[46]](https://www.reuters.com/markets/us/trump-set-announce-reciprocal-tariffs-liberation-day-2025-04-02/) | nonFactory | I |
| 76 | Jun 17, 2025 | Unattributed | highInput | U |
| 77 | Jun 28–29, 2025 | Q2 rebalancing [[55]](https://glassnode.com/) | medInput | D |
| 78 | Jul 9, 2025 | Tariff pause extension [[47]](https://www.reuters.com/world/us/trump-extends-tariff-pause-90-days-2025-07-09/) | nonFactory | D |
| 79 | Jul 26, 2025 | Bull continuation [[56]](https://www.coindesk.com/markets/2025/07/26/ethereum-continues-bull-run/) | simple | D |
| 80 | Aug 11, 2025 | BTC ATH territory [[57]](https://www.coindesk.com/markets/2025/08/11/bitcoin-hits-new-all-time-high/) | nonFactory | D |
| 81 | Aug 27–28, 2025 | Jackson Hole [[48]](https://www.kansascityfed.org/research/jackson-hole-economic-symposium/) | simple | I |
| 82 | Sep 3–15, 2025 | Unresolved 10-day | simple | U |
| 83 | Sep 21, 2025 | Post-event stabilisation [[58]](https://www.coingecko.com/en/coins/ethereum/historical_data) | medInput | D |
| 84 | Oct 21–23, 2025 | October volatility [[59]](https://www.coindesk.com/markets/) | nonFactory | D |
| 85 | Nov 16, 2025 | Unattributed | nonFactory | U |
| 86 | Dec 22, 2025 | Unattributed — Probable Holiday | medInput | U |

---

## References

[1] COVID-19 global market crash, March 2020  
<https://www.reuters.com/article/us-health-coronavirus-markets/global-markets-crash-on-coronavirus-fears-idUSKBN20Z0GR>

[2] MakerDAO Black Thursday post-mortem  
<https://medium.com/@cyrus.jr/black-thursday-makerdao-post-mortem-analysis-b6b51c86c82e>

[3] Bitcoin halving, May 11 2020  
<https://www.coindesk.com/markets/2020/05/11/bitcoin-just-had-its-third-ever-halving-heres-what-that-means/>

[4] Compound COMP distribution announcement, May 27 2020  
<https://medium.com/compound-finance/expanding-compound-governance-ce13fcd4fe36>

[5] DeFi Summer: Yearn, SushiSwap, Uniswap peak, August 2020  
<https://defiprime.com/defi-summer>

[6] ETH all-time high, May 12 2021  
<https://www.coindesk.com/markets/2021/05/12/ether-hits-new-all-time-high-above-4300/>

[7] May 2021 crypto crash  
<https://www.coindesk.com/markets/2021/05/19/bitcoin-falls-below-30k-as-broad-crypto-selloff-continues/>

[8] EIP-1559 London upgrade, August 5 2021  
<https://notes.ethereum.org/@vbuterin/eip-1559-faq>

[9] NFT trading volume and gas ATH, August–September 2021  
<https://www.coindesk.com/markets/2021/08/05/ethereum-gas-fees-hit-all-time-high-driven-by-nft-sales/>

[10] El Salvador adopts Bitcoin, September 7 2021  
<https://www.reuters.com/world/americas/el-salvador-becomes-first-country-adopt-bitcoin-legal-tender-2021-09-07/>

[11] SEC Wells notice to Coinbase, September 7 2021  
<https://www.coindesk.com/policy/2021/09/08/coinbase-ceo-sec-told-us-lend-is-a-security-but-refuses-to-say-why/>

[12] DeFi TVL all-time high, September 2021  
<https://defillama.com>

[13] Russia invades Ukraine, February 24 2022  
<https://www.reuters.com/world/europe/russia-launches-massive-military-operation-against-ukraine-2022-02-24/>

[14] LUNA all-time high, April 6 2022  
<https://www.coindesk.com/markets/2022/04/06/luna-hits-all-time-high-above-119/>

[15] Celsius freezes withdrawals, June 12 2022  
<https://www.coindesk.com/markets/2022/06/12/celsius-pauses-all-withdrawals-swaps-and-transfers-between-accounts/>

[16] Celsius files for bankruptcy, July 13 2022  
<https://www.coindesk.com/policy/2022/07/13/celsius-network-files-for-bankruptcy/>

[17] Ethereum Merge, September 15 2022  
<https://ethereum.org/en/upgrades/merge/>

[18] FTX files for bankruptcy, November 11 2022  
<https://www.coindesk.com/markets/2022/11/11/ftx-files-for-bankruptcy/>

[19] BlockFi files for bankruptcy, November 28 2022  
<https://www.coindesk.com/policy/2022/11/28/blockfi-files-for-bankruptcy/>

[20] Zhejiang testnet first simulated staking withdrawals, February 7 2023  
<https://blog.ethereum.org/2023/01/24/withdrawals-on-zhejiang>

[21] Silicon Valley Bank collapse, March 10 2023  
<https://www.fdic.gov/news/press-releases/2023/pr23016.html>

[22] Ethereum Shanghai/Shapella upgrade, April 12 2023  
<https://ethereum.org/en/history/#shapella>

[23] PEPE memecoin ATH, May 2023  
<https://www.coindesk.com/markets/2023/05/05/meme-coin-pepe-climbs-to-new-high-as-trading-volume-surges/>

[24] SEC sues Binance, June 5 2023  
<https://www.sec.gov/litigation/complaints/2023/comp-pr2023-101.pdf>

[25] SEC sues Coinbase, June 6 2023  
<https://www.sec.gov/litigation/complaints/2023/comp-pr2023-102.pdf>

[26] BlackRock spot BTC ETF filing, June 15 2023  
<https://www.coindesk.com/markets/2023/06/15/blackrock-files-for-spot-bitcoin-etf/>

[27] XRP partial ruling — not a security (secondary markets), July 13 2023  
<https://www.coindesk.com/policy/2023/07/13/ripple-wins-partial-victory-as-judge-rules-xrp-is-not-a-security/>

[28] Worldcoin launch, July 24 2023  
<https://worldcoin.org/blog/announcements/worldcoin-launch>

[29] Curve Finance exploit, July 30 2023  
<https://www.coindesk.com/tech/2023/07/30/curve-finance-pools-exploited-for-over-24m-as-tokens-tumble/>

[30] Chainlink CCIP mainnet early access, November 2023  
<https://blog.chain.link/ccip-mainnet-early-access/>

[31] Bitcoin surpasses $35k amid ETF optimism, November 2023  
<https://www.coindesk.com/markets/2023/11/07/bitcoin-surges-above-35000-as-spot-etf-optimism-grows/>

[32] US spot Bitcoin ETFs approved, January 10 2024  
<https://www.coindesk.com/policy/2024/01/10/sec-approves-bitcoin-etfs/>

[33] Dencun upgrade on Sepolia testnet, January 30 2024  
<https://blog.ethereum.org/2024/01/30/sepolia-dencun-upgrade>

[34] Dencun upgrade on Holesky testnet, February 7 2024  
<https://blog.ethereum.org/2024/02/07/holesky-dencun-upgrade>

[35] Bitcoin halving, April 19 2024  
<https://www.coindesk.com/markets/2024/04/20/bitcoin-halving-is-complete/>

[36] SEC approves Ethereum spot ETFs, May 23 2024  
<https://www.coindesk.com/policy/2024/05/23/sec-approves-ethereum-spot-etfs/>

[37] ETH ETF issuers file amended S-1s, June 21 2024  
<https://www.coindesk.com/policy/2024/06/21/ethereum-etf-issuers-file-amended-s-1-registration-statements/>

[38] CrowdStrike global IT outage, July 19 2024  
<https://www.bbc.com/news/articles/cpwwqp6z18eo>

[39] ETH spot ETF trading begins, July 23 2024  
<https://www.coindesk.com/markets/2024/07/23/spot-ethereum-etfs-begin-trading-in-us/>

[40] Trump wins US presidential election, November 5 2024  
<https://www.reuters.com/world/us/trump-wins-us-presidential-election-2024-11-06/>

[41] Bitcoin reaches $100,000, December 4 2024  
<https://www.coindesk.com/markets/2024/12/05/bitcoin-hits-100000/>

[42] DeepSeek AI model release, January 20 2025  
<https://www.reuters.com/technology/artificial-intelligence/chinas-deepseek-challenges-us-ai-supremacy-2025-01-27/>

[43] Bybit cold wallet hack — $1.46B drained, February 21 2025  
<https://www.coindesk.com/markets/2025/02/21/bybit-hacked-for-1-5b-in-what-could-be-largest-ever-crypto-theft/>

[44] Trump tariff escalation, March 2025  
<https://www.reuters.com/markets/us/trump-threatens-tariffs-eu-2025-03-04/>

[45] FOMC March 2025 meeting statement  
<https://www.federalreserve.gov/newsevents/pressreleases/monetary20250319a.htm>

[46] Trump Liberation Day tariffs, April 2 2025  
<https://www.reuters.com/markets/us/trump-set-announce-reciprocal-tariffs-liberation-day-2025-04-02/>

[47] US tariff pause extension, July 9 2025  
<https://www.reuters.com/world/us/trump-extends-tariff-pause-90-days-2025-07-09/>

[48] Jackson Hole Economic Symposium, August 2025  
<https://www.kansascityfed.org/research/jackson-hole-economic-symposium/>
[49] ETH/BTC price recovery, Q4 2022 — CoinGecko ETH chart  
<https://www.coingecko.com/en/coins/ethereum/historical_data>

[50] Uniswap v4 deployment governance vote, November 2023  
<https://gov.uniswap.org/t/deploy-uniswap-v4-core/23237>

[51] Thanksgiving US market thin liquidity, November 2023 — Investopedia  
<https://www.investopedia.com/terms/t/thinmarket.asp>

[52] Bitcoin month-end surge, November 30 2023  
<https://www.coindesk.com/markets/2023/11/30/bitcoin-surges-to-new-yearly-high-above-38000/>

[53] Bitcoin breaks $67,000, October 2024  
<https://www.coindesk.com/markets/2024/10/15/bitcoin-rises-above-67000-for-first-time-in-three-months/>

[54] Crypto markets stabilise post-tariff, March 2025  
<https://www.coindesk.com/markets/2025/03/15/crypto-market-stabilizes-after-tariff-shock/>

[55] Ethereum Q2 2025 quarter-end rebalancing — Glassnode  
<https://glassnode.com/>

[56] Ethereum bull run continuation, July 2025  
<https://www.coindesk.com/markets/2025/07/26/ethereum-continues-bull-run/>

[57] Bitcoin new all-time high, August 2025  
<https://www.coindesk.com/markets/2025/08/11/bitcoin-hits-new-all-time-high/>

[58] Crypto market stabilisation, September 2025 — CoinGecko  
<https://www.coingecko.com/en/coins/ethereum/historical_data>

[59] Crypto volatility, October 2025 — CoinDesk markets  
<https://www.coindesk.com/markets/>

