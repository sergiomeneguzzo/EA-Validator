# PortfolioEA — MQL5 Infrastructure

Portfolio EA framework with local vol targeting and risk-based sizing.
Strategy EAs write signals to the SignalBus (Global Variables); `MasterRiskManager` reads them and opens/closes positions.

---

## Architecture

```
PortfolioEA/
├── MasterRiskManager.mq5     ← single EA that opens positions
└── Core/
    ├── SignalBus.mqh          ← GlobalVar signal bus (MREG_{SYM}_{STRAT}_{FIELD})
    ├── RiskManager.mqh        ← lot clamping utility
    └── TradeManager.mqh
```

**Flow:**
1. Your strategy EAs call `SignalWrite()` each tick with direction, SL pips, TP pips
2. MasterRiskManager scans all signals every tick, calculates lot size, opens position
3. When the signal clears (`SignalClear()`), MRM closes the position

---

## Writing a Strategy EA

```mql5
#include "Core/SignalBus.mqh"

// Signal is active → MRM will open/manage the position
SignalWrite(_Symbol, "MyStrategy",
            direction,     // 1=BUY, -1=SELL
            1.0,           // weight (reserved)
            sl_pips,       // 0 = MRM falls back to ATR
            tp_pips);      // 0 = no TP

// Signal cleared → MRM closes position
SignalClear(_Symbol, "MyStrategy");
```

MRM uses the strategy name to derive the magic number (hash-based, deterministic).

---

## Installation

### 1. Copy the folder

```
<MT5_DATA>/MQL5/Experts/PortfolioEA/
```

macOS (Wine):
```
~/Library/Application Support/net.metaquotes.wine.metatrader5/
  drive_c/Program Files/MetaTrader 5/MQL5/Experts/PortfolioEA/
```

### 2. Compile

MetaEditor → open `MasterRiskManager.mq5` → F7. Must compile without errors.

---

## MasterRiskManager Configuration

```
=== PORTFOLIO ===
InpVolTarget        = 0.25      (25% annualised vol target)
InpVolLookbackDays  = 60
InpBaseMagic        = 999000

=== KILL SWITCH ===
InpKillSwitch       = false
```

The kill switch closes all positions with magic in range `[InpBaseMagic, InpBaseMagic+9973)`.

The `InpDefaultRisk` input applies to all strategies. Override per-strategy risk by passing it via the signal's `weight` field from your strategy EA.

---

## Vol Targeting

Every hour MRM recalculates:
```
vol_realized = StdDev(daily_equity_returns) × sqrt(252)
vol_scale    = InpVolTarget / vol_realized   → clamped [0.20, 3.00]
```

`vol_scale` multiplies every lot calculation. Monitor via Experts log:
```
[STATUS] equity=125430.00  vol_scale=1.243  positions=3
```

First 30 days: `vol_scale = 1.0` (not enough history).

---

## Sizing Logic

**Risk-based** (most strategies):
```
lots = (equity × risk%) / (sl_pts × tick_val/tick_sz) × vol_scale
```

**Exposure-based** (buy-and-hold, e.g. equity indices):
```
lots = equity × exposure% × vol_scale / notional_per_lot
```

---

## Pre-launch Checklist

- [ ] `PortfolioEA/` copied to `MQL5/Experts/`
- [ ] `MasterRiskManager.mq5` compiled without errors
- [ ] Strategy EAs compiled and attached to their charts
- [ ] `InpKillSwitch = false`
- [ ] Verify Experts log: `[MRM] Init — vol_target=25%  lookback=60 days  magic=999000`

Check signal bus in MT5: **Tools → Global Variables** — look for `MREG_` prefix.
