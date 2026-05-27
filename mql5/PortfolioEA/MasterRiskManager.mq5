#property version   "1.00"

#include "Core/SignalBus.mqh"
#include <Trade/Trade.mqh>

input group "=== PORTFOLIO ==="
input double InpVolTarget       = 0.20;   // annualised vol target (e.g. 0.20 = 20%)
input int    InpVolLookbackDays = 60;     // days of equity history for vol estimate
input int    InpBaseMagic       = 900000; // base magic — strategies offset from this

input group "=== RISK ==="
input double InpDefaultRisk     = 1.0;    // % equity risked per trade (default for all strategies)

input group "=== KILL SWITCH ==="
input bool   InpKillSwitch      = false;  // true = close all managed positions immediately

CTrade   g_trade;
double   g_vol_scale     = 1.0;
double   g_peak_equity   = 0.0;
double   g_daily_equity[];
int      g_equity_days   = 0;
datetime g_last_eq_date  = 0;
datetime g_last_vol_calc = 0;
datetime g_last_status   = 0;

// Deterministic magic from strategy name — stable across restarts
int StrategyHash(string name)
{
   int hash = 0;
   int n    = StringLen(name);
   int i;
   for(i = 0; i < n; i++)
      hash = (hash * 31 + (int)StringGetCharacter(name, i)) % 9973;
   return hash;
}

int MagicFor(string name) { return InpBaseMagic + StrategyHash(name); }

void UpdateDailyEquity()
{
   datetime today = (datetime)((long)TimeCurrent() / 86400 * 86400);
   if(today == g_last_eq_date) return;
   g_last_eq_date = today;

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq > g_peak_equity) g_peak_equity = eq;

   ArrayResize(g_daily_equity, g_equity_days + 1);
   g_daily_equity[g_equity_days] = eq;
   g_equity_days++;

   if(g_equity_days > InpVolLookbackDays)
   {
      int excess = g_equity_days - InpVolLookbackDays;
      int j;
      for(j = 0; j < InpVolLookbackDays; j++)
         g_daily_equity[j] = g_daily_equity[j + excess];
      ArrayResize(g_daily_equity, InpVolLookbackDays);
      g_equity_days = InpVolLookbackDays;
   }
}

void RecalcVolScale()
{
   if(g_equity_days < 30) { g_vol_scale = 1.0; return; }

   int    m = g_equity_days - 1;
   double returns[];
   ArrayResize(returns, m);

   int i;
   for(i = 0; i < m; i++)
   {
      if(g_daily_equity[i] <= 0.0) { g_vol_scale = 1.0; return; }
      returns[i] = (g_daily_equity[i + 1] - g_daily_equity[i]) / g_daily_equity[i];
   }

   double mean = 0.0;
   for(i = 0; i < m; i++) mean += returns[i];
   mean /= m;

   double var = 0.0;
   for(i = 0; i < m; i++) var += (returns[i] - mean) * (returns[i] - mean);
   var /= m;

   double vol = MathSqrt(var) * MathSqrt(252.0);
   if(vol < 0.0001) { g_vol_scale = 1.0; return; }

   g_vol_scale = InpVolTarget / vol;
   if(g_vol_scale < 0.20) g_vol_scale = 0.20;
   if(g_vol_scale > 3.00) g_vol_scale = 3.00;
}

// Risk-based lot sizing: equity * risk% / (sl_pts * tick_val) * vol_scale
double CalcLots(string symbol, double sl_points, double risk_pct)
{
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_amt  = equity * risk_pct / 100.0;
   double tick_val  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_sz   = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double mn        = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double mx        = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double step      = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(tick_sz <= 0.0 || tick_val <= 0.0 || sl_points <= 0.0) return mn;
   if(step <= 0.0) step = 0.01;

   // sl_points is price distance in points; tick_val is account currency per point per lot
   double base_lots  = risk_amt / (sl_points * tick_val);
   double final_lots = MathFloor(base_lots * g_vol_scale / step) * step;
   return MathMax(mn, MathMin(mx, NormalizeDouble(final_lots, 2)));
}

void CloseAllPositions()
{
   int i;
   for(i = PositionsTotal() - 1; i >= 0; i--)
   {
      int magic = (int)PositionGetInteger(POSITION_MAGIC);
      if(magic < InpBaseMagic || magic >= InpBaseMagic + 9973) continue;
      g_trade.SetExpertMagicNumber(magic);
      g_trade.PositionClose((ulong)PositionGetInteger(POSITION_TICKET));
   }
}

bool HasPosition(string symbol, int magic)
{
   int i;
   for(i = 0; i < PositionsTotal(); i++)
      if(PositionGetSymbol(i) == symbol &&
         PositionGetInteger(POSITION_MAGIC) == (long)magic) return true;
   return false;
}

void StoreOpenMeta(string sym, string strat, double sl, double tp)
{
   string key = "MREG_OPEN_" + sym + "_" + strat;
   GlobalVariableSet(key + "_SL",     sl);
   GlobalVariableSet(key + "_TP",     tp);
   GlobalVariableSet(key + "_SPREAD", (double)SymbolInfoInteger(sym, SYMBOL_SPREAD));
}

void LogTradeClose(ulong deal)
{
   string sym   = HistoryDealGetString(deal, DEAL_SYMBOL);
   string cmt   = HistoryDealGetString(deal, DEAL_COMMENT);
   string strat = (StringFind(cmt, "MRM_") == 0) ? StringSubstr(cmt, 4) : "";

   long   deal_type = HistoryDealGetInteger(deal, DEAL_TYPE);
   string direction = (deal_type == DEAL_TYPE_SELL) ? "BUY" : "SELL";

   double pnl_gross = HistoryDealGetDouble(deal, DEAL_PROFIT);
   double pnl_net   = pnl_gross
                    + HistoryDealGetDouble(deal, DEAL_COMMISSION)
                    + HistoryDealGetDouble(deal, DEAL_SWAP);
   double eq_after  = AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq_after > g_peak_equity) g_peak_equity = eq_after;

   // Clean up stored SL/TP meta
   string key = "MREG_OPEN_" + sym + "_" + strat;
   GlobalVariableDel(key + "_SL");
   GlobalVariableDel(key + "_TP");
   GlobalVariableDel(key + "_SPREAD");

   Print("[CLOSE] ", sym, " ", strat, " ", direction,
         " pnl=", NormalizeDouble(pnl_net, 2),
         " equity=", NormalizeDouble(eq_after, 2));
}

int OnInit()
{
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_FOK);
   g_peak_equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   g_equity_days   = 0;
   g_vol_scale     = 1.0;
   g_last_eq_date  = 0;
   g_last_vol_calc = 0;
   g_last_status   = 0;
   ArrayResize(g_daily_equity, 0);
   PrintFormat("[MRM] Init — vol_target=%.0f%%  lookback=%d days  magic_base=%d  kill=%s",
               InpVolTarget * 100.0, InpVolLookbackDays, InpBaseMagic,
               InpKillSwitch ? "ON" : "off");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {}

void OnTick()
{
   if(InpKillSwitch) { CloseAllPositions(); return; }

   UpdateDailyEquity();

   datetime now = TimeCurrent();
   if(now - g_last_vol_calc >= 3600) { g_last_vol_calc = now; RecalcVolScale(); }
   if(now - g_last_status   >= 3600)
   {
      g_last_status = now;
      PrintFormat("[STATUS] equity=%.2f  vol_scale=%.3f  positions=%d",
                  AccountInfoDouble(ACCOUNT_EQUITY), g_vol_scale, PositionsTotal());
   }

   // Close positions whose signal has been cleared
   int pi;
   for(pi = PositionsTotal() - 1; pi >= 0; pi--)
   {
      string sym   = PositionGetSymbol(pi);
      int    magic = (int)PositionGetInteger(POSITION_MAGIC);
      if(magic < InpBaseMagic || magic >= InpBaseMagic + 9973) continue;

      string cmt   = PositionGetString(POSITION_COMMENT);
      string strat = (StringFind(cmt, "MRM_") == 0) ? StringSubstr(cmt, 4) : "";
      if(strat == "") continue;

      SignalData sig = SignalRead(sym, strat);
      if(!sig.is_active || sig.direction == 0)
      {
         g_trade.SetExpertMagicNumber(magic);
         g_trade.PositionClose((ulong)PositionGetInteger(POSITION_TICKET));
      }
   }

   // Open new positions for active signals
   SignalData signals[];
   int count = SignalScanAll(signals);
   int i;
   for(i = 0; i < count; i++)
   {
      if(!signals[i].is_active || signals[i].direction == 0) continue;
      if(signals[i].sl_pips <= 0.0) continue;   // SL required for risk sizing

      string sym   = signals[i].symbol;
      string strat = signals[i].strategy;
      int    dir   = signals[i].direction;
      int    magic = MagicFor(strat);

      if(HasPosition(sym, magic)) continue;

      // Strategy may pass its own risk% via weight field (0 = use default)
      double risk_pct = (signals[i].weight > 0.0) ? signals[i].weight : InpDefaultRisk;
      double lots     = CalcLots(sym, signals[i].sl_pips, risk_pct);
      if(lots < SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN)) continue;

      g_trade.SetExpertMagicNumber(magic);
      string comment = "MRM_" + strat;
      double point   = SymbolInfoDouble(sym, SYMBOL_POINT);

      if(dir == 1)
      {
         double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
         double sl  = ask - signals[i].sl_pips * point;
         double tp  = (signals[i].tp_pips > 0.0) ? ask + signals[i].tp_pips * point : 0.0;
         if(g_trade.Buy(lots, sym, ask, sl, tp, comment))
         {
            PrintFormat("[OPEN] %s %s BUY lots=%.2f sl=%.0f pts  vol_scale=%.3f",
                        sym, strat, lots, signals[i].sl_pips, g_vol_scale);
            StoreOpenMeta(sym, strat, sl, tp);
         }
      }
      else if(dir == -1)
      {
         double bid = SymbolInfoDouble(sym, SYMBOL_BID);
         double sl  = bid + signals[i].sl_pips * point;
         double tp  = (signals[i].tp_pips > 0.0) ? bid - signals[i].tp_pips * point : 0.0;
         if(g_trade.Sell(lots, sym, bid, sl, tp, comment))
         {
            PrintFormat("[OPEN] %s %s SELL lots=%.2f sl=%.0f pts  vol_scale=%.3f",
                        sym, strat, lots, signals[i].sl_pips, g_vol_scale);
            StoreOpenMeta(sym, strat, sl, tp);
         }
      }
   }
}

void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest&     request,
                        const MqlTradeResult&      result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   ulong deal = trans.deal;
   if(!HistoryDealSelect(deal)) return;
   if(HistoryDealGetInteger(deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;
   int magic = (int)HistoryDealGetInteger(deal, DEAL_MAGIC);
   if(magic < InpBaseMagic || magic >= InpBaseMagic + 9973) return;
   LogTradeClose(deal);
}
