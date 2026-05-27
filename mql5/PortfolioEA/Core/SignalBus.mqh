#ifndef SIGNAL_BUS_MQH
#define SIGNAL_BUS_MQH

#define SIGNAL_STALE_SECONDS 300

struct SignalData
{
   string   symbol;
   string   strategy;
   int      direction;
   double   weight;
   double   sl_pips;
   double   tp_pips;
   datetime last_update;
   bool     is_active;
};

string SigKey(string symbol, string strategy, string field)
{
   return "MREG_" + symbol + "_" + strategy + "_" + field;
}

void SignalWrite(string symbol, string strategy, int direction,
                 double weight, double sl_pips, double tp_pips)
{
   GlobalVariableSet(SigKey(symbol, strategy, "DIR"),    (double)direction);
   GlobalVariableSet(SigKey(symbol, strategy, "WEIGHT"), weight);
   GlobalVariableSet(SigKey(symbol, strategy, "SL"),     sl_pips);
   GlobalVariableSet(SigKey(symbol, strategy, "TP"),     tp_pips);
   GlobalVariableSet(SigKey(symbol, strategy, "ACTIVE"), 1.0);
   GlobalVariableSet(SigKey(symbol, strategy, "TIME"),   (double)TimeCurrent());
}

void SignalClear(string symbol, string strategy)
{
   GlobalVariableSet(SigKey(symbol, strategy, "DIR"),    0.0);
   GlobalVariableSet(SigKey(symbol, strategy, "WEIGHT"), 0.0);
   GlobalVariableSet(SigKey(symbol, strategy, "ACTIVE"), 0.0);
}

SignalData SignalRead(string symbol, string strategy)
{
   SignalData s;
   s.symbol      = symbol;
   s.strategy    = strategy;
   s.direction   = 0;
   s.weight      = 0.0;
   s.sl_pips     = 0.0;
   s.tp_pips     = 0.0;
   s.last_update = 0;
   s.is_active   = false;

   string keyDir  = SigKey(symbol, strategy, "DIR");
   string keyTime = SigKey(symbol, strategy, "TIME");

   if(!GlobalVariableCheck(keyDir) || !GlobalVariableCheck(keyTime))
      return s;

   s.direction   = (int)GlobalVariableGet(keyDir);
   s.weight      = GlobalVariableGet(SigKey(symbol, strategy, "WEIGHT"));
   s.sl_pips     = GlobalVariableGet(SigKey(symbol, strategy, "SL"));
   s.tp_pips     = GlobalVariableGet(SigKey(symbol, strategy, "TP"));
   s.last_update = (datetime)GlobalVariableGet(keyTime);
   s.is_active   = (GlobalVariableGet(SigKey(symbol, strategy, "ACTIVE")) > 0.5)
                   && (TimeCurrent() - s.last_update <= SIGNAL_STALE_SECONDS);
   return s;
}

bool SignalIsStale(string symbol, string strategy)
{
   string keyTime = SigKey(symbol, strategy, "TIME");
   if(!GlobalVariableCheck(keyTime)) return true;
   datetime last = (datetime)GlobalVariableGet(keyTime);
   return (TimeCurrent() - last > SIGNAL_STALE_SECONDS);
}

int SignalScanAll(SignalData& result[])
{
   int    total       = GlobalVariablesTotal();
   string found_keys[];
   int    found_count = 0;
   int    i;

   for(i = 0; i < total; i++)
   {
      string name = GlobalVariableName(i);
      if(StringFind(name, "MREG_") != 0) continue;
      if(StringLen(name) < 9) continue;

      string suffix = StringSubstr(name, StringLen(name) - 4);
      if(suffix != "_DIR") continue;

      string inner = StringSubstr(name, 5, StringLen(name) - 9);
      int    sep   = StringFind(inner, "_");
      if(sep <= 0) continue;

      string sym   = StringSubstr(inner, 0, sep);
      string strat = StringSubstr(inner, sep + 1);
      if(StringLen(sym) == 0 || StringLen(strat) == 0) continue;

      bool dup = false;
      int  j;
      for(j = 0; j < found_count; j++)
      {
         if(found_keys[j] == inner) { dup = true; break; }
      }
      if(dup) continue;

      ArrayResize(found_keys, found_count + 1);
      found_keys[found_count] = inner;
      found_count++;
   }

   ArrayResize(result, found_count);
   int valid = 0;
   for(i = 0; i < found_count; i++)
   {
      int    sep   = StringFind(found_keys[i], "_");
      string sym   = StringSubstr(found_keys[i], 0, sep);
      string strat = StringSubstr(found_keys[i], sep + 1);
      result[valid] = SignalRead(sym, strat);
      valid++;
   }

   ArrayResize(result, valid);
   return valid;
}

#endif
