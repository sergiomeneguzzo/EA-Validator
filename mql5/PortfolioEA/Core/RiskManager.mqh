#ifndef RISK_MANAGER_V2_MQH
#define RISK_MANAGER_V2_MQH

class CRiskManager
{
private:
   double m_min_scale;
   double m_max_scale;

public:
   CRiskManager(double min_scale = 0.05,
                double max_scale = 3.00)
   {
      m_min_scale = min_scale;
      m_max_scale = max_scale;
   }

   // Clamp lots to broker limits and volume step
   double GetFinalLots(string symbol, double base_lots)
   {
      if(base_lots <= 0.0) return base_lots;
      double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
      if(step > 0.0)
         base_lots = MathFloor(base_lots / step) * step;
      double mn = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double mx = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      return NormalizeDouble(MathMax(mn, MathMin(mx, base_lots)), 2);
   }
};

#endif
