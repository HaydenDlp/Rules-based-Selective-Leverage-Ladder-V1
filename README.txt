README — Tax-Aware Ladder Monte Carlo

1) What you MUST change to run
- CSV path is set on LINE 15:
  CSV_PATH   = r"C:\...\your_file.csv"

- Your CSV must have these exact columns:
  • date   (daily trading dates; parseable by pandas)
  • SP500  (S&P 500 price/level; used for ATH + drawdown triggers)
  • return (daily simple return as a decimal; e.g. 0.01 = +1%)

- Data used begins at:
  START_DATE = "1957-01-02"


2) Inputs that actually change results

Monte Carlo sampling
- NUM_WINDOWS  (how many random windows)
- SEED         (changes which windows are sampled)
- Window length is randomized 15–30 years inside generate_windows()

Taxes + TLH (major impact)
- TAX_RATE
- DO_TLH
- TLH_TRIGGER
- TLH_COOLDOWN_DAYS

Ladder exposure limits (major impact)
- MAX_LEVER_FRAC  (max combined SSO + UPRO exposure as % of portfolio)
- MAX_SSO_FRAC    (max SSO exposure as % of portfolio)
