Place your Seeking Alpha portfolio export files here.

These are .xlsx files exported from Seeking Alpha for each brokerage account.
Each file should contain 4 sheets: Summary, Ratings, Holdings, Dividends.

Expected naming convention:
  {Account Name} YYYY-MM-DD.xlsx

Examples:
  Fidelity - Individual - Brokerage 2026-02-16.xlsx
  Fidelity - ROTH IRA 2026-02-16.xlsx

The sa_export_prefix in your ~/.threshold/config.yaml must match
the beginning of each filename for automatic account mapping.

IMPORTANT: These files contain personal financial data and are
excluded from git via .gitignore. They will NOT be committed.
