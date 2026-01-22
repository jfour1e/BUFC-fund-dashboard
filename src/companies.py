sector_designations = {
    # Index ETFs
    'PSCI': 'Index', 'PSCT': 'Index', 'PSCM': 'Index', 'PSCU': 'Index',
    'RSPG': 'Index', 'RSPF': 'Index', 'PSR': 'Index', 'KBWR': 'Index',
    'PSCC': 'Index', 'PSCH': 'Index',

    # Consumer Discretionary
    'MODG': 'Consumer Discretionary',   # Sporting goods / golf equipment
    'SKY' : 'Consumer Discretionary',   # Manufactured housing / RVs (consumer cyclical)

    # Financials
    'STEP': 'Financials',               # Investment banking & advisory

    # Health Care
    'ENSG': 'Health Care',              # Skilled nursing / post-acute care
    'EHC' : 'Health Care',              # Hospital & healthcare services
    'INMD': 'Health Care',              # Medical devices (aesthetics)
    'PRVA': 'Health Care',

    # Information Technology
    'NXT' : 'Information Technology',   # Grid software / energy management tech
    'CVLT': 'Information Technology',   # Enterprise backup & data management software

    # Industrials
    'AGCO': 'Industrials',              # Agricultural & heavy machinery
}

NORMALIZED_SECTOR_MAP = {
    # ETFs → sectors (explicit)
    'PSCI': 'Information Technology',
    'PSCT': 'Information Technology',
    'PSCM': 'Materials',
    'PSCU': 'Utilities',
    'RSPG': 'Energy',
    'RSPF': 'Financials',
    'PSR' : 'Real Estate',
    'PSCC': 'Consumer Discretionary',
    'PSCH': 'Health Care',
    # Consider adding an Industrials ETF

    # Stocks → normalized sectors
    'MODG': 'Consumer Discretionary',
    'SKY' : 'Consumer Discretionary',
    'STEP': 'Financials',

    'ENSG': 'Health Care',
    'EHC' : 'Health Care',
    'INMD': 'Health Care',
    'PRVA': 'Health Care',
    'NXT' : 'Information Technology',
    'CVLT': 'Information Technology',
    'AGCO': 'Industrials',
}

EXPECTED_RETURNS = {
    # Indices
    'PSCI': 0.0983, 'PSCT': 0.0896, 'PSCM': 0.1392, 'PSCU': 0.08,'RSPF': 0.1312,
    'PSR': 0.09577, 'PSCC': 0.1181, 'PSCH': 0.116,'RSPG': 0.14329, 

    #Stocks
    'NXT': 0.1000, 'STEP': 0.0905, 'ENSG': 0.14425, 'MODG': 0.0562, 'PRVA': 0.12,
    'AGCO': 0.0791, 'SKY': 0.124607, 'CVLT': 0.14991, 'EHC': 0.0849, 'INMD': 0.1276, 
}

# Update 12/31/2025
RUSSELL_SECTOR_WEIGHTS = {
    'Health Care'           : 0.1875,
    'Industrials'           : 0.1808,
    'Financials'            : 0.1723,
    'Information Technology': 0.1234,
    'Consumer Discretionary': 0.1088,
    'Real Estate'           : 0.059,
    'Energy'                : 0.0527,
    'Materials'             : 0.0403,
    'Utilities'             : 0.0341,
    'Communications'        : 0.0246,
    'Consumer Staples'      : 0.0165,
}

ETF_TICKERS = [
    'PSCI', 'PSCT', 'PSCM', 'PSCU', 'RSPG', 'RSPF', 'PSR', 'PSCC', 'PSCH'
]

STOCK_TICKERS = [
    'MODG', 'SKY', 'STEP', 'NXT', 'ENSG', 'EHC', 'INMD', 'PRVA',
    'AGCO', 'CVLT'
]

"""
COMPANIES TO SELL: 
Consumer discretionary 
WINA, PATK,  

Healthcare: 

Industrials: 
BMI, 

Information Technolog
MITK,

Real Estate 
PECO

Indices: 
KBWR
"""
