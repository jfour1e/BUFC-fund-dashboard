sector_designations = {
    # Index ETFs
    'PSCI': 'Index', 'PSCT': 'Index', 'PSCM': 'Index', 'PSCU': 'Index',
    'RSPG': 'Index', 'RSPF': 'Index', 'PSR': 'Index', 'KBWR': 'Index',
    'PSCC': 'Index', 'PSCH': 'Index',

    # Consumer Discretionary
    'MODG': 'Consumer Discretionary', 'WINA': 'Consumer Discretionary',
    'SKY': 'Consumer Discretionary', 'PATK': 'Consumer Discretionary',

    # Financials
    'STEP': 'Financials',

    # Health Care
    'ENSG': 'Health Care', 'EHC': 'Health Care', 'INMD': 'Health Care',

    # Industrials
    'AGCO': 'Industrials', 'BMI': 'Industrials',

    # Information Technology
    'MITK': 'Information Technology', 'NXT': 'Information Technology',
    'PUBM': 'Information Technology', 'CVLT': 'Information Technology',

    # Real Estate
    'PECO': 'Real Estate'
}

companies = [
    'PSCI', 'PSCT', 'PSCM', 'PSCU', 'RSPG', 'RSPF',
    'MITK', 'NXT', 'PATK', 'PECO', 'PUBM', 'STEP',
    'ENSG', 'MODG', 'WINA', 'AGCO', 'BMI', 'SKY',
    'CVLT', 'EHC', 'INMD', 'PSR', 'KBWR', 'PSCC', 'PSCH'
]

ETF_TICKERS = [
    'PSCI', 'PSCT', 'PSCM', 'PSCU', 'RSPG', 'RSPF', 'PSR', 'KBWR', 'PSCC', 'PSCH'
]

STOCK_TICKERS = [t for t in companies if t not in ETF_TICKERS]

NORMALIZED_SECTOR_MAP = {
    # ETFs → sectors (explicit)
    'PSCI': 'Information Technology',
    'PSCT': 'Information Technology',
    'PSCM': 'Materials',
    'PSCU': 'Utilities',
    'RSPG': 'Energy',
    'RSPF': 'Financials',
    'PSR' : 'Real Estate',
    'KBWR': 'Financials',
    'PSCC': 'Consumer Discretionary',
    'PSCH': 'Health Care',

    # Stocks → normalized sectors
    'MITK': 'Information Technology',
    'NXT' : 'Information Technology',
    'PATK': 'Industrials',
    'PECO': 'Real Estate',
    'PUBM': 'Information Technology',
    'STEP': 'Financials',
    'ENSG': 'Health Care',
    'MODG': 'Consumer Discretionary',
    'WINA': 'Consumer Discretionary',
    'AGCO': 'Industrials',
    'BMI' : 'Industrials',
    'SKY' : 'Consumer Discretionary',
    'CVLT': 'Information Technology',
    'EHC' : 'Health Care',
    'INMD': 'Health Care',
}

RUSSELL_SECTOR_WEIGHTS = {
    'Communications'        : 0.05,
    'Consumer Discretionary': 0.12,
    'Consumer Staples'      : 0.04,
    'Energy'                : 0.06,
    'Financials'            : 0.15,
    'Health Care'           : 0.13,
    'Industrials'           : 0.10,
    'Information Technology': 0.17,
    'Materials'             : 0.05,
    'Utilities'             : 0.03,
    'Real Estate'           : 0.10,
}

EXPECTED_RETURNS = {
    'PSCI': 0.0983, 'PSCT': 0.0896, 'PSCM': 0.1392, 'PSCU': 0.08, 
    'RSPG': 0.14329, 'RSPF': 0.1312, 'MITK': 0.1243, 'NXT': 0.1000, 
    'PATK': 0.1045, 'PECO': 0.133, 'PUBM': 0.10245, 'STEP': 0.0905, 
    'ENSG': 0.14425, 'MODG': 0.0862, 'WINA': 0.1167, 'AGCO': 0.1091, 
    'BMI': 0.08571, 'SKY': 0.124607, 'CVLT': 0.14991, 'EHC': 0.0849, 
    'INMD': 0.1276, 'PSR': 0.09577, 'KBWR': 0.12168, 'PSCC': 0.1181, 
    'PSCH': 0.116
    }
