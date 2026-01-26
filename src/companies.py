"""
Dictionaries to designate current holdings sector 
and expected return mappings. 
Russell 2000 sector weights with last update date. 
"""

# Updated 12/31/2025
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

# Updated 01/22/2026
HOLDINGS_INFO = {
    # ETFs / Indices
    'PSCI': {'name': 'Invesco S&P SmallCap Industrials ETF', 'expected_return': 0.10, 'expense_ratio': 0.0029, 'asset_type': 'index', 'sector': 'Industrials'},
    'PSCT': {'name': 'Invesco S&P SmallCap Information Technology ETF', 'expected_return': 0.10, 'expense_ratio': 0.0029, 'asset_type': 'index', 'sector': 'Information Technology'},
    'PSCM': {'name': 'Invesco S&P SmallCap Materials ETF', 'expected_return': 0.10, 'expense_ratio': 0.0029, 'asset_type': 'index', 'sector': 'Materials'},
    'PSCU': {'name': 'Invesco S&P SmallCap Utilities ETF', 'expected_return': 0.10, 'expense_ratio': 0.0029, 'asset_type': 'index', 'sector': 'Utilities'},
    'RSPG': {'name': 'Invesco S&P 500 Equal Weight Energy ETF', 'expected_return': 0.10, 'expense_ratio': 0.004, 'asset_type': 'index', 'sector': 'Energy'},
    'RSPF': {'name': 'Invesco S&P 500 Equal Weight Financials ETF', 'expected_return': 0.10, 'expense_ratio': 0.004, 'asset_type': 'index', 'sector': 'Financials'},
    'PSR' : {'name': 'Invesco Active U.S. Real Estate ETF', 'expected_return': 0.10, 'expense_ratio': 0.0055,'asset_type': 'index', 'sector': 'Real Estate'},
    'PSCC': {'name': 'Invesco S&P SmallCap Consumer Discretionary ETF', 'expected_return': 0.10, 'expense_ratio': 0.0029, 'asset_type': 'index', 'sector': 'Consumer Discretionary'},
    'PSCH': {'name': 'Invesco S&P SmallCap Health Care ETF', 'expected_return': 0.10, 'expense_ratio': 0.0029, 'asset_type': 'index', 'sector': 'Health Care'},

    # Stocks
    'CALY': {'name': 'Callaway Golf / Topgolf Callaway Brands', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Consumer Discretionary'},
    'SKY' : {'name': 'Champion Homes', 'expected_return': 0.15, 'asset_type': 'stock', 'sector': 'Consumer Discretionary'},
    'STEP': {'name': 'StepStone Group', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Financials'},
    'ELLO': {'name': 'Ellomay Capital', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Utilities'},

    'ENSG': {'name': 'The Ensign Group', 'expected_return': 0.25, 'asset_type': 'stock', 'sector': 'Health Care'},
    'EHC' : {'name': 'Encompass Health', 'expected_return': 0.30, 'asset_type': 'stock', 'sector': 'Health Care'},
    'INMD': {'name': 'InMode Ltd.', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Health Care'},
    'PRVA': {'name': 'Privia Health', 'expected_return': 0.30, 'asset_type': 'stock', 'sector': 'Health Care'},

    'NXT' : {'name': 'Nextracker', 'expected_return': 0.25, 'asset_type': 'stock', 'sector': 'Information Technology'},
    'CVLT': {'name': 'Commvault Systems', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Information Technology'},
    'AUR' : {'name': 'Aurora Innovation', 'expected_return': 0.25, 'asset_type': 'stock', 'sector': 'Information Technology'},
    'NVTS': {'name': 'Navitas Semiconductor', 'expected_return': 0.25, 'asset_type': 'stock', 'sector': 'Information Technology'},
    'LASR': {'name': 'nLIGHT', 'expected_return': 0.25, 'asset_type': 'stock', 'sector': 'Information Technology'},

    'AGCO': {'name': 'AGCO Corporation', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Industrials'},
    'ACHR': {'name': 'Archer Aviation', 'expected_return': 0.35, 'asset_type': 'stock', 'sector': 'Industrials'},
    #'FLY' : {'name': 'Flywire', 'expected_return': 0.30, 'asset_type': 'stock', 'sector': 'Industrials'},
    'TTC' : {'name': 'Toro Company', 'expected_return': 0.12, 'asset_type': 'stock', 'sector': 'Industrials'},

    'UAA' : {'name': 'Under Armour', 'expected_return': 0.20, 'asset_type': 'stock', 'sector': 'Consumer Discretionary'},
}