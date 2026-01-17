### SECTION 1  Import all the libraries
import yfinance as yf
import pandas as pd
import numpy as np
from email.message import EmailMessage
import smtplib
from google.colab import userdata

!pip install mplfinance
import mplfinance as mpf

### SECTION 2 Create a list of stock tickers
tickers = [
    'A1M.AX', 'A2M.AX', 'A4N.AX', 'AAT.AX', 'AAV.AX', 'ABG.AX', 'ABP.AX', 'ACX.AX', 'ADH.AX', 'ADJ.AX',
    'AD8.AX', 'AEF.AX', 'AFG.AX', 'AFI.AX', 'AFL.AX', 'AFS.AX', 'AFT.AX', 'AIA.AX', 'ALD.AX', 'ALL.AX',
    'ALQ.AX', 'ALU.AX', 'ALX.AX', 'AMC.AX', 'AMP.AX', 'ANG.AX', 'ANZ.AX', 'APA.AX', 'APE.AX', 'APM.AX',
    'APO.AX', 'APS.AX', 'APT.AX', 'APX.AX', 'ARB.AX', 'ARG.AX', 'ARL.AX', 'ARU.AX', 'ASB.AX', 'ASS.AX',
    'AST.AX', 'ASX.AX', 'AUB.AX', 'AVE.AX', 'AVJ.AX', 'AVN.AX', 'AVP.AX', 'AVA.AX', 'AVZ.AX', 'AWC.AX',
    'AWV.AX', 'AWR.AX', 'AX1.AX', 'AXL.AX', 'AZL.AX', 'AZJ.AX', 'BAP.AX', 'BAR.AX', 'BBL.AX', 'BDR.AX',
    'BEN.AX', 'BER.AX', 'BGA.AX', 'BG8.AX', 'BHP.AX', 'BIG.AX', 'BKI.AX', 'BKL.AX', 'BKW.AX', 'BLD.AX',
    'BLS.AX', 'BME.AX', 'BOE.AX', 'BOL.AX', 'BOT.AX', 'BOQ.AX', 'BPS.AX', 'BPT.AX', 'BRG.AX', 'BRN.AX',
    'BSL.AX', 'BTT.AX', 'BWP.AX', 'BXB.AX', 'BYE.AX', 'CAJ.AX', 'CAL.AX', 'CAV.AX', 'CAR.AX', 'CBA.AX',
    'CBL.AX', 'CCL.AX', 'CCP.AX', 'CCT.AX', 'CDA.AX', 'CDP.AX', 'CEN.AX', 'CGF.AX', 'CGO.AX', 'CGR.AX',
    'CHC.AX', 'CHM.AX', 'CHN.AX', 'CIM.AX', 'CIP.AX', 'CKS.AX', 'CKF.AX', 'CL1.AX', 'CLW.AX', 'CMA.AX',
    'CMM.AX', 'CMW.AX', 'CNU.AX', 'COB.AX', 'COE.AX', 'COF.AX', 'COH.AX', 'COL.AX', 'COM.AX', 'CPU.AX',
    'CQR.AX', 'CRA.AX', 'CRN.AX', 'CRZ.AX', 'CSL.AX', 'CSR.AX', 'CTD.AX', 'CTX.AX', 'CUC.AX', 'CUV.AX',
    'CWN.AX', 'CWY.AX', 'DCN.AX', 'DEG.AX', 'DGL.AX', 'DHG.AX', 'DMP.AX', 'DOW.AX', 'DRR.AX', 'DSK.AX',
    'DUI.AX', 'DVL.AX', 'DXI.AX', 'DXS.AX', 'EBO.AX', 'EGR.AX', 'EGR.AX', 'EHL.AX', 'ELD.AX', 'ELL.AX',
    'EML.AX', 'EPR.AX', 'EQT.AX', 'ETH.AX', 'EUA.AX', 'EVN.AX', 'EVT.AX', 'FBU.AX', 'FCT.AX', 'FDC.AX',
    'FGG.AX', 'FGH.AX', 'FMG.AX', 'FNV.AX', 'FPH.AX', 'FPO.AX', 'FRE.AX', 'FXJ.AX', 'GCI.AX', 'GDI.AX',
    'GEM.AX', 'GJT.AX', 'GL1.AX', 'GMA.AX', 'GMG.AX', 'GNC.AX', 'GNE.AX', 'GOZ.AX', 'GPT.AX', 'GUD.AX',
    'GWA.AX', 'GXL.AX', 'HACK.AX', 'HAR.AX', 'HBL.AX', 'HLS.AX', 'HPI.AX', 'HUB.AX', 'HVN.AX', 'HXD.AX',
    'IAG.AX', 'IBD.AX', 'ICP.AX', 'ICU.AX', 'IDP.AX', 'IEL.AX', 'IFL.AX', 'IFT.AX', 'IGO.AX', 'ILM.AX',
    'ILU.AX', 'IMD.AX', 'INR.AX', 'IOP.AX', 'IOP.AX', 'IPA.AX', 'IPL.AX', 'IRE.AX', 'ISX.AX', 'ITD.AX',
    'IVZ.AX', 'JBH.AX', 'JHX.AX', 'KCN.AX', 'KGN.AX', 'KMD.AX', 'LAD.AX', 'LAS.AX', 'LCL.AX', 'LFS.AX',
    'LGI.AX', 'LHC.AX', 'LIC.AX', 'LND.AX', 'LNK.AX', 'LSF.AX', 'LTR.AX', 'LYC.AX', 'MAH.AX', 'MAP.AX',
    'MCY.AX', 'MEZ.AX', 'MFF.AX', 'MFG.AX', 'MGF.AX', 'MGR.AX', 'MIN.AX', 'MKL.AX', 'MLT.AX', 'MNF.AX',
    'MOQ.AX', 'MP1.AX', 'MPL.AX', 'MQG.AX', 'MSB.AX', 'MSL.AX', 'MTO.AX', 'MTS.AX', 'NAB.AX', 'NAN.AX',
    'NCM.AX', 'NEC.AX', 'NFG.AX', 'NHF.AX', 'NIC.AX', 'NSR.AX', 'NST.AX', 'NUF.AX', 'NWL.AX', 'NXT.AX',
    'OHE.AX', 'OML.AX', 'ORA.AX', 'ORG.AX', 'ORI.AX', 'OSH.AX', 'OVN.AX', 'OZL.AX', 'PAL.AX', 'PBH.AX',
    'PDL.AX', 'PFG.AX', 'PGG.AX', 'PGH.AX', 'PHL.AX', 'PIL.AX', 'PLS.AX', 'PME.AX', 'PMV.AX', 'PNI.AX',
    'PNV.AX', 'PPT.AX', 'PRN.AX', 'PTM.AX', 'PWN.AX', 'QAN.AX', 'QBE.AX', 'QUB.AX', 'RAM.AX', 'REA.AX',
    'REH.AX', 'RHC.AX', 'RIO.AX', 'RKN.AX', 'RMD.AX', 'RRL.AX', 'RSG.AX', 'RWC.AX', 'S32.AX', 'SAR.AX',
    'SCG.AX', 'SCP.AX', 'SDF.AX', 'SEK.AX', 'SGR.AX', 'SGT.AX', 'SHL.AX', 'SIO.AX', 'SIR.AX', 'SKC.AX',
    'SLK.AX', 'SOL.AX', 'SPK.AX', 'SRV.AX', 'SST.AX', 'STO.AX', 'STP.AX', 'SUL.AX', 'SUN.AX', 'SVW.AX',
    'SYD.AX', 'TAH.AX', 'TCL.AX', 'TGR.AX', 'TLS.AX', 'TLT.AX', 'TNE.AX', 'TPG.AX', 'TPW.AX', 'TWE.AX',
    'TYR.AX', 'VAP.AX', 'VGI.AX', 'VGR.AX', 'VGS.AX', 'VNT.AX', 'VUL.AX', 'VVR.AX', 'WBC.AX', 'WDS.AX',
    'WEB.AX', 'WES.AX', 'WFD.AX', 'WHA.AX', 'WPL.AX', 'WPR.AX', 'WOW.AX', 'WTC.AX', 'XRO.AX', 'YAL.AX',
    'ZEL.AX', 'ZIM.AX'
]

# Create an empty dictionary to store the dataframes
stock_data = {}

for ticker in tickers:
    try:
        # Download data for a single ticker with all columns
        data = yf.download(ticker, period='8y', interval='1wk', auto_adjust=False)

        # Check if the dataframe is empty
        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue

        # Add the data to the dictionary
        stock_data[ticker] = data
        print(f"Successfully downloaded data for {ticker}.")
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}. Skipping.")

print("Finished downloading all data.")


### Section 3: Clean and Prepare Data ---
print("\n--- Starting Data Cleaning ---")
cleaned_data = {}
for ticker, data in stock_data.items():

    # Use a try...except block to handle potential formatting errors
    try:
        # Step 1: Handle Multi-Level Columns
        # This will remove the 'Price' and 'Ticker' levels
        data.columns = data.columns.droplevel(level=1)
        data.columns = data.columns.droplevel(level=2)
    except IndexError:
        # If there's no multi-level index, the droplevel will fail, which is okay.
        pass

    # Step 2: Reset the index to make 'Date' a column
    data = data.reset_index()

    # Step 3: Rename Columns for Consistency
    # We rename 'Adj Close' to 'Close' for consistent analysis
    data = data.rename(columns={'Adj Close': 'Close'})

    # Store the cleaned data in a new dictionary
    cleaned_data[ticker] = data

print("--- Finished Data Cleaning ---")

### Section 4:  Print the head and tail of a specific stock to inspect the data

try:
    print("\n--- Displaying data for XRO.AX ---")
    print("\nHead of DataFrame:")
    print(stock_data['XRO.AX'].head())

    print("\nTail of DataFrame:")
    print(stock_data['XRO.AX'].tail())

except KeyError:
    print("\nXRO.AX data was not found in the dictionary.")

### Section 5: Analyze Each Stock and Generate Alerts --- IMPORTANT - THE ANALYSIS ! 
print("\n--- Starting Analysis ---")
for ticker, data in stock_data.items():
    print(f"\nAnalyzing {ticker}...")

    # *** FIX: Flatten the index to avoid KeyErrors ***
    # This ensures a simple numeric index for all operations.
    try:
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=0, drop=True)
        data = data.reset_index()
    except Exception as e:
        print(f"Error resetting index for {ticker}: {e}. Skipping.")
        continue

    # Ensure the dataframe has enough data for the 200-week MA
    if len(data) < 60:
        print(f"Not enough data for {ticker} for the 200-week MA. Skipping.")
        continue

    # Using a 5-week (Short) and 55-week (Long) Golden Cross strategy
    data['SMA_Short'] = data['Close'].rolling(window=5).mean()
    data['SMA_Long'] = data['Close'].rolling(window=55).mean()

#    Create Crossover Signals based on these new names
    data.loc[(data['SMA_Short'].shift(1) < data['SMA_Long'].shift(1)) & (data['SMA_Short'] > data['SMA_Long']), 'Signal'] = 1


    # Calculate Moving Averages
    #data['SMA20'] = data['Close'].rolling(window=15).mean()
    #data['SMA200'] = data['Close'].rolling(window=60).mean()

    # Create Crossover Signals
    #data['Signal'] = 0.0
    #data.loc[(data['SMA20'].shift(1) < data['SMA200'].shift(1)) & (data['SMA20'] > data['SMA200']), 'Signal'] = 1
    #data.loc[(data['SMA20'].shift(1) > data['SMA200'].shift(1)) & (data['SMA20'] < data['SMA200']), 'Signal'] = -1

    # Create Bearish and Bullish Engulfing Signals
    if 'Open' in data.columns and 'Close' in data.columns:
        data['EngulfingSignal'] = 0.0

        # Bearish Engulfing
        prev_bullish = data['Close'].shift(1) > data['Open'].shift(1)
        curr_bearish = data['Close'] < data['Open']
        engulfs_bearish = (data['Open'] > data['Close'].shift(1)) & (data['Close'] < data['Open'].shift(1))
        data.loc[prev_bullish & curr_bearish & engulfs_bearish, 'EngulfingSignal'] = -1

        # Bullish Engulfing
        prev_bearish = data['Close'].shift(1) < data['Open'].shift(1)
        curr_bullish = data['Close'] > data['Open']
        engulfs_bullish = (data['Open'] < data['Close'].shift(1)) & (data['Close'] > data['Open'].shift(1))
        data.loc[prev_bearish & curr_bullish & engulfs_bullish, 'EngulfingSignal'] = 1

    # Combine Signals for a Confirmed Alert
    if 'EngulfingSignal' in data.columns:
        data['CombinedSignal'] = 0.0
        data.loc[(data['Signal'] == -1) & (data['EngulfingSignal'] == -1), 'CombinedSignal'] = -1
        data.loc[(data['Signal'] == 1) & (data['EngulfingSignal'] == 1), 'CombinedSignal'] = 1

    # --- NEW FEATURE: Find and print the last confirmed signal ---
    # reverse the DataFrame and find the first non-zero signal
    signals_found = data[data['CombinedSignal'] != 0]
    if not signals_found.empty:
        last_signal_data = signals_found.iloc[-1]
        last_signal_date = last_signal_data['Date'].strftime('%Y-%m-%d')
        last_signal_type = "Buy" if last_signal_data['CombinedSignal'] == 1 else "Sell"
        print(f"The last confirmed signal for {ticker} was a {last_signal_type} signal on {last_signal_date}.")
    else:
        print(f"No confirmed signals found in the history of {ticker}.")

    # Check for an alert on the last row
    last_row = data.iloc[-1]

    if 'CombinedSignal' in data.columns and last_row['CombinedSignal'] != 0.0:
        signal_type = "Buy" if last_row['CombinedSignal'] == 1 else "Sell"
        alert_body = f"A confirmed {signal_type} signal has been detected for {ticker} as of {last_row['Date'].strftime('%Y-%m-%d')}."
        print(alert_body)

        # Uncomment the line below to send the email
        # send_notification(f"Confirmed {signal_type} Signal for {ticker}", alert_body)
    else:
        print(f"No new confirmed signal detected for {ticker}.")

            # At the end of the loop, add this conditional statement
    if ticker == 'BHP.AX':
        print(f"\n--- Final DataFrame for {ticker} ---")
        print(data.head(200))
        print("---------------------------------")

    print("-----------------------------------")

print("\n--- Analysis Complete! ---")


### Section 6 --- REUSABLE SIGNAL FUNCTION ---
def apply_strategy_logic(df):
    # 1. Ensure 'Date' is a column if it's currently the index
    if 'Date' not in df.columns:
        df = df.reset_index()
    
    # 2. Fix potential Multi-Index columns (common in new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 3. Calculate MAs
    # Note: Weekly intervals, so 15 weeks (~4 months) and 60 weeks (~1 year)
    df['SMA20'] = df['Close'].rolling(window=15).mean()
    df['SMA200'] = df['Close'].rolling(window=60).mean()
    
    # 4. Generate Crossover Signal
    df['Signal'] = 0.0
    # 1.0 when SMA20 is above SMA200 (Bullish), 0.0 otherwise
    df['Signal'] = np.where(df['SMA20'] > df['SMA200'], 1.0, 0.0)
    
    # 5. Calculate Returns
    df['Market_Return'] = df['Close'].pct_change()
    # Shift position by 1 day so we trade on the NEXT opening
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Market_Return']
    
    # 6. Calculate Cumulative Growth
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    
    return df

# SECTION 7 Evaluate a single stock --- XERO EVALUATION FOR EXAMPLE ---
ticker_to_test = 'XRO.AX'

if ticker_to_test in stock_data:
    # Get fresh copy
    xero_df = stock_data[ticker_to_test].copy()
    xero_df = apply_strategy_logic(xero_df)
    
    # Remove rows with NaN from the moving averages to start the plot cleanly
    xero_df = xero_df.dropna(subset=['SMA200'])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    
    # We use .values if 'Date' is giving issues, but the reset_index above should fix it
    plt.plot(xero_df['Date'], xero_df['Cumulative_Market'], label='Buy & Hold Xero', color='gray', alpha=0.5)
    plt.plot(xero_df['Date'], xero_df['Cumulative_Strategy'], label='Golden Cross Strategy', color='green', linewidth=2)
    
    plt.title(f"{ticker_to_test} Performance: Strategy vs Buy & Hold")
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    final_strategy = (xero_df['Cumulative_Strategy'].iloc[-1] - 1) * 100
    final_market = (xero_df['Cumulative_Market'].iloc[-1] - 1) * 100
    print(f"Xero Strategy Final Return: {final_strategy:.2f}%")
    print(f"Xero Buy & Hold Return: {final_market:.2f}%")

### SECTION 8 Look for the raw crossovers (SMA 5 > SMA 55)
crossovers = xero_df[xero_df['Signal'] != 0]

if not crossovers.empty:
    print("--- 5/55 Crossover Events for Xero ---")
    print(crossovers[['Date', 'Close', 'Signal']])
else:
    print("Still no crossovers found. Check if SMA_Short and SMA_Long are being calculated correctly.")



# SECTION 9 - VIZUALISE the chosen stock - XERO AS EXAMPLE
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download & Clean fresh data
ticker = 'XRO.AX'
data = yf.download(ticker, period='8y', interval='1wk', auto_adjust=False)

# Flatten columns (yfinance fix)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data = data.reset_index()

# 2. Strategy Logic (5/55 Weekly)
data['SMA_Short'] = data['Close'].rolling(window=5).mean()
data['SMA_Long'] = data['Close'].rolling(window=55).mean()

# Position: 1 if Short > Long, else 0
data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1.0, 0.0)
data['Action'] = data['Signal'].diff()

# 3. Create the Visualization
plt.figure(figsize=(16, 8))

# Plot Price and Averages
plt.plot(data['Date'], data['Close'], label='XRO.AX Price', color='black', alpha=0.15)
plt.plot(data['Date'], data['SMA_Short'], label='5-Week SMA (Fast)', color='blue', linewidth=1.5)
plt.plot(data['Date'], data['SMA_Long'], label='55-Week SMA (Slow)', color='orange', linewidth=1.5)

# 4. Plot Arrows
# Buy (Action = 1)
plt.plot(data.loc[data['Action'] == 1, 'Date'], 
         data.loc[data['Action'] == 1, 'SMA_Short'], 
         '^', markersize=12, color='green', label='Buy Signal', lw=0)

# Sell (Action = -1)
plt.plot(data.loc[data['Action'] == -1, 'Date'], 
         data.loc[data['Action'] == -1, 'SMA_Short'], 
         'v', markersize=12, color='red', label='Sell Signal', lw=0)

plt.title('Xero (XRO.AX): Optimized 5/55 Weekly Golden Cross', fontsize=16)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# 5. Final Audit Trail
last_action = "BUY" if data['Signal'].iloc[-1] == 1 else "SELL/CASH"
print(f"Current Strategy Position: {last_action}")

### Section 10 - SHOW the Trade Log Generator -- list the historic signals Date and prices with a % outcome
# 1. Identify where the signal changes (Entry/Exit points)
xero_df['Trade_Action'] = xero_df['Signal'].diff()

# 2. Filter for only the weeks where a trade occurred
trade_events = xero_df[xero_df['Trade_Action'] != 0].copy()

# 3. Create the Log Table
trade_log = []
for i in range(1, len(trade_events)):
    # If the previous action was 1 (Buy) and current is -1 (Sell)
    if trade_events.iloc[i-1]['Trade_Action'] == 1:
        entry_date = trade_events.iloc[i-1]['Date']
        entry_price = trade_events.iloc[i-1]['Close']
        exit_date = trade_events.iloc[i]['Date']
        exit_price = trade_events.iloc[i]['Close']
        
        profit_loss = (exit_price / entry_price) - 1
        
        trade_log.append({
            'Entry Date': entry_date.strftime('%Y-%m-%d'),
            'Exit Date': exit_date.strftime('%Y-%m-%d'),
            'Entry Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'P&L %': round(profit_loss * 100, 2)
        })

# 4. Display as a Clean Table
df_log = pd.DataFrame(trade_log)
print("--- XERO STRATEGY TRADE LOG ---")
print(df_log)

# 5. Summary Statistics
print(f"\nTotal Trades: {len(df_log)}")
print(f"Average Profit per Trade: {df_log['P&L %'].mean():.2f}%")
print(f"Best Trade: {df_log['P&L %'].max():.2f}%")

## Section 11 - Answer the question of what is the "optimaal" MA for this stock UES XERO AS EXAMPLE
def optimize_windows(df, short_range, long_range):
    results = []
    
    # 1. Clean data once
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Iterate through all combinations
    for s in short_range:
        for l in long_range:
            if s >= l: continue # Short must be shorter than long
            
            temp_df = df.copy()
            temp_df['S_SMA'] = temp_df['Close'].rolling(window=s).mean()
            temp_df['L_SMA'] = temp_df['Close'].rolling(window=l).mean()
            
            # Simple Return Logic
            temp_df['Sig'] = np.where(temp_df['S_SMA'] > temp_df['L_SMA'], 1.0, 0.0)
            temp_df['Ret'] = temp_df['Sig'].shift(1) * temp_df['Close'].pct_change()
            final_ret = (1 + temp_df['Ret']).cumprod().iloc[-1]
            
            results.append({'Short': s, 'Long': l, 'Return': final_ret})
            
    return pd.DataFrame(results)

# Run the optimizer for Xero
# Testing Short (5 to 15 weeks) and Long (30 to 60 weeks)
opt_results = optimize_windows(stock_data['XRO.AX'], range(5, 16), range(30, 61))

# Find the winner
winner = opt_results.loc[opt_results['Return'].idxmax()]
print(f"BEST WINDOWS: Short={winner['Short']}, Long={winner['Long']} with {winner['Return']:.2f}x growth")
