import pandas as pd
import numpy as np
import plotly.graph_objects as go

def load_and_prepare_data(file_name):
    df = pd.read_csv(file_name, sep=';', encoding='iso-8859-1', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    def clean_val(v):
        if pd.isna(v) or str(v).strip() in ['D', 'C', 'W', 'B']: return 0.0
        return float(str(v).replace('.', '').replace(',', '.'))
    df['Bedrag_num'] = df['Bedrag'].apply(clean_val).abs()
    df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True, errors='coerce')
    df_omzet = df[df['Grootboekrekening Code'].astype(str).str.startswith('8')].copy()
    return df_omzet.dropna(subset=['Datum'])

# 1. Data laden
df_clean = load_and_prepare_data("Prognoee Peter .csv")
monthly_hist = df_clean.resample('ME', on='Datum')['Bedrag_num'].sum().reset_index()

# 2. Seizoensgebonden Forecast berekenen
# We kijken naar het gemiddelde van elke maand (januari, februari, etc.) in de historie
monthly_hist['month_num'] = monthly_hist['Datum'].dt.month
seasonal_index = monthly_hist.groupby('month_num')['Bedrag_num'].mean()
overall_mean = monthly_hist['Bedrag_num'].mean()

# Bereken de 'growth trend' (stijgt of daalt de omzet over de jaren heen?)
x_idx = np.arange(len(monthly_hist))
trend_fit = np.polyfit(x_idx, monthly_hist['Bedrag_num'], 1)
trend_model = np.poly1d(trend_fit)

# 3. Voorspelling voor 2026 bouwen
future_dates = pd.date_range(start='2026-01-01', periods=12, freq='ME')
forecast_values = []

for i, date in enumerate(future_dates):
    # Basis trend waarde
    trend_val = trend_model(len(monthly_hist) + i)
    # Vermenigvuldig met de seizoensfactor van die specifieke maand
    month = date.month
    season_factor = seasonal_index[month] / overall_mean if overall_mean != 0 else 1
    forecast_values.append(max(0, trend_val * season_factor))

forecast_df = pd.DataFrame({'Datum': future_dates, 'Bedrag_num': forecast_values})

# 4. Interactief Dashboard
fig = go.Figure()
fig.add_trace(go.Bar(x=monthly_hist['Datum'], y=monthly_hist['Bedrag_num'], name='Historie', marker_color='#3498db'))
fig.add_trace(go.Bar(x=forecast_df['Datum'], y=forecast_df['Bedrag_num'], name='Forecast 2026 (Seizoensgebonden)', marker_color='#e67e22'))

fig.update_layout(
    title="Geavanceerde Omzet Prognose 2026 (met seizoenspatronen)",
    yaxis=dict(title="Omzet (€)", tickprefix="€", tickformat=",.0f"),
    template="plotly_white",
    hovermode="x unified"
)

fig.show()
