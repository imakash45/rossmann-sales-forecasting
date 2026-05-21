# ── Imports ───────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import os

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, '..', 'data', 'processed', 'train_cleaned.csv')
FEAT_PATH  = os.path.join(BASE_DIR, '..', 'data', 'processed', 'train_features.csv')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'src', 'model.pkl')

# ── Load data ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df_open = df[df['Open'] == 1].copy()
df_open['Year']  = df_open['Date'].dt.year
df_open['Month'] = df_open['Date'].dt.month
df_open['Week']  = df_open['Date'].dt.isocalendar().week.astype(int)

df_features = pd.read_csv(FEAT_PATH, low_memory=False)
df_features['Date'] = pd.to_datetime(df_features[['Year', 'Month', 'Day']])

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# ── Pre-compute KPIs ──────────────────────────────────────────────
total_sales      = df_open['Sales'].sum()
avg_daily_sales  = df_open['Sales'].mean()
total_customers  = df_open['Customers'].sum()
best_store       = df_open.groupby('Store')['Sales'].sum().idxmax()
best_store_sales = df_open.groupby('Store')['Sales'].sum().max()

# ── Pre-compute chart data ────────────────────────────────────────
monthly_sales = df_open.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(
    monthly_sales['Year'].astype(str) + '-' +
    monthly_sales['Month'].astype(str))
monthly_sales = monthly_sales.sort_values('Date')

store_perf = df_open.groupby('Store').agg(
    Total_Sales    = ('Sales', 'sum'),
    Avg_Sales      = ('Sales', 'mean'),
    Total_Customers= ('Customers', 'sum')
).reset_index().sort_values('Total_Sales', ascending=False)

promo_sales = df_open.groupby('Promo')['Sales'].mean().reset_index()
promo_sales['Promo'] = promo_sales['Promo'].map({0: 'No Promo', 1: 'Promo'})

dow_sales = df_open.groupby('DayOfWeek')['Sales'].mean().reset_index()
dow_sales['DayName'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ── Recursive Forecast Function ───────────────────────────────────
def forecast_future(store_id, weeks_ahead):
    store_data = df_features[
        df_features['Store'] == store_id
    ].copy().sort_values('Date')

    if len(store_data) == 0:
        return None, None

    feature_cols = [c for c in df_features.columns
                    if c not in ['Sales', 'Date']]

    last_date    = store_data['Date'].max()
    last_sales   = store_data['Sales'].values
    recent_sales = list(last_sales[-28:])
    future_dates = []
    future_preds = []
    days_ahead   = weeks_ahead * 7

    for day in range(1, days_ahead + 1):
        next_date = last_date + pd.Timedelta(days=day)

        if next_date.dayofweek == 6:
            continue

        last_row = store_data.iloc[-1].copy()

        last_row['Year']         = next_date.year
        last_row['Month']        = next_date.month
        last_row['Day']          = next_date.day
        last_row['Week']         = next_date.isocalendar()[1]
        last_row['DayOfWeek']    = next_date.dayofweek + 1
        last_row['DayOfYear']    = next_date.timetuple().tm_yday
        last_row['Quarter']      = (next_date.month - 1) // 3 + 1
        last_row['IsWeekend']    = 1 if next_date.dayofweek >= 5 else 0
        last_row['IsMonthStart'] = 1 if next_date.day == 1 else 0
        last_row['IsMonthEnd']   = 1 if next_date.day == \
            pd.Timestamp(next_date.year, next_date.month, 1).days_in_month else 0

        month = next_date.month
        if month in [12, 1, 2]:   last_row['Season'] = 3
        elif month in [3, 4, 5]:  last_row['Season'] = 1
        elif month in [6, 7, 8]:  last_row['Season'] = 2
        else:                      last_row['Season'] = 0

        last_row['Lag_7']  = recent_sales[-7]  if len(recent_sales) >= 7  else np.mean(recent_sales)
        last_row['Lag_14'] = recent_sales[-14] if len(recent_sales) >= 14 else np.mean(recent_sales)
        last_row['Lag_21'] = recent_sales[-21] if len(recent_sales) >= 21 else np.mean(recent_sales)
        last_row['Lag_28'] = recent_sales[-28] if len(recent_sales) >= 28 else np.mean(recent_sales)

        last_row['Rolling_Mean_7']  = np.mean(recent_sales[-7:])
        last_row['Rolling_Mean_14'] = np.mean(recent_sales[-14:])
        last_row['Rolling_Mean_28'] = np.mean(recent_sales[-28:])
        last_row['Rolling_Std_7']   = np.std(recent_sales[-7:])
        last_row['Rolling_Std_14']  = np.std(recent_sales[-14:])
        last_row['Rolling_Std_28']  = np.std(recent_sales[-28:])

        X_pred = pd.DataFrame([last_row[feature_cols]])
        pred   = model.predict(X_pred,
                               num_iteration=model.best_iteration)[0]
        pred   = max(0, pred)

        future_dates.append(next_date)
        future_preds.append(pred)

        recent_sales.append(pred)
        if len(recent_sales) > 28:
            recent_sales.pop(0)

    return future_dates, future_preds

# ── Colors ────────────────────────────────────────────────────────
COLORS = {
    'primary' : '#2196F3',
    'success' : '#4CAF50',
    'warning' : '#FF9800',
    'danger'  : '#F44336',
    'dark'    : '#263238',
    'light'   : '#F5F5F5'
}

# ── KPI Card ──────────────────────────────────────────────────────
def kpi_card(title, value, subtitle, color):
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, style={'color': '#666', 'fontSize': '13px'}),
            html.H3(value, style={'color': color, 'fontWeight': 'bold'}),
            html.P(subtitle, style={'color': '#999', 'fontSize': '12px', 'margin': '0'})
        ])
    ], style={'borderLeft': f'4px solid {color}', 'borderRadius': '8px'})

# ── App ───────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# ── Layout ────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Rossmann Sales Forecasting Dashboard",
                    style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
            html.P("Interactive analytics and ML-powered sales forecasting",
                   style={'color': '#666'})
        ])
    ], style={'padding': '20px 0 10px 0'}),

    html.Hr(),

    # KPIs
    dbc.Row([
        dbc.Col(kpi_card("Total Sales", f"€{total_sales/1e6:.0f}M",
                         "Jan 2013 — Jul 2015", COLORS['primary']), width=3),
        dbc.Col(kpi_card("Avg Daily Sales", f"€{avg_daily_sales:,.0f}",
                         "Per store per day", COLORS['success']), width=3),
        dbc.Col(kpi_card("Total Customers", f"{total_customers/1e6:.0f}M",
                         "Total visits", COLORS['warning']), width=3),
        dbc.Col(kpi_card("Best Store", f"Store {best_store}",
                         f"€{best_store_sales/1e6:.1f}M total",
                         COLORS['danger']), width=3),
    ], style={'marginBottom': '20px'}),

    # Trend + DOW
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Monthly Sales Trend"),
                dbc.CardBody([
                    dcc.Graph(figure=px.line(
                        monthly_sales, x='Date', y='Sales',
                        color='Year',
                        labels={'Sales': 'Total Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10, b=10)
                    ))
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Avg Sales by Day"),
                dbc.CardBody([
                    dcc.Graph(figure=px.bar(
                        dow_sales, x='DayName', y='Sales',
                        color='Sales',
                        color_continuous_scale='Blues',
                        labels={'Sales': 'Avg Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10, b=10),
                        showlegend=False
                    ))
                ])
            ])
        ], width=4),
    ], style={'marginBottom': '20px'}),

    # Store Performance + Promo
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Top 15 Stores by Total Sales"),
                dbc.CardBody([
                    dcc.Graph(figure=px.bar(
                        store_perf.head(15),
                        x='Store', y='Total_Sales',
                        color='Total_Sales',
                        color_continuous_scale='Greens',
                        labels={'Total_Sales': 'Total Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10, b=10)
                    ))
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Promo vs No Promo"),
                dbc.CardBody([
                    dcc.Graph(figure=px.bar(
                        promo_sales, x='Promo', y='Sales',
                        color='Promo',
                        color_discrete_map={
                            'No Promo': COLORS['danger'],
                            'Promo'   : COLORS['success']},
                        labels={'Sales': 'Avg Sales (€)'}
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        margin=dict(t=10, b=10),
                        showlegend=False
                    ))
                ])
            ])
        ], width=4),
    ], style={'marginBottom': '20px'}),

    # Actual vs Predicted Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sales Forecast — Select Store"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='store-dropdown',
                                options=[{'label': f'Store {i}', 'value': i}
                                         for i in sorted(df_open['Store'].unique())],
                                value=1,
                                clearable=False,
                                style={'marginBottom': '10px'}
                            )
                        ], width=3),
                        dbc.Col([
                            html.P("Select any store to see actual vs predicted sales",
                                   style={'color': '#666', 'marginTop': '8px'})
                        ], width=9)
                    ]),
                    dcc.Graph(id='forecast-chart')
                ])
            ])
        ], width=12)
    ], style={'marginBottom': '20px'}),

    # Future Forecast Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Future Sales Forecast — ML Powered"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Store:",
                                       style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='future-store-dropdown',
                                options=[{'label': f'Store {i}', 'value': i}
                                         for i in sorted(df_open['Store'].unique())],
                                value=1,
                                clearable=False,
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Weeks Ahead:",
                                       style={'fontWeight': 'bold'}),
                            dcc.Slider(
                                id='weeks-slider',
                                min=4, max=16, step=4,
                                value=8,
                                marks={4: '4 weeks', 8: '8 weeks',
                                       12: '12 weeks', 16: '16 weeks'},
                            )
                        ], width=5),
                        dbc.Col([
                            html.Br(),
                            dbc.Button("Generate Forecast",
                                       id='forecast-btn',
                                       color='primary',
                                       className='mt-1')
                        ], width=4)
                    ], style={'marginBottom': '15px'}),
                    dcc.Loading(
                        id='loading-forecast',
                        type='circle',
                        children=[dcc.Graph(id='future-forecast-chart')]
                    ),
                    html.Div(id='forecast-summary',
                             style={'marginTop': '10px'})
                ])
            ])
        ], width=12)
    ]),

    html.Br()

], fluid=True, style={'backgroundColor': COLORS['light']})


# ── Callback — Actual vs Predicted ───────────────────────────────
@app.callback(
    Output('forecast-chart', 'figure'),
    Input('store-dropdown', 'value')
)
def update_forecast(store_id):
    store_data = df_features[
        (df_features['Store'] == store_id) &
        (df_features['Date'] >= '2015-06-01')
    ].copy()

    if len(store_data) == 0:
        return go.Figure()

    feature_cols = [c for c in df_features.columns
                    if c not in ['Sales', 'Date']]

    preds = model.predict(store_data[feature_cols],
                          num_iteration=model.best_iteration)
    preds = np.clip(preds, 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=store_data['Date'], y=store_data['Sales'],
        name='Actual',
        line=dict(color=COLORS['primary'], width=2),
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=store_data['Date'], y=preds,
        name='Predicted',
        line=dict(color=COLORS['danger'], width=2, dash='dash'),
        mode='lines+markers', marker=dict(size=4)
    ))
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h'),
        xaxis_title='Date',
        yaxis_title='Sales (€)',
        margin=dict(t=10)
    )
    return fig


# ── Callback — Future Forecast ────────────────────────────────────
@app.callback(
    Output('future-forecast-chart', 'figure'),
    Output('forecast-summary', 'children'),
    Input('forecast-btn', 'n_clicks'),
    Input('future-store-dropdown', 'value'),
    Input('weeks-slider', 'value')
)
def update_future_forecast(n_clicks, store_id, weeks_ahead):
    try:
        historical = df_features[
            df_features['Store'] == store_id
        ].tail(60)[['Date', 'Sales']].copy()

        future_dates, future_preds = forecast_future(store_id, weeks_ahead)

        if future_dates is None:
            return go.Figure(), "No data available"

        forecast_df = pd.DataFrame({
            'Date'           : future_dates,
            'Predicted_Sales': future_preds
        })

        fig = go.Figure()

        # Historical
        fig.add_trace(go.Scatter(
            x=historical['Date'], y=historical['Sales'],
            name='Historical Sales',
            line=dict(color=COLORS['primary'], width=2),
            mode='lines+markers', marker=dict(size=4)
        ))

        # Future
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Predicted_Sales'],
            name='Future Forecast',
            line=dict(color=COLORS['danger'], width=2, dash='dash'),
            mode='lines+markers', marker=dict(size=4)
        ))

        # Upper band
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Sales'] * 1.10,
            line=dict(color=COLORS['danger'], width=0),
            showlegend=False
        ))

        # Lower band with fill
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted_Sales'] * 0.90,
            fill='tonexty',
            fillcolor='rgba(244,67,54,0.15)',
            line=dict(color=COLORS['danger'], width=0),
            showlegend=False
        ))

        # Forecast start line
        forecast_start = str(historical['Date'].max().date())
        fig.add_shape(
            type='line',
            x0=forecast_start, x1=forecast_start,
            y0=0, y1=1, yref='paper',
            line=dict(color='green', width=2, dash='dot')
        )
        fig.add_annotation(
            x=forecast_start, y=1, yref='paper',
            text='Forecast Start',
            showarrow=False, yshift=10,
            font=dict(color='green')
        )

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(orientation='h'),
            xaxis_title='Date',
            yaxis_title='Sales (€)',
            margin=dict(t=30),
            height=450
        )

        avg_pred   = forecast_df['Predicted_Sales'].mean()
        max_pred   = forecast_df['Predicted_Sales'].max()
        min_pred   = forecast_df['Predicted_Sales'].min()
        total_pred = forecast_df['Predicted_Sales'].sum()

        summary = dbc.Row([
            dbc.Col(kpi_card("Avg Daily Forecast", f"€{avg_pred:,.0f}",
                             f"Over {weeks_ahead} weeks",
                             COLORS['primary']), width=3),
            dbc.Col(kpi_card("Peak Day Forecast", f"€{max_pred:,.0f}",
                             "Highest predicted day",
                             COLORS['success']), width=3),
            dbc.Col(kpi_card("Lowest Day Forecast", f"€{min_pred:,.0f}",
                             "Lowest predicted day",
                             COLORS['warning']), width=3),
            dbc.Col(kpi_card("Total Forecast Revenue", f"€{total_pred:,.0f}",
                             f"Next {weeks_ahead} weeks total",
                             COLORS['danger']), width=3),
        ])

        return fig, summary

    except Exception as e:
        import traceback
        print("❌ ERROR:", traceback.format_exc())
        return go.Figure(), html.P(f"Error: {str(e)}",
                                    style={'color': 'red'})


# ── Run ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=8050)