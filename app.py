import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="Zerodha Trading Analytics", 
    layout="wide",
    page_icon="📊"
)

# Title
st.title("📊 Zerodha Trading Performance Dashboard")
st.markdown("""
Advanced analytics platform for Zerodha trading data with performance metrics, risk analysis, and trader insights.
""")

# Load data from CSV
@st.cache_data
def load_data():
    # Replace with your actual CSV file path
    df = pd.read_csv('Zerodha_Trading_Behavior_Dataset.csv')  # Update with your file path
    
    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Trade_Value'] = df['Quantity'] * df['Price']
    
    # Calculate additional metrics
    df['Absolute_PL'] = abs(df['Profit_Loss'])
    df['Win_Loss'] = np.where(df['Profit_Loss'] > 0, 'Win', 'Loss')
    
    # Create random time data if not available (for demo purposes)
    if 'Time' not in df.columns:
        np.random.seed(42)
        times = pd.to_datetime([
            f"{np.random.randint(9, 15)}:{np.random.randint(0, 60):02d}:00" 
            for _ in range(len(df))
        ]).time
        df['Time'] = times
    
    return df

try:
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range selector
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Other filters
    selected_users = st.sidebar.multiselect(
        "Select Traders",
        options=df['User_ID'].unique(),
        default=df['User_ID'].unique()
    )
    
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks",
        options=df['Stock'].unique(),
        default=df['Stock'].unique()
    )
    
    selected_trade_types = st.sidebar.multiselect(
        "Select Trade Types",
        options=df['Trade_Type'].unique(),
        default=df['Trade_Type'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df['User_ID'].isin(selected_users)) &
        (df['Stock'].isin(selected_stocks)) &
        (df['Trade_Type'].isin(selected_trade_types)) &
        (df['Date'] >= date_range[0]) &
        (df['Date'] <= date_range[1])
    ].copy()
    
    # KPI Cards
    st.header("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(filtered_df)
        st.metric("Total Trades", f"{total_trades:,}")
    
    with col2:
        total_volume = filtered_df['Quantity'].sum()
        st.metric("Total Volume", f"{total_volume:,} shares")
    
    with col3:
        net_pl = filtered_df['Profit_Loss'].sum()
        st.metric("Net P&L", f"₹{net_pl:,.2f}", 
                 delta_color="inverse" if net_pl < 0 else "normal")
    
    with col4:
        win_rate = len(filtered_df[filtered_df['Profit_Loss'] > 0]) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Overview", 
        "💹 Stock Analysis", 
        "👤 Trader Performance", 
        "⚙️ Advanced Metrics"
    ])
    
    with tab1:
        st.header("Trading Activity Overview")
        
        # Time series of trades and P&L
        fig = go.Figure()
        
        # Add trade count
        trade_counts = filtered_df.groupby('Date').size().reset_index(name='Count')
        fig.add_trace(go.Bar(
            x=trade_counts['Date'],
            y=trade_counts['Count'],
            name='Trade Count',
            marker_color='#636EFA'
        ))
        
        # Add P&L line
        daily_pl = filtered_df.groupby('Date')['Profit_Loss'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=daily_pl['Date'],
            y=daily_pl['Profit_Loss'],
            name='Daily P&L',
            yaxis='y2',
            line=dict(color='#FFA15A')
        ))
        
        fig.update_layout(
            title='Daily Trading Activity & P&L',
            yaxis=dict(title='Trade Count'),
            yaxis2=dict(
                title='P&L (₹)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade type distribution with P&L
            trade_type_pl = filtered_df.groupby('Trade_Type').agg({
                'User_ID': 'count',
                'Profit_Loss': 'sum'
            }).reset_index()
            
            fig = px.bar(trade_type_pl, x='Trade_Type', y='User_ID',
                        color='Profit_Loss',
                        title="Trade Type Distribution with P&L",
                        labels={'User_ID': 'Trade Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Buy/Sell distribution with P&L
            action_pl = filtered_df.groupby('Action').agg({
                'User_ID': 'count',
                'Profit_Loss': 'sum'
            }).reset_index()
            
            fig = px.bar(action_pl, x='Action', y='User_ID',
                        color='Profit_Loss',
                        title="Buy/Sell Distribution with P&L",
                        labels={'User_ID': 'Trade Count'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Stock-wise Analysis")
        
        # Stock performance heatmap
        stock_metrics = filtered_df.groupby('Stock').agg({
            'Profit_Loss': ['sum', 'mean', 'count'],
            'Quantity': 'sum'
        }).reset_index()
        
        stock_metrics.columns = ['Stock', 'Total_PL', 'Avg_PL', 'Trade_Count', 'Total_Volume']
        
        fig = px.treemap(stock_metrics, 
                        path=['Stock'], 
                        values='Trade_Count',
                        color='Total_PL',
                        color_continuous_scale='RdYlGn',
                        title="Stock Performance Heatmap (Size=Trades, Color=P&L)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Stock correlation matrix
        st.subheader("Stock Correlation Analysis")
        
        # Pivot to get daily P&L by stock
        daily_stock_pl = filtered_df.pivot_table(
            index='Date',
            columns='Stock',
            values='Profit_Loss',
            aggfunc='sum'
        ).fillna(0)
        
        # Calculate correlation matrix
        corr_matrix = daily_stock_pl.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(title="Stock P&L Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Trader Performance Analysis")
        
        # Trader leaderboard
        st.subheader("Trader Leaderboard")
        
        trader_stats = filtered_df.groupby('User_ID').agg({
            'Profit_Loss': ['sum', 'mean', 'count'],
            'Absolute_PL': 'sum',
            'Win_Loss': lambda x: (x == 'Win').mean() * 100
        }).reset_index()
        
        trader_stats.columns = [
            'User_ID', 'Total_PL', 'Avg_PL', 'Trade_Count', 
            'Absolute_PL', 'Win_Rate'
        ]
        
        # Calculate risk-reward ratio
        trader_stats['Risk_Reward'] = trader_stats['Absolute_PL'] / trader_stats['Total_PL'].abs()
        
        # Display interactive table
        st.dataframe(
            trader_stats.sort_values('Total_PL', ascending=False).reset_index(drop=True),
            column_config={
                "Total_PL": st.column_config.NumberColumn(
                    "Total P&L",
                    format="₹%.2f"
                ),
                "Avg_PL": st.column_config.NumberColumn(
                    "Avg P&L per Trade",
                    format="₹%.2f"
                ),
                "Win_Rate": st.column_config.NumberColumn(
                    "Win Rate %",
                    format="%.1f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Trader performance scatter plot
        st.subheader("Trader Performance Analysis")
        
        fig = px.scatter(trader_stats, 
                        x='Trade_Count', 
                        y='Total_PL',
                        size='Absolute_PL',
                        color='Win_Rate',
                        hover_name='User_ID',
                        title="Trade Count vs. Total P&L (Size=Risk, Color=Win Rate)")
        
        fig.update_layout(
            yaxis_title="Total P&L (₹)",
            xaxis_title="Number of Trades"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Advanced Trading Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L Distribution
            st.subheader("P&L Distribution")
            
            fig = px.histogram(filtered_df, x='Profit_Loss',
                              nbins=50,
                              title="Distribution of Profit/Loss per Trade")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk-Reward Analysis
            st.subheader("Risk-Reward Analysis")
            
            winning_trades = filtered_df[filtered_df['Profit_Loss'] > 0]
            losing_trades = filtered_df[filtered_df['Profit_Loss'] < 0]
            
            avg_win = winning_trades['Profit_Loss'].mean()
            avg_loss = losing_trades['Profit_Loss'].mean()
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=avg_win,
                title={"text": "Average Win (₹)"},
                domain={'row':0, 'column':0}
            ))
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=abs(avg_loss),
                title={"text": "Average Loss (₹)"},
                domain={'row':0, 'column':1}
            ))
            
            fig.add_trace(go.Indicator(
                mode="number",
                value=abs(avg_win/avg_loss) if avg_loss != 0 else 0,
                title={"text": "Risk-Reward Ratio"},
                domain={'row':1, 'column':0}
            ))
            
            fig.update_layout(
                grid={'rows':2, 'columns':2, 'pattern':"independent"},
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade Timing Analysis - Fixed version
        st.subheader("Trade Timing Analysis")
        
        # Create hour column from Time if it exists
        if 'Time' in filtered_df.columns:
            # Convert time to datetime to extract hour
            filtered_df['Hour'] = pd.to_datetime(filtered_df['Time'].astype(str)).dt.hour
            
            # Group by hour and calculate metrics
            hourly_stats = filtered_df.groupby('Hour').agg({
                'User_ID': 'count',
                'Profit_Loss': 'sum',
                'Profit_Loss': lambda x: (x > 0).mean() * 100  # Win rate
            }).reset_index()
            
            hourly_stats.columns = ['Hour', 'Trade_Count', 'Total_PL', 'Win_Rate']
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add trade count bars
            fig.add_trace(go.Bar(
                x=hourly_stats['Hour'],
                y=hourly_stats['Trade_Count'],
                name='Trade Count',
                marker_color='#636EFA'
            ))
            
            # Add win rate line
            fig.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['Win_Rate'],
                name='Win Rate (%)',
                yaxis='y2',
                line=dict(color='#00CC96')
            ))
            
            # Add P&L line
            fig.add_trace(go.Scatter(
                x=hourly_stats['Hour'],
                y=hourly_stats['Total_PL'],
                name='Total P&L (₹)',
                yaxis='y3',
                line=dict(color='#FFA15A')
            ))
            
            fig.update_layout(
                title='Trading Activity by Hour of Day',
                xaxis=dict(title='Hour of Day'),
                yaxis=dict(title='Trade Count'),
                yaxis2=dict(
                    title='Win Rate (%)',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                yaxis3=dict(
                    title='P&L (₹)',
                    overlaying='y',
                    side='right',
                    anchor='free',
                    position=1.1
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Time data not available in the dataset")

except FileNotFoundError:
    st.error("Error: CSV file not found. Please ensure 'zerodha_trading_data.csv' is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
**Note:** This is a demo dashboard for Zerodha trading analytics. 
For accurate results, ensure your CSV file contains the required columns: 
User_ID, Date, Stock, Action, Quantity, Price, Trade_Type, Profit_Loss.
""")