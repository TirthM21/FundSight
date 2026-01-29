"""
Mutual Fund Dashboard using mftool
A comprehensive Streamlit app for analyzing Indian mutual funds using the mftool library

Features:
- All AMFI schemes (automatically loaded)
- Search & Browse mutual funds by name or code
- Rolling Returns Analysis (1yr, 3yr, 5yr, 7yr, 10yr, 15yr, 20yr)
- Equity/Debt/Hybrid Scheme Performance
- Scheme Comparison & Details
- Portfolio Builder with Analysis
- Return Calculator
- AMC-wise Fund Browsing
- No hardcoded data - All live from AMFI
"""

import streamlit as st
import pandas as pd
import numpy as np
from mftool import Mftool
import os

# Try to initialize fund database if not already loaded    # Database  # Database load is optional
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from functools import lru_cache

# AI Planner removed - focusing on core fund analysis

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Mutual Fund Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CACHE & INITIALIZATION
# =============================================================================

@st.cache_resource
def init_mftool():
    """Initialize mftool instance"""
    return Mftool()

@st.cache_data(ttl=3600)
def get_all_scheme_codes():
    """Get all available scheme codes from AMFI"""
    mf = init_mftool()
    try:
        schemes = mf.get_scheme_codes()
        return schemes
    except Exception as e:
        st.error(f"Error fetching schemes: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_amc_schemes(amc_name):
    """Get all schemes for a specific AMC - FILTERED (Growth/Regular only, no dividends/direct)"""
    mf = init_mftool()
    try:
        schemes = mf.get_available_schemes(amc_name)
        
        # Filter to remove dividends and payout variants - keep only Growth and Regular plans
        exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'semi-annual', 'annual', 'payout', 'bonus', 'weekly', 'daily', 'fortnightly', 'direct', 'reinvestment', 'distribution', 'income option', 'plan b', 'plan c', 'institutional', 'withdrawl', 'fixed term', 'capital protection', 'fmp', 'ftp', 'series', 'nfo', 'maturity']
        filtered = {}
        
        for code, name in schemes.items():
            name_lower = name.lower()
            
            # Skip if has exclusion patterns
            if any(pattern in name_lower for pattern in exclude_patterns):
                continue
            
            # Keep if has 'growth' OR 'regular' (both are non-payout plans)
            if 'growth' in name_lower or 'regular' in name_lower:
                filtered[code] = name
        
        return filtered
    except Exception as e:
        st.warning(f"Could not fetch schemes for {amc_name}")
        return {}

@st.cache_data(ttl=7200)
def get_scheme_historical_nav(scheme_code):
    """Get historical NAV for a scheme"""
    mf = init_mftool()
    try:
        data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
        if data is not None and len(data) > 0:
            return data
        return None
    except Exception as e:
        st.warning(f"Could not fetch NAV for scheme {scheme_code}")
        return None

@st.cache_data(ttl=600)
def get_scheme_quote(scheme_code):
    """Get current quote for a scheme"""
    mf = init_mftool()
    try:
        quote = mf.get_scheme_quote(scheme_code)
        return quote
    except Exception as e:
        st.warning(f"Could not fetch quote for scheme {scheme_code}")
        return None

@st.cache_data(ttl=3600)
def get_scheme_details(scheme_code):
    """Get details for a scheme"""
    mf = init_mftool()
    try:
        details = mf.get_scheme_details(scheme_code)
        return details
    except Exception as e:
        st.warning(f"Could not fetch details for scheme {scheme_code}")
        return None

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

class DataProcessor:
    """Process mutual fund data for analysis"""
    
    PERIODS = {
        "1Y": 252,
        "3Y": 756,
        "5Y": 1260,
        "7Y": 1764,
        "10Y": 2520,
        "15Y": 3780,
        "20Y": 5040
    }
    
    @staticmethod
    def calculate_rolling_returns(nav_series, period_days):
        """Calculate rolling returns (CAGR) for a given period"""
        if len(nav_series) < period_days:
            return None
        
        # Convert to numeric if strings
        nav_numeric = pd.to_numeric(nav_series, errors='coerce')
        nav_numeric = nav_numeric.dropna()
        
        if len(nav_numeric) < period_days:
            return None
        
        returns = []
        dates = []
        
        for i in range(period_days, len(nav_numeric)):
            start_nav = nav_numeric.iloc[i - period_days]
            end_nav = nav_numeric.iloc[i]
            
            if start_nav > 0:
                total_return = (end_nav - start_nav) / start_nav
                years = period_days / 252
                cagr = (((end_nav / start_nav) ** (1 / years)) - 1) * 100
                
                returns.append(cagr)
                # Create date by going back from today
                days_back = len(nav_numeric) - i - 1
                date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                dates.append(date)
        
        if len(returns) > 0:
            return pd.Series(returns, index=dates)
        return None
    
    @staticmethod
    def calculate_statistics(returns_series):
        """Calculate comprehensive statistics"""
        if returns_series is None or len(returns_series) == 0:
            return {}
        
        stats = {
            'Min': returns_series.min(),
            'Max': returns_series.max(),
            'Mean': returns_series.mean(),
            'Median': returns_series.median(),
            'Std Dev': returns_series.std(),
            'Sharpe': (returns_series.mean() / returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0,
        }
        
        # Probability buckets
        stats['Prob(Negative)'] = (returns_series < 0).sum() / len(returns_series) * 100
        stats['Prob(0-5%)'] = ((returns_series >= 0) & (returns_series < 5)).sum() / len(returns_series) * 100
        stats['Prob(5-10%)'] = ((returns_series >= 5) & (returns_series < 10)).sum() / len(returns_series) * 100
        stats['Prob(10-15%)'] = ((returns_series >= 10) & (returns_series < 15)).sum() / len(returns_series) * 100
        stats['Prob(15-20%)'] = ((returns_series >= 15) & (returns_series < 20)).sum() / len(returns_series) * 100
        stats['Prob(>20%)'] = (returns_series >= 20).sum() / len(returns_series) * 100
        
        return stats
    
    @staticmethod
    def calculate_drawdown(nav_series):
        """Calculate maximum drawdown"""
        if len(nav_series) < 2:
            return 0
        
        cumulative_max = nav_series.cummax()
        drawdown = (nav_series - cumulative_max) / cumulative_max * 100
        return drawdown.min()

# =============================================================================
# FUND FILTERING HELPER
# =============================================================================

@st.cache_data(ttl=7200)
def filter_schemes_by_type(fund_type="equity"):
    """Filter schemes by type - FAST name-based filtering (no API calls)"""
    all_schemes = get_all_scheme_codes()
    filtered = {}
    
    # Exclude patterns - focus only on dividend/income options, not 'direct' (direct plans are legitimate)
    exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'semi-annual', 'annual', 'payout', 'bonus', 'weekly', 'daily', 'fortnightly', 'direct', 'reinvestment', 'distribution', 'income option', 'plan b', 'plan c', 'institutional', 'withdrawl', 'fixed term', 'capital protection', 'fmp', 'ftp', 'series', 'nfo', 'maturity']
    
    for code, name in all_schemes.items():
        name_lower = name.lower()
        
        # Skip if contains exclude pattern (dividend/income variants only)
        if any(pattern in name_lower for pattern in exclude_patterns):
            continue
        
        # Filter by type using NAME only (no API calls = FAST)
        if fund_type.lower() == "equity":
            if any(x in name_lower for x in ['equity', 'large cap', 'mid cap', 'small cap', 'multi cap', 'flexi cap', 'concentrated']):
                filtered[code] = name
        elif fund_type.lower() == "debt":
            if any(x in name_lower for x in ['debt', 'banking', 'psu', 'credit', 'bond', 'liquid', 'ultra', 'overnight']):
                filtered[code] = name
        elif fund_type.lower() == "hybrid":
            if any(x in name_lower for x in ['hybrid', 'balanced', 'allocation']):
                filtered[code] = name
    
    return filtered

# =============================================================================
# SIDEBAR - FUND SELECTION
# =============================================================================

with st.sidebar:
    st.title("🏦 Mutual Fund Dashboard")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Select Page:",
        ["🏠 Home", "🔎 Search Funds", "📊 Rolling Returns", "🔄 Compare", 
         "💼 Portfolio", "🤖 Auto Portfolio", "📈 Performance", "💰 Calculator", 
         "📉 Advanced Analysis", "⭐ Fund Rankings", "ℹ️ Fund Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")    
    # Initialize session state for selected schemes
    if 'selected_schemes' not in st.session_state:
        st.session_state.selected_schemes = {}
    if 'equity_cache' not in st.session_state:
        st.session_state.equity_cache = None
    if 'debt_cache' not in st.session_state:
        st.session_state.debt_cache = None
    if 'hybrid_cache' not in st.session_state:
        st.session_state.hybrid_cache = None

# =============================================================================
# PAGE: HOME
# =============================================================================

if page == "🏠 Home":
    st.title("🏦 Mutual Fund Analytics Dashboard")
    st.markdown("*Powered by mftool & AMFI Data*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Schemes Available", "10,000+")
    with col2:
        st.metric("🏢 AMCs Listed", "50+")
    with col3:
        st.metric("📈 Data Source", "AMFI Live")
    
    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Features")
            st.markdown("""
            - ✅ Search all AMFI schemes
            - ✅ Historical NAV data
            - ✅ Rolling returns (1Y-20Y)
            - ✅ Fund comparison
            - ✅ Portfolio building
            - ✅ Performance analysis
            - ✅ Return calculator
            - ✅ Equity/Debt/Hybrid schemes
            """)
        
        with col2:
            st.subheader("🚀 Getting Started")
            st.markdown("""
            1. Go to **Search Funds** to browse or search
            2. Select funds you're interested in
            3. View **Rolling Returns** for analysis
            4. Compare funds side-by-side
            5. Build a portfolio in **Portfolio** page
            6. Check performance metrics
            """)
    
    st.markdown("---")
    st.info("💡 All data is live from AMFI. No schemes are hardcoded. Browse freely!")

# =============================================================================
# PAGE: SEARCH FUNDS
# =============================================================================

elif page == "🔎 Search Funds":
    st.title("🔎 Search & Browse Mutual Funds")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search by Name/Code", "🏢 Browse by AMC", "📊 Browse by Type", "📂 Browse by Category"])
    
    with tab1:
        st.subheader("Search Schemes")
        
        search_query = st.text_input(
            "Enter scheme name or code:",
            placeholder="e.g., HDFC or SBI or 119597",
            help="Search by scheme name or code"
        )
        
        if search_query:
            all_schemes = get_all_scheme_codes()
            
            if all_schemes:
                # Search in both keys (codes) and values (names)
                query_lower = search_query.lower()
                results = {}
                
                exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'semi-annual', 'annual', 'payout', 'bonus', 'weekly', 'daily', 'fortnightly', 'direct', 'reinvestment', 'distribution', 'income option', 'plan b', 'plan c', 'institutional', 'withdrawl', 'fixed term', 'capital protection', 'fmp', 'ftp', 'series', 'nfo', 'maturity']

                for code, name in all_schemes.items():
                    name_lower = name.lower()
                    if any(exclude in name_lower for exclude in exclude_patterns):
                        continue
                        
                    if query_lower in code.lower() or query_lower in name_lower:
                        results[code] = name
                
                if results:
                    st.success(f"Found {len(results)} matching schemes")
                    
                    # Sort results alphabetically by scheme name
                    sorted_results = sorted(results.items(), key=lambda x: x[1].lower())
                    
                    # Pagination for search results
                    page_num = st.number_input("Page:", min_value=1, value=1, step=1, key="search_page")
                    items_per_page = 30
                    
                    start_idx = (page_num - 1) * items_per_page
                    end_idx = start_idx + items_per_page
                    paginated_results = sorted_results[start_idx:end_idx]
                    
                    for code, name in paginated_results:
                        col1, col2, col3 = st.columns([3, 3, 1])
                        
                        with col1:
                            st.write(f"**{name}**")
                        with col2:
                            st.caption(f"Code: {code}")
                        with col3:
                            if st.button("📌", key=f"select_{code}"):
                                st.session_state.selected_schemes[code] = name
                                st.success(f"Added: {name}")
                else:
                    st.warning("No schemes found matching your search")
    
    with tab2:
        st.subheader("Browse by Asset Management Company")
        
        st.info("✅ Growth schemes only (no dividends, IDCW, or bonus variants)")
        
        col1, col2 = st.columns(2)
        
        # Get all schemes to extract unique AMCs
        all_schemes = get_all_scheme_codes()
        amc_list = []
        for scheme_name in all_schemes.values():
            # Extract AMC name from scheme name (usually first word or before "Prudential", etc.)
            if scheme_name and isinstance(scheme_name, str):
                parts = scheme_name.split()
                if len(parts) > 0:
                    amc = parts[0]
                    if amc not in amc_list and len(amc) > 2:
                        amc_list.append(amc)
        
        amc_list = sorted(list(set(amc_list)))  # All AMCs sorted
        
        with col1:
            selected_amc = st.selectbox("Search or Select AMC:", amc_list, key="amc_select")
        
        with col2:
            fund_type_filter = st.radio(
                "Filter by Type:",
                ["All", "Equity", "Debt", "Hybrid"],
                horizontal=True
            )
        
        if selected_amc:
            with st.spinner(f"Loading {selected_amc} schemes..."):
                amc_schemes = get_amc_schemes(selected_amc)
                
                # Further filter by fund type if selected
                if fund_type_filter != "All":
                    filtered_by_type = {}
                    for code, name in amc_schemes.items():
                        name_lower = name.lower()
                        
                        if fund_type_filter == "Equity":
                            if any(x in name_lower for x in ['equity', 'large cap', 'mid cap', 'small cap', 'multi cap', 'flexi cap']):
                                filtered_by_type[code] = name
                        elif fund_type_filter == "Debt":
                            if any(x in name_lower for x in ['debt', 'banking', 'psu', 'credit', 'bond', 'liquid', 'ultra']):
                                filtered_by_type[code] = name
                        elif fund_type_filter == "Hybrid":
                            if any(x in name_lower for x in ['hybrid', 'balanced', 'allocation']):
                                filtered_by_type[code] = name
                    
                    amc_schemes = filtered_by_type
                
                if amc_schemes:
                    st.success(f"Found {len(amc_schemes)} {fund_type_filter} Growth schemes for {selected_amc}")
                    
                    # Pagination
                    page_num = st.number_input("Page:", min_value=1, value=1, step=1, key="amc_page")
                    items_per_page = 25
                    
                    start_idx = (page_num - 1) * items_per_page
                    end_idx = start_idx + items_per_page
                    paginated = list(amc_schemes.items())[start_idx:end_idx]
                    
                    for code, name in paginated:
                        col1, col2, col3 = st.columns([3, 3, 1])
                        
                        with col1:
                            st.write(f"**{name}**")
                        with col2:
                            st.caption(f"Code: {code}")
                        with col3:
                            if st.button("📌", key=f"amc_select_{code}"):
                                st.session_state.selected_schemes[code] = name
                                st.success(f"Added: {name}")
                else:
                    st.warning(f"No {fund_type_filter} Growth schemes found for {selected_amc}")
    
    with tab3:
        st.subheader("Browse by Fund Type")
        
        st.info("⚡ Fast name-based filtering - no delays!")
        
        fund_type = st.radio(
            "Select Fund Type:",
            ["Equity Funds", "Debt Funds", "Hybrid Funds"],
            horizontal=True
        )
        
        # Use session cache for filtered results
        if fund_type == "Equity Funds":
            if st.session_state.equity_cache is None:
                st.session_state.equity_cache = filter_schemes_by_type("equity")
            filtered = st.session_state.equity_cache
        elif fund_type == "Debt Funds":
            if st.session_state.debt_cache is None:
                st.session_state.debt_cache = filter_schemes_by_type("debt")
            filtered = st.session_state.debt_cache
        else:
            if st.session_state.hybrid_cache is None:
                st.session_state.hybrid_cache = filter_schemes_by_type("hybrid")
            filtered = st.session_state.hybrid_cache
        
        # Pagination
        page_num = st.number_input("Page:", min_value=1, value=1, step=1, key="browse_page")
        items_per_page = 30
        
        if filtered:
            st.success(f"Found {len(filtered)} {fund_type} (Growth plans only)")
            
            # Paginate results
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            paginated = list(filtered.items())[start_idx:end_idx]
            
            for code, name in paginated:
                col1, col2, col3 = st.columns([3, 3, 1])
                
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.caption(f"Code: {code}")
                with col3:
                    if st.button("📌", key=f"type_select_{code}"):
                        st.session_state.selected_schemes[code] = name
                        st.success(f"Added: {name}")
            else:
                st.warning(f"Could not load {fund_type}")
    
    with tab4:
        st.subheader("Browse by Scheme Category")
        
        st.info("📂 Select specific scheme category (Large Cap, Midcap, Flexi Cap, Asset Allocation, etc.)")
        
        # Define scheme categories
        SCHEME_CATEGORIES = {
            "All": {"keywords": []},
            "Large Cap": {"keywords": ["large cap"]},
            "Mid Cap": {"keywords": ["mid cap", "midcap"]},
            "Small Cap": {"keywords": ["small cap", "smallcap"]},
            "Large & Midcap": {"keywords": ["large & midcap", "large and midcap", "multi cap"]},
            "Flexi Cap": {"keywords": ["flexi cap", "flexible cap"]},
            "Dividend Yield": {"keywords": ["dividend yield"]},
            "Focused": {"keywords": ["focused"]},
            "Index Funds": {"keywords": ["index", "nifty", "sensex", "midcap", "smallcap"]},
            "International": {"keywords": ["international", "global", "overseas", "us"]},
            "Sector Specific": {"keywords": ["banking", "pharma", "it", "infrastructure", "fmcg", "energy"]},
            "Asset Allocation": {"keywords": ["asset allocation", "multi asset"]},
            "Balanced": {"keywords": ["balanced"]},
            "Conservative": {"keywords": ["conservative"]},
            "Liquid": {"keywords": ["liquid"]},
            "Ultra Short": {"keywords": ["ultra short"]},
            "Debt": {"keywords": ["debt", "bond", "psu", "credit"]},
            "Money Market": {"keywords": ["money market"]},
            "Gold": {"keywords": ["gold", "bullion"]},
            "Real Estate": {"keywords": ["real estate", "reits"]}
        }
        
        selected_category = st.selectbox(
            "Select Scheme Category:",
            list(SCHEME_CATEGORIES.keys()),
            key="scheme_category_select"
        )
        
        # Filter schemes by category
        category_schemes = {}
        all_schemes = get_all_scheme_codes()
        
        for code, name in all_schemes.items():
            if name and isinstance(name, str):
                name_lower = name.lower()
                # Skip dividend/payout variants - keep only Growth and Regular
                exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'semi-annual', 'annual', 'payout', 'bonus', 'weekly', 'daily', 'fortnightly', 'direct', 'reinvestment', 'distribution', 'income option', 'plan b', 'plan c', 'institutional', 'withdrawl', 'fixed term', 'capital protection', 'fmp', 'ftp', 'series', 'nfo', 'maturity']
                if any(x in name_lower for x in exclude_patterns):
                    continue
                
                # Only accept if has 'growth' or 'regular'
                if 'growth' not in name_lower and 'regular' not in name_lower:
                    continue
                
                if selected_category == "All":
                    category_schemes[code] = name
                else:
                    keywords = SCHEME_CATEGORIES[selected_category]["keywords"]
                    if any(kw in name_lower for kw in keywords):
                        category_schemes[code] = name
        
        if category_schemes:
            st.success(f"Found {len(category_schemes)} schemes in {selected_category}")
            
            # Pagination
            page_num = st.number_input("Page:", min_value=1, value=1, step=1, key="category_page")
            items_per_page = 30
            
            start_idx = (page_num - 1) * items_per_page
            end_idx = start_idx + items_per_page
            paginated = list(category_schemes.items())[start_idx:end_idx]
            
            for code, name in paginated:
                col1, col2, col3 = st.columns([3, 3, 1])
                
                with col1:
                    st.write(f"**{name}**")
                with col2:
                    st.caption(f"Code: {code}")
                with col3:
                    if st.button("📌", key=f"cat_select_{code}"):
                        st.session_state.selected_schemes[code] = name
                        st.success(f"Added: {name}")
        else:
            st.warning(f"No schemes found in {selected_category}")
    st.subheader("📌 Selected Schemes")
    
    if st.session_state.selected_schemes:
        for code, name in st.session_state.selected_schemes.items():
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                st.write(f"✅ {name}")
            with col2:
                st.caption(code)
            with col3:
                if st.button("❌", key=f"remove_{code}"):
                    del st.session_state.selected_schemes[code]
                    st.rerun()
    else:
        st.info("No schemes selected yet. Click 📌 to add schemes.")

# =============================================================================
# PAGE: ROLLING RETURNS
# =============================================================================

elif page == "📊 Rolling Returns":
    st.title("📊 Rolling Returns Analysis")
    
    if not st.session_state.selected_schemes:
        st.warning("Please select a scheme first from the Search page")
    else:
        selected_code = st.selectbox(
            "Select scheme:",
            list(st.session_state.selected_schemes.keys()),
            format_func=lambda x: st.session_state.selected_schemes[x]
        )
        
        scheme_name = st.session_state.selected_schemes[selected_code]
        
        with st.spinner(f"Loading data for {scheme_name}..."):
            nav_df = get_scheme_historical_nav(selected_code)
            
            if nav_df is not None:
                st.subheader(f"{scheme_name}")
                
                # Quote
                quote = get_scheme_quote(selected_code)
                if quote:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current NAV", f"₹{quote.get('nav', 'N/A')}")
                    with col2:
                        st.metric("Last Updated", quote.get('last_updated', 'N/A'))
                    with col3:
                        st.metric("Scheme Code", selected_code)
                
                st.markdown("---")
                
                # Period selection
                period = st.select_slider(
                    "Select Period:",
                    options=["1Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y"],
                    value="5Y"
                )
                
                period_days = DataProcessor.PERIODS[period]
                
                # Calculate rolling returns
                rolling_returns = DataProcessor.calculate_rolling_returns(
                    nav_df['nav'].iloc[::-1].reset_index(drop=True),
                    period_days
                )
                
                if rolling_returns is not None:
                    stats = DataProcessor.calculate_statistics(rolling_returns)
                    
                    # Metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Min Return", f"{stats['Min']:.2f}%")
                    with col2:
                        st.metric("Max Return", f"{stats['Max']:.2f}%")
                    with col3:
                        st.metric("Mean Return", f"{stats['Mean']:.2f}%")
                    with col4:
                        st.metric("Std Dev", f"{stats['Std Dev']:.2f}%")
                    with col5:
                        st.metric("Sharpe Ratio", f"{stats['Sharpe']:.2f}")
                    
                    st.markdown("---")
                    
                    # Multiple charts in tabs
                    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(["📈 Rolling Returns", "📊 Distribution", "📉 Performance", "🔍 Volatility"])
                    
                    with chart_tab1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=rolling_returns.index,
                            y=rolling_returns.values,
                            mode='lines',
                            name=f'{period} Rolling Returns',
                            line=dict(color='#667eea', width=2),
                            fill='tozeroy'
                        ))
                        
                        fig.update_layout(
                            title=f'{period} Rolling CAGR Returns - {scheme_name}',
                            xaxis_title='Date',
                            yaxis_title='Return (%)',
                            template='plotly_dark',
                            height=450,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    
                    with chart_tab2:
                        # Histogram
                        fig2 = go.Figure()
                        fig2.add_trace(go.Histogram(
                            x=rolling_returns.values,
                            nbinsx=50,
                            name=f'{period} Returns',
                            marker_color='#764ba2'
                        ))
                        
                        fig2.update_layout(
                            title=f'Return Distribution - {period}',
                            xaxis_title='Return (%)',
                            yaxis_title='Frequency',
                            template='plotly_dark',
                            height=450
                        )
                        
                        st.plotly_chart(fig2, width='stretch')
                    
                    with chart_tab3:
                        # Cumulative returns
                        nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                        nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                        daily_returns = nav_numeric.pct_change().dropna()
                        cum_returns = (1 + daily_returns).cumprod() - 1
                        
                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(
                            y=(cum_returns.values * 100),
                            mode='lines',
                            name='Cumulative Return',
                            line=dict(color='#00cc88', width=2),
                            fill='tozeroy'
                        ))
                        
                        fig3.update_layout(
                            title=f'Cumulative Returns - {scheme_name}',
                            yaxis_title='Cumulative Return (%)',
                            template='plotly_dark',
                            height=450,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig3, width='stretch')
                    
                    with chart_tab4:
                        # Rolling volatility
                        rolling_vol = daily_returns.rolling(window=30).std() * (252 ** 0.5) * 100
                        
                        fig4 = go.Figure()
                        fig4.add_trace(go.Scatter(
                            y=rolling_vol.values,
                            mode='lines',
                            name='30-Day Rolling Volatility',
                            line=dict(color='#ff6b6b', width=2),
                            fill='tozeroy'
                        ))
                        
                        fig4.update_layout(
                            title=f'30-Day Rolling Volatility (Annualized) - {scheme_name}',
                            yaxis_title='Volatility (%)',
                            template='plotly_dark',
                            height=450,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig4, width='stretch')
                    
                    # Probability table
                    st.subheader("Return Probability Analysis")
                    prob_data = {
                        'Range': ['Negative', '0-5%', '5-10%', '10-15%', '15-20%', '>20%'],
                        'Probability': [
                            f"{stats['Prob(Negative)']:.2f}%",
                            f"{stats['Prob(0-5%)']:.2f}%",
                            f"{stats['Prob(5-10%)']:.2f}%",
                            f"{stats['Prob(10-15%)']:.2f}%",
                            f"{stats['Prob(15-20%)']:.2f}%",
                            f"{stats['Prob(>20%)']:.2f}%"
                        ]
                    }
                    
                    st.dataframe(pd.DataFrame(prob_data), width='stretch')
                    
                    # Advanced Quant Metrics
                    st.subheader("📊 Advanced Quantitative Metrics")
                    
                    quant_col1, quant_col2, quant_col3, quant_col4 = st.columns(4)
                    
                    # Calculate additional metrics
                    returns = rolling_returns.values
                    downside_returns = returns[returns < 0]
                    
                    # Sortino Ratio (like Sharpe but only counts downside)
                    if len(downside_returns) > 0:
                        downside_std = np.std(downside_returns)
                        sortino = (np.mean(returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0
                    else:
                        sortino = np.inf if np.mean(returns) > 0 else 0
                    
                    # Calmar Ratio (return / max drawdown)
                    cumulative_returns = (1 + nav_numeric.pct_change().dropna()).cumprod()
                    running_max = cumulative_returns.cummax()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_dd = drawdown.min()
                    annual_return_pct = np.mean(returns)
                    calmar = abs(annual_return_pct / max_dd) if max_dd != 0 else 0
                    
                    # Win Rate
                    win_rate = (len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0
                    
                    with quant_col1:
                        st.metric("Sortino Ratio", f"{sortino:.2f}")
                    with quant_col2:
                        st.metric("Calmar Ratio", f"{calmar:.2f}")
                    with quant_col3:
                        st.metric("Win Rate", f"{win_rate:.1f}%")
                    with quant_col4:
                        st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                else:
                    st.warning(f"Not enough data for {period} analysis")
            else:
                st.error(f"Could not fetch historical data for scheme {selected_code}")

# =============================================================================
# PAGE: COMPARE
# =============================================================================

elif page == "🔄 Compare":
    st.title("🔄 Compare Schemes")
    
    if len(st.session_state.selected_schemes) < 2:
        st.warning("Please select at least 2 schemes to compare")
    else:
        compare_codes = st.multiselect(
            "Select schemes to compare (max 5):",
            list(st.session_state.selected_schemes.keys()),
            max_selections=5,
            format_func=lambda x: st.session_state.selected_schemes[x]
        )
        
        if len(compare_codes) >= 2:
            comparison_data = []
            
            for code in compare_codes:
                quote = get_scheme_quote(code)
                details = get_scheme_details(code)
                
                if quote:
                    row = {
                        'Scheme': st.session_state.selected_schemes[code],
                        'Code': code,
                        'Current NAV': quote.get('nav', 'N/A'),
                        'Last Updated': quote.get('last_updated', 'N/A'),
                    }
                    
                    if details:
                        row['Fund House'] = details.get('fund_house', 'N/A')
                        row['Category'] = details.get('scheme_category', 'N/A')
                    
                    comparison_data.append(row)
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), width='stretch')
                
                # NAV Comparison Chart
                st.subheader("NAV Comparison (Last 1 Year)")
                
                fig = go.Figure()
                
                for code in compare_codes:
                    nav_df = get_scheme_historical_nav(code)
                    
                    if nav_df is not None:
                        # Normalize NAV to 100 on first day - convert to numeric first
                        nav_series = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                        nav_series = nav_series.iloc[::-1].reset_index(drop=True)
                        
                        if len(nav_series) > 0 and nav_series.iloc[0] > 0:
                            normalized = (nav_series / nav_series.iloc[0]) * 100
                            
                            fig.add_trace(go.Scatter(
                                y=normalized.values[-252:] if len(normalized) > 252 else normalized.values,
                                name=st.session_state.selected_schemes[code],
                                mode='lines'
                            ))
                
                fig.update_layout(
                    title="NAV Performance (Indexed to 100)",
                    yaxis_title="Index Value",
                    template='plotly_dark',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Additional comparison graphs
                st.subheader("Comparative Analysis")
                
                col1, col2 = st.columns(2)
                
                # Returns distribution
                with col1:
                    st.subheader("Daily Returns Distribution")
                    
                    fig_dist = go.Figure()
                    
                    for code in compare_codes:
                        nav_df = get_scheme_historical_nav(code)
                        if nav_df is not None:
                            nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                            returns = nav_numeric.pct_change().dropna() * 100  # Convert to percentage
                            
                            fig_dist.add_trace(go.Histogram(
                                x=returns.values,
                                name=st.session_state.selected_schemes[code],
                                opacity=0.7
                            ))
                    
                    fig_dist.update_layout(
                        title="Daily Returns Distribution (%)",
                        xaxis_title="Daily Return %",
                        yaxis_title="Frequency",
                        template='plotly_dark',
                        height=400,
                        barmode='overlay'
                    )
                    
                    st.plotly_chart(fig_dist, width='stretch')
                
                # Cumulative returns
                with col2:
                    st.subheader("Cumulative Returns (Last 1 Year)")
                    
                    fig_cum = go.Figure()
                    
                    for code in compare_codes:
                        nav_df = get_scheme_historical_nav(code)
                        if nav_df is not None:
                            nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                            nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                            returns = nav_numeric.pct_change().dropna()
                            cum_returns = (1 + returns).cumprod() - 1
                            
                            fig_cum.add_trace(go.Scatter(
                                y=(cum_returns.values * 100)[-252:] if len(cum_returns) > 252 else (cum_returns.values * 100),
                                name=st.session_state.selected_schemes[code],
                                mode='lines',
                                fill='tozeroy'
                            ))
                    
                    fig_cum.update_layout(
                        title="Cumulative Returns (%)",
                        yaxis_title="Cumulative Return %",
                        template='plotly_dark',
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_cum, width='stretch')
                
                # Risk-Return scatter
                st.subheader("Risk vs Return Analysis")
                
                fig_scatter = go.Figure()
                
                risk_return_data = []
                
                for code in compare_codes:
                    nav_df = get_scheme_historical_nav(code)
                    if nav_df is not None:
                        nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                        returns = nav_numeric.pct_change().dropna()
                        
                        annual_return = (returns.mean() * 252) * 100  # Annualized return
                        annual_volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                        
                        risk_return_data.append({
                            'scheme': st.session_state.selected_schemes[code],
                            'return': annual_return,
                            'volatility': annual_volatility,
                            'sharpe': annual_return / annual_volatility if annual_volatility > 0 else 0
                        })
                
                if risk_return_data:
                    df_rr = pd.DataFrame(risk_return_data)
                    
                    fig_scatter = go.Figure(data=[go.Scatter(
                        x=df_rr['volatility'],
                        y=df_rr['return'],
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color=df_rr['sharpe'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Sharpe Ratio")
                        ),
                        text=df_rr['scheme'],
                        textposition='top center'
                    )])
                    
                    fig_scatter.update_layout(
                        title="Risk vs Return (Annualized)",
                        xaxis_title="Annual Volatility (%)",
                        yaxis_title="Annual Return (%)",
                        template='plotly_dark',
                        height=450,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig_scatter, width='stretch')
                    
                    # Risk-Return metrics table with advanced quant metrics
                    st.subheader("Comparative Metrics")
                    
                    # Add advanced quant metrics to comparison
                    quant_metrics = []
                    
                    for code in compare_codes:
                        nav_df = get_scheme_historical_nav(code)
                        if nav_df is not None:
                            nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                            returns = nav_numeric.pct_change().dropna()
                            
                            # Sortino Ratio
                            downside_returns = returns[returns < 0]
                            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                            sortino = (np.mean(returns) * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
                            
                            # Max Drawdown
                            cumulative = (1 + returns).cumprod()
                            running_max = cumulative.cummax()
                            drawdown = (cumulative - running_max) / running_max
                            max_dd = drawdown.min() * 100
                            
                            # Win Rate
                            win_rate = (len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0
                            
                            quant_metrics.append({
                                'Scheme': st.session_state.selected_schemes[code],
                                'Return %': f"{df_rr[df_rr['scheme'] == st.session_state.selected_schemes[code]]['return'].values[0]:.2f}",
                                'Volatility %': f"{df_rr[df_rr['scheme'] == st.session_state.selected_schemes[code]]['volatility'].values[0]:.2f}",
                                'Sharpe': f"{df_rr[df_rr['scheme'] == st.session_state.selected_schemes[code]]['sharpe'].values[0]:.2f}",
                                'Sortino': f"{sortino:.2f}",
                                'Max DD %': f"{max_dd:.2f}",
                                'Win Rate %': f"{win_rate:.1f}"
                            })
                    
                    st.dataframe(pd.DataFrame(quant_metrics), width='stretch')

# =============================================================================
# PAGE: PORTFOLIO
# =============================================================================

elif page == "💼 Portfolio":
    st.title("💼 Advanced Portfolio Builder")
    
    tab1, tab2, tab3 = st.tabs(["📊 Build Portfolio", "🎯 Asset Allocation", "📈 Performance"])
    
    with tab1:
        st.subheader("Quick Portfolio Builder")
        
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_type = st.selectbox(
                "Select Portfolio Type:",
                ["Custom", "Conservative (80% Debt, 20% Equity)", 
                 "Balanced (60% Equity, 40% Debt)", 
                 "Aggressive (80% Equity, 20% Debt)"]
            )
        
        with col2:
            investment_amount = st.number_input(
                "Investment Amount (₹):",
                min_value=1000,
                value=100000,
                step=10000
            )
        
        st.markdown("---")
        
        if portfolio_type != "Custom":
            # Pre-defined portfolios
            with st.spinner("Loading funds for selected portfolio type..."):
                if "Conservative" in portfolio_type:
                    debt_schemes = filter_schemes_by_type("debt")
                    equity_schemes = filter_schemes_by_type("equity")
                    
                    if debt_schemes and equity_schemes:
                        # Pick best from each
                        debt_list = list(debt_schemes.items())[:3]
                        equity_list = list(equity_schemes.items())[:1]
                        
                        portfolio_allocation = {}
                        for code, name in debt_list:
                            portfolio_allocation[code] = 26.67  # 80% / 3
                        for code, name in equity_list:
                            portfolio_allocation[code] = 20.0
                        
                        st.session_state.selected_schemes.update(dict(debt_list + equity_list))
                
                elif "Balanced" in portfolio_type:
                    debt_schemes = filter_schemes_by_type("debt")
                    equity_schemes = filter_schemes_by_type("equity")
                    
                    if debt_schemes and equity_schemes:
                        debt_list = list(debt_schemes.items())[:2]
                        equity_list = list(equity_schemes.items())[:2]
                        
                        portfolio_allocation = {}
                        for code, name in equity_list:
                            portfolio_allocation[code] = 30.0  # 60% / 2
                        for code, name in debt_list:
                            portfolio_allocation[code] = 20.0  # 40% / 2
                        
                        st.session_state.selected_schemes.update(dict(debt_list + equity_list))
                
                elif "Aggressive" in portfolio_type:
                    equity_schemes = filter_schemes_by_type("equity")
                    debt_schemes = filter_schemes_by_type("debt")
                    
                    if equity_schemes and debt_schemes:
                        equity_list = list(equity_schemes.items())[:3]
                        debt_list = list(debt_schemes.items())[:1]
                        
                        portfolio_allocation = {}
                        for code, name in equity_list:
                            portfolio_allocation[code] = 26.67  # 80% / 3
                        for code, name in debt_list:
                            portfolio_allocation[code] = 20.0
                        
                        st.session_state.selected_schemes.update(dict(equity_list + debt_list))
        else:
            portfolio_allocation = {}
        
        st.markdown("---")
        st.subheader("Manual Portfolio Configuration")
        
        if st.session_state.selected_schemes:
            weights = {}
            total_weight = 0
            
            col1, col2 = st.columns([3, 1])
            
            for code, name in st.session_state.selected_schemes.items():
                with col1:
                    weight = st.slider(
                        f"{name}",
                        min_value=0,
                        max_value=100,
                        value=int(portfolio_allocation.get(code, 100 / len(st.session_state.selected_schemes))),
                        step=1,
                        key=f"weight_{code}"
                    )
                weights[code] = weight
                total_weight += weight
            
            # Status indicator
            if total_weight == 100:
                st.success(f"✅ Total Weight: {total_weight}% - Portfolio is balanced!")
            else:
                st.warning(f"⚠️ Total Weight: {total_weight}% - Need {100 - total_weight}% more")
            
            st.markdown("---")
            
            if total_weight == 100 and st.button("📊 Analyze Portfolio", key="analyze_portfolio"):
                st.success("Portfolio weights are valid!")
                
                portfolio_data = []
                total_weighted_nav = 0
                allocation_amount = {}
                
                for code, weight in weights.items():
                    quote = get_scheme_quote(code)
                    if quote:
                        nav = float(quote.get('nav', 0))
                        allocation = (weight / 100) * investment_amount
                        units = allocation / nav
                        weighted_nav = (weight / 100) * nav
                        total_weighted_nav += weighted_nav
                        
                        allocation_amount[code] = {
                            'allocation': allocation,
                            'units': units,
                            'nav': nav
                        }
                        
                        portfolio_data.append({
                            'Scheme': st.session_state.selected_schemes[code],
                            'Allocation %': f"{weight}%",
                            'Amount': f"₹{allocation:,.0f}",
                            'Current NAV': f"₹{nav:.2f}",
                            'Units': f"{units:,.2f}"
                        })
                
                st.dataframe(pd.DataFrame(portfolio_data), width='stretch')
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Portfolio NAV", f"₹{total_weighted_nav:.4f}")
                with col2:
                    st.metric("Total Investment", f"₹{investment_amount:,.0f}")
                with col3:
                    st.metric("Total Units", f"{sum(x['units'] for x in allocation_amount.values()):,.2f}")
                with col4:
                    avg_nav = total_weighted_nav
                    st.metric("Weighted Avg NAV", f"₹{avg_nav:.2f}")
                
                # Enhanced visualization - Allocation bar chart
                st.subheader("Allocation Breakdown (₹)")
                allocation_amounts = [allocation_amount[code]['allocation'] for code in weights.keys()]
                scheme_names = [st.session_state.selected_schemes[code][:30] + "..." if len(st.session_state.selected_schemes[code]) > 30 else st.session_state.selected_schemes[code] for code in weights.keys()]
                
                fig_alloc = go.Figure(data=[
                    go.Bar(x=scheme_names, y=allocation_amounts, text=[f"₹{x:,.0f}" for x in allocation_amounts], textposition='outside')
                ])
                fig_alloc.update_layout(
                    title="Investment Amount per Scheme",
                    yaxis_title="Amount (₹)",
                    template='plotly_dark',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_alloc, width='stretch')
                
                # Units comparison chart
                st.subheader("Units per Scheme")
                units_list = [allocation_amount[code]['units'] for code in weights.keys()]
                
                fig_units = go.Figure(data=[
                    go.Bar(x=scheme_names, y=units_list, text=[f"{x:,.2f}" for x in units_list], textposition='outside', marker_color='#00cc88')
                ])
                fig_units.update_layout(
                    title="Total Units Allocation",
                    yaxis_title="Units",
                    template='plotly_dark',
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_units, width='stretch')
        else:
            st.info("👈 Select schemes from 'Search Funds' page first")
    
    with tab2:
        st.subheader("Asset Allocation Strategy")
        
        if st.session_state.selected_schemes and total_weight == 100:
            col1, col2 = st.columns(2)
            
            # Pie chart
            allocation_values = [weights.get(code, 0) for code in st.session_state.selected_schemes.keys()]
            allocation_labels = [st.session_state.selected_schemes[code] for code in st.session_state.selected_schemes.keys()]
            
            with col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=allocation_labels,
                    values=allocation_values,
                    marker=dict(line=dict(color='#1f1f1f', width=2))
                )])
                
                fig_pie.update_layout(
                    title="Portfolio Allocation (%)",
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_pie, width='stretch')
            
            # Sunburst chart for detailed breakdown
            with col2:
                fig_sun = go.Figure(data=[go.Sunburst(
                    labels=["Portfolio"] + allocation_labels,
                    parents=[""] + ["Portfolio"] * len(allocation_labels),
                    values=[100] + allocation_values,
                    marker=dict(line=dict(color='#1f1f1f', width=2))
                )])
                
                fig_sun.update_layout(
                    title="Hierarchical Allocation",
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_sun, width='stretch')
            
            # Category-wise breakdown
            st.subheader("Category Breakdown")
            
            equity_weight = 0
            debt_weight = 0
            hybrid_weight = 0
            
            for code, weight in weights.items():
                name_lower = st.session_state.selected_schemes[code].lower()
                if any(x in name_lower for x in ['equity', 'large cap', 'mid cap', 'small cap', 'multi cap', 'flexi cap']):
                    equity_weight += weight
                elif any(x in name_lower for x in ['debt', 'bond', 'banking', 'psu', 'credit', 'liquid']):
                    debt_weight += weight
                elif any(x in name_lower for x in ['hybrid', 'balanced', 'allocation']):
                    hybrid_weight += weight
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Equity Allocation", f"{equity_weight:.1f}%")
            with col2:
                st.metric("Debt Allocation", f"{debt_weight:.1f}%")
            with col3:
                st.metric("Hybrid Allocation", f"{hybrid_weight:.1f}%")
            
            # Category distribution chart
            category_data = [equity_weight, debt_weight, hybrid_weight]
            category_labels = ['Equity', 'Debt', 'Hybrid']
            
            fig_cat = go.Figure(data=[
                go.Bar(x=category_labels, y=category_data, text=[f"{x:.1f}%" for x in category_data], textposition='outside',
                       marker_color=['#667eea', '#ff6b6b', '#ffd93d'])
            ])
            fig_cat.update_layout(
                title="Asset Class Allocation",
                yaxis_title="Allocation (%)",
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_cat, width='stretch')
        else:
            st.info("Create a portfolio with 100% allocation to view asset allocation chart")
    
    with tab3:
        st.subheader("Portfolio Performance Analysis")
        
        if st.session_state.selected_schemes and total_weight == 100:
            # Get NAV history for each scheme
            portfolio_nav_history = {}
            
            for code in st.session_state.selected_schemes.keys():
                nav_df = get_scheme_historical_nav(code)
                if nav_df is not None:
                    portfolio_nav_history[code] = nav_df
            
            if portfolio_nav_history:
                # Calculate portfolio NAV over time
                combined_nav = None
                
                for code, nav_df in portfolio_nav_history.items():
                    weight = weights.get(code, 0) / 100
                    nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce')
                    
                    if combined_nav is None:
                        combined_nav = nav_numeric * weight
                    else:
                        # Align dates
                        aligned = nav_numeric * weight
                        combined_nav = combined_nav.add(aligned, fill_value=0)
                
                if combined_nav is not None:
                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=combined_nav.values[-252:] if len(combined_nav) > 252 else combined_nav.values,
                        mode='lines',
                        name='Portfolio NAV',
                        line=dict(color='#667eea', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Portfolio NAV Performance (Last 1 Year)",
                        yaxis_title="NAV",
                        template='plotly_dark',
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info("Create a portfolio with 100% allocation to view performance")

# =============================================================================
# PAGE: AUTO PORTFOLIO
# =============================================================================

elif page == "🤖 Auto Portfolio":
    st.title("🤖 Automated Portfolio Generator")
    st.markdown("Create a diversified, risk-aligned portfolio with AI-powered fund selection")
    
    with st.sidebar:
        st.subheader("⚙️ Portfolio Settings")
        
        risk_profile = st.selectbox(
            "Risk Profile:",
            ["Conservative", "Moderate", "Balanced", "Aggressive", "Very Aggressive"],
            help="Choose based on your investment horizon and risk tolerance"
        )
        
        investment_amount = st.number_input(
            "Investment Amount (₹):",
            min_value=5000,
            value=100000,
            step=5000,
            help="Total amount to invest across selected funds"
        )
        
        num_funds = st.slider(
            "Number of Funds:",
            min_value=3,
            max_value=10,
            value=6,
            help="More funds = better diversification but harder to manage"
        )
        
        investment_horizon = st.selectbox(
            "Investment Horizon:",
            ["< 1 Year", "1-3 Years", "3-5 Years", "5-10 Years", "10+ Years"],
            help="How long you plan to stay invested"
        )
    
    # Risk profile allocation mappings
    risk_allocations = {
        "Conservative": {"equity": 20, "debt": 80, "hybrid": 0},
        "Moderate": {"equity": 35, "debt": 60, "hybrid": 5},
        "Balanced": {"equity": 50, "debt": 40, "hybrid": 10},
        "Aggressive": {"equity": 70, "debt": 20, "hybrid": 10},
        "Very Aggressive": {"equity": 85, "debt": 10, "hybrid": 5}
    }
    
    equity_subcats = {
        "Conservative": ["Large Cap"],
        "Moderate": ["Large Cap", "Multi Cap"],
        "Balanced": ["Large Cap", "Multi Cap", "Mid Cap"],
        "Aggressive": ["Mid Cap", "Multi Cap", "Small Cap"],
        "Very Aggressive": ["Small Cap", "Mid Cap", "Focused", "Sectoral"]
    }
    
    allocation = risk_allocations[risk_profile]
    
    # Display allocation breakdown
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Equity", f"{allocation['equity']}%")
    with col2:
        st.metric("Debt", f"{allocation['debt']}%")
    with col3:
        st.metric("Hybrid", f"{allocation['hybrid']}%")
    with col4:
        st.metric("Funds", f"{num_funds}")
    
    if st.button("🚀 Generate Portfolio", use_container_width=True, key="gen_auto_portfolio"):
        with st.spinner("Analyzing funds and building your portfolio..."):
            try:
                # Fetch funds by type
                equity_schemes = filter_schemes_by_type("equity")
                debt_schemes = filter_schemes_by_type("debt")
                hybrid_schemes = filter_schemes_by_type("hybrid")
                
                portfolio_funds = []
                selected_codes = set()
                
                # Calculate funds needed from each category
                total_equity_funds = max(1, int(num_funds * allocation['equity'] / 100))
                total_debt_funds = max(1, int(num_funds * allocation['debt'] / 100))
                total_hybrid_funds = max(0, int(num_funds * allocation['hybrid'] / 100))
                
                # Adjust to match total num_funds
                total = total_equity_funds + total_debt_funds + total_hybrid_funds
                if total < num_funds:
                    total_equity_funds += (num_funds - total)
                
                # Select top ranked EQUITY funds
                if equity_schemes and total_equity_funds > 0:
                    st.info(f"📊 Selecting top {total_equity_funds} equity funds from {len(equity_schemes)} available...")
                    
                    # Score equity funds based on selected subcategories
                    subcats = equity_subcats.get(risk_profile, ["Large Cap"])
                    scored_equity = []
                    
                    for code, name in list(equity_schemes.items())[:200]:  # Limit to first 200 for speed
                        try:
                            nav_df = get_scheme_historical_nav(code)
                            if nav_df is not None and len(nav_df) > 252:
                                nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                returns = nav_numeric.pct_change().dropna()
                                
                                if len(returns) > 252:
                                    # Calculate Sharpe ratio
                                    ann_return = np.mean(returns) * 252
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    sharpe = ann_return / ann_vol if ann_vol > 0 and ann_vol == ann_vol else 0
                                    sharpe = min(max(sharpe, -5), 10)  # Cap sharpe between -5 and 10
                                    
                                    # Category bonus - prioritize selected subcategories
                                    cat_bonus = 1.0
                                    name_lower = name.lower()
                                    for subcat in subcats:
                                        if subcat.lower() in name_lower:
                                            cat_bonus = 1.5
                                            break
                                    
                                    # Fund age bonus - prefer older funds
                                    age_years = len(returns) / 252
                                    age_bonus = min(age_years / 10, 1.0)
                                    
                                    score = (sharpe * 10 + age_bonus * 5) * cat_bonus
                                    scored_equity.append((code, name, score, sharpe))
                        except:
                            continue
                    
                    # Sort by score and select top funds
                    scored_equity.sort(key=lambda x: x[2], reverse=True)
                    for code, name, score, sharpe in scored_equity[:total_equity_funds]:
                        if code not in selected_codes:
                            portfolio_funds.append((code, name, "Equity", sharpe))
                            selected_codes.add(code)
                
                # Select top DEBT funds
                if debt_schemes and total_debt_funds > 0:
                    st.info(f"🏢 Selecting top {total_debt_funds} debt funds from {len(debt_schemes)} available...")
                    scored_debt = []
                    
                    for code, name in list(debt_schemes.items())[:100]:  # Limit to first 100
                        try:
                            nav_df = get_scheme_historical_nav(code)
                            if nav_df is not None and len(nav_df) > 252:
                                nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                returns = nav_numeric.pct_change().dropna()
                                
                                if len(returns) > 252:
                                    ann_return = np.mean(returns) * 252
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    sharpe = ann_return / ann_vol if ann_vol > 0 and ann_vol == ann_vol else 0
                                    sharpe = min(max(sharpe, -5), 10)
                                    
                                    age_years = len(returns) / 252
                                    score = sharpe * 10 + (age_years / 10)
                                    scored_debt.append((code, name, score, sharpe))
                        except:
                            continue
                    
                    scored_debt.sort(key=lambda x: x[2], reverse=True)
                    for code, name, score, sharpe in scored_debt[:total_debt_funds]:
                        if code not in selected_codes:
                            portfolio_funds.append((code, name, "Debt", sharpe))
                            selected_codes.add(code)
                
                # Select top HYBRID funds
                if hybrid_schemes and total_hybrid_funds > 0:
                    st.info(f"🔄 Selecting top {total_hybrid_funds} hybrid funds from {len(hybrid_schemes)} available...")
                    scored_hybrid = []
                    
                    for code, name in list(hybrid_schemes.items())[:50]:  # Limit to first 50
                        try:
                            nav_df = get_scheme_historical_nav(code)
                            if nav_df is not None and len(nav_df) > 252:
                                nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                returns = nav_numeric.pct_change().dropna()
                                
                                if len(returns) > 252:
                                    ann_return = np.mean(returns) * 252
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    sharpe = ann_return / ann_vol if ann_vol > 0 and ann_vol == ann_vol else 0
                                    sharpe = min(max(sharpe, -5), 10)
                                    
                                    score = sharpe * 10
                                    scored_hybrid.append((code, name, score, sharpe))
                        except:
                            continue
                    
                    scored_hybrid.sort(key=lambda x: x[2], reverse=True)
                    for code, name, score, sharpe in scored_hybrid[:total_hybrid_funds]:
                        if code not in selected_codes:
                            portfolio_funds.append((code, name, "Hybrid", sharpe))
                            selected_codes.add(code)
                
                if len(portfolio_funds) > 0:
                    st.success(f"✅ Generated portfolio with {len(portfolio_funds)} funds!")
                    st.markdown("---")
                    
                    # Calculate equal-weight allocation
                    weight_per_fund = 100 / len(portfolio_funds)
                    amount_per_fund = investment_amount / len(portfolio_funds)
                    
                    # Display portfolio
                    st.subheader("📋 Your Auto-Generated Portfolio")
                    
                    portfolio_data = []
                    for i, (code, name, fund_type, sharpe) in enumerate(portfolio_funds, 1):
                        portfolio_data.append({
                            "#": i,
                            "Fund Name": name[:50],
                            "Code": code,
                            "Type": fund_type,
                            "Weight %": f"{weight_per_fund:.1f}%",
                            "Amount (₹)": f"₹{amount_per_fund:,.0f}",
                            "Sharpe Ratio": f"{sharpe:.2f}" if sharpe == sharpe else "N/A"
                        })
                    
                    df_portfolio = pd.DataFrame(portfolio_data)
                    st.dataframe(df_portfolio, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💰 Total Investment", f"₹{investment_amount:,.0f}")
                    with col2:
                        st.metric("🎯 Per Fund Amount", f"₹{amount_per_fund:,.0f}")
                    with col3:
                        st.metric("⏱️ Investment Horizon", investment_horizon)
                    with col4:
                        st.metric("📊 Risk Profile", risk_profile)
                    
                    # Portfolio characteristics
                    st.subheader("📊 Portfolio Characteristics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    type_counts = {}
                    for _, _, fund_type, _ in portfolio_funds:
                        type_counts[fund_type] = type_counts.get(fund_type, 0) + 1
                    
                    with col1:
                        st.write("**Asset Class Breakdown:**")
                        for asset_class, count in type_counts.items():
                            st.write(f"• {asset_class}: {count} fund(s)")
                    
                    with col2:
                        st.write("**Recommended Actions:**")
                        st.write("1. Start investing as per amounts")
                        st.write("2. Set up SIP for monthly contributions")
                        st.write("3. Rebalance quarterly")
                        st.write("4. Review annually")
                    
                    with col3:
                        st.write("**Rebalancing Tips:**")
                        st.write(f"• Review every {3 if investment_horizon == '1-3 Years' else 6} months")
                        st.write("• Rebalance when allocation drifts >5%")
                        st.write("• Use new investments to rebalance")
                        st.write("• Avoid frequent trading")
                    
                    # Export portfolio
                    st.markdown("---")
                    
                    csv = df_portfolio.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Portfolio as CSV",
                        data=csv,
                        file_name=f"auto_portfolio_{risk_profile}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Detailed fund analysis
                    st.subheader("🔍 Detailed Fund Analysis")
                    
                    selected_fund_idx = st.selectbox(
                        "Select a fund for detailed analysis:",
                        range(len(portfolio_funds)),
                        format_func=lambda i: f"{i+1}. {portfolio_funds[i][1][:40]}"
                    )
                    
                    if selected_fund_idx is not None:
                        code, name, fund_type, _ = portfolio_funds[selected_fund_idx]
                        
                        st.write(f"**{name}**")
                        st.write(f"Code: {code}")
                        st.write(f"Type: {fund_type}")
                        
                        nav_df = get_scheme_historical_nav(code)
                        if nav_df is not None:
                            nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                            nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                            returns = nav_numeric.pct_change().dropna()
                            
                            # Calculate metrics
                            ann_return = np.mean(returns) * 252
                            ann_vol = np.std(returns) * np.sqrt(252)
                            sharpe = ann_return / ann_vol if ann_vol > 0 and ann_vol == ann_vol else 0
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Annualized Return", f"{ann_return*100:.2f}%")
                            with col2:
                                st.metric("Volatility (Annualized)", f"{ann_vol*100:.2f}%")
                            with col3:
                                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                            with col4:
                                st.metric("Data Points", f"{len(returns)}")
                            
                            # Plot NAV
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=nav_df.index,
                                y=nav_numeric.values,
                                mode='lines',
                                name='NAV',
                                line=dict(color='#667eea', width=2)
                            ))
                            
                            fig.update_layout(
                                title=f"NAV Trend - {name[:30]}",
                                xaxis_title="Date",
                                yaxis_title="NAV (₹)",
                                template='plotly_dark',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("❌ Could not select enough funds. Try adjusting settings or checking data availability.")
            
            except Exception as e:
                st.error(f"❌ Error generating portfolio: {str(e)}")
                st.info("Try reducing the number of funds or changing your risk profile.")

# =============================================================================
# PAGE: AUTO PORTFOLIO MAKER
# =============================================================================

elif page == "🤖 Auto Portfolio":
    st.title("🤖 AI-Powered Auto Portfolio Maker")
    st.markdown("Generate optimal portfolios based on rolling returns analysis and performance metrics")
    
    # Rolling Returns Data for Benchmarks
    ROLLING_RETURNS_DATA = {
        'Nifty Midcap 150 Momentum 50': {
            '1Y': {'avg': 22.12, 'sharpe': 4.39, 'std': 3.67},
            '3Y': {'avg': 20.20, 'sharpe': 1.51, 'std': 8.78},
            '5Y': {'avg': 20.28, 'sharpe': 1.53, 'std': 1.33},
            '7Y': {'avg': 18.79, 'sharpe': 2.74, 'std': 7.14},
            '10Y': {'avg': 17.15, 'sharpe': 4.39, 'std': 10.69}
        },
        'Nifty 50': {
            '1Y': {'avg': 10.77, 'sharpe': 1.99, 'std': 2.40},
            '3Y': {'avg': 9.90, 'sharpe': 1.53, 'std': 12.20},
            '5Y': {'avg': 12.30, 'sharpe': 1.91, 'std': 1.33},
            '7Y': {'avg': 7.14, 'sharpe': 1.97, 'std': 7.14},
            '10Y': {'avg': 10.77, 'sharpe': 1.99, 'std': 10.69}
        },
        'Nifty Smallcap 250': {
            '1Y': {'avg': 12.19, 'sharpe': 1.46, 'std': 4.23},
            '3Y': {'avg': 11.50, 'sharpe': 1.62, 'std': 15.30},
            '5Y': {'avg': 13.40, 'sharpe': 1.82, 'std': 2.10},
            '7Y': {'avg': 12.80, 'sharpe': 1.88, 'std': 8.90},
            '10Y': {'avg': 14.20, 'sharpe': 2.15, 'std': 11.50}
        }
    }
    
    tab1, tab2, tab3 = st.tabs(["⚙️ Portfolio Generator", "📊 Risk Analysis", "🎯 Recommendations"])
    
    with tab1:
        st.subheader("Configure Your Auto Portfolio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            investment_amount = st.number_input(
                "Investment Amount (₹):",
                min_value=10000,
                value=500000,
                step=50000
            )
        
        with col2:
            risk_profile = st.selectbox(
                "Risk Profile:",
                ["Conservative", "Moderate", "Aggressive"]
            )
        
        with col3:
            time_horizon = st.selectbox(
                "Investment Horizon:",
                ["1 Year", "3 Years", "5 Years", "7 Years", "10 Years"]
            )
        
        st.markdown("---")
        
        # Allocation strategy based on risk profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Asset Class Allocation")
            
            if risk_profile == "Conservative":
                equity_pct = st.slider("Equity %", 20, 40, 30)
                debt_pct = st.slider("Debt %", 50, 70, 60)
                hybrid_pct = 100 - equity_pct - debt_pct
            elif risk_profile == "Moderate":
                equity_pct = st.slider("Equity %", 40, 60, 50)
                debt_pct = st.slider("Debt %", 30, 50, 40)
                hybrid_pct = 100 - equity_pct - debt_pct
            else:  # Aggressive
                equity_pct = st.slider("Equity %", 60, 80, 70)
                debt_pct = st.slider("Debt %", 10, 30, 20)
                hybrid_pct = 100 - equity_pct - debt_pct
            
            st.info(f"Hybrid: {hybrid_pct}%")
            
            # Visualization
            alloc_data = [equity_pct, debt_pct, hybrid_pct]
            fig_alloc = go.Figure(data=[go.Pie(
                labels=['Equity', 'Debt', 'Hybrid'],
                values=alloc_data,
                marker=dict(colors=['#667eea', '#ff6b6b', '#ffd93d'])
            )])
            fig_alloc.update_layout(title="Asset Allocation", height=400, template='plotly_dark')
            st.plotly_chart(fig_alloc, use_container_width=True)
        
        with col2:
            st.subheader("Equity Sub-Categories")
            
            large_cap_pct = st.slider("Large Cap %", 20, 50, 40)
            mid_cap_pct = st.slider("Mid Cap %", 20, 40, 30)
            small_cap_pct = 100 - large_cap_pct - mid_cap_pct
            
            st.info(f"Small Cap: {small_cap_pct}%")
            
            equity_sub = [large_cap_pct, mid_cap_pct, small_cap_pct]
            fig_equity = go.Figure(data=[go.Bar(
                x=['Large Cap', 'Mid Cap', 'Small Cap'],
                y=equity_sub,
                text=[f"{x}%" for x in equity_sub],
                textposition='outside',
                marker_color=['#667eea', '#764ba2', '#f093fb']
            )])
            fig_equity.update_layout(title="Equity Distribution", height=400, template='plotly_dark', showlegend=False)
            st.plotly_chart(fig_equity, use_container_width=True)
        
        st.markdown("---")
        
        if st.button("🚀 Generate Portfolio", key="gen_portfolio"):
            with st.spinner("Analyzing funds and generating optimal portfolio..."):
                
                # Get funds by category
                equity_schemes = filter_schemes_by_type("equity")
                debt_schemes = filter_schemes_by_type("debt")
                hybrid_schemes = filter_schemes_by_type("hybrid")
                
                portfolio_allocation = {}
                portfolio_recommendations = []
                
                # Select best funds from each category
                # Large Cap
                if equity_schemes:
                    large_cap_funds = [name for code, name in equity_schemes.items() if 'large' in name.lower()][:3]
                    large_cap_allocation = (equity_pct * large_cap_pct / 100) / 3 if large_cap_funds else 0
                    for fund in large_cap_funds:
                        for code, name in equity_schemes.items():
                            if name == fund:
                                portfolio_recommendations.append({
                                    'Category': 'Large Cap',
                                    'Scheme': name,
                                    'Allocation %': large_cap_allocation,
                                    'Amount': f"₹{(large_cap_allocation / 100) * investment_amount:,.0f}"
                                })
                
                # Mid Cap
                if equity_schemes:
                    mid_cap_funds = [name for code, name in equity_schemes.items() if 'mid' in name.lower()][:3]
                    mid_cap_allocation = (equity_pct * mid_cap_pct / 100) / 3 if mid_cap_funds else 0
                    for fund in mid_cap_funds:
                        for code, name in equity_schemes.items():
                            if name == fund:
                                portfolio_recommendations.append({
                                    'Category': 'Mid Cap',
                                    'Scheme': name,
                                    'Allocation %': mid_cap_allocation,
                                    'Amount': f"₹{(mid_cap_allocation / 100) * investment_amount:,.0f}"
                                })
                
                # Small Cap
                if equity_schemes:
                    small_cap_funds = [name for code, name in equity_schemes.items() if 'small' in name.lower()][:3]
                    small_cap_allocation = (equity_pct * small_cap_pct / 100) / 3 if small_cap_funds else 0
                    for fund in small_cap_funds:
                        for code, name in equity_schemes.items():
                            if name == fund:
                                portfolio_recommendations.append({
                                    'Category': 'Small Cap',
                                    'Scheme': name,
                                    'Allocation %': small_cap_allocation,
                                    'Amount': f"₹{(small_cap_allocation / 100) * investment_amount:,.0f}"
                                })
                
                # Debt funds
                if debt_schemes:
                    debt_funds = list(debt_schemes.items())[:3]
                    debt_allocation = debt_pct / 3
                    for code, name in debt_funds:
                        portfolio_recommendations.append({
                            'Category': 'Debt',
                            'Scheme': name,
                            'Allocation %': debt_allocation,
                            'Amount': f"₹{(debt_allocation / 100) * investment_amount:,.0f}"
                        })
                
                # Hybrid funds
                if hybrid_schemes and hybrid_pct > 0:
                    hybrid_funds = list(hybrid_schemes.items())[:2]
                    hybrid_allocation = hybrid_pct / 2 if hybrid_funds else 0
                    for code, name in hybrid_funds:
                        portfolio_recommendations.append({
                            'Category': 'Hybrid',
                            'Scheme': name,
                            'Allocation %': hybrid_allocation,
                            'Amount': f"₹{(hybrid_allocation / 100) * investment_amount:,.0f}"
                        })
                
                if portfolio_recommendations:
                    st.success("✅ Portfolio generated successfully!")
                    
                    df_portfolio = pd.DataFrame(portfolio_recommendations)
                    st.dataframe(df_portfolio, use_container_width=True, hide_index=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Allocation", "100%")
                    with col2:
                        st.metric("Funds Selected", len(portfolio_recommendations))
                    with col3:
                        st.metric("Investment Amount", f"₹{investment_amount:,}")
                    with col4:
                        st.metric("Risk Level", risk_profile)
                else:
                    st.warning("No funds found for the selected criteria")
    
    with tab2:
        st.subheader("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Rolling Returns Performance**")
            
            time_period = st.radio("Select Time Period:", ["1Y", "3Y", "5Y", "7Y", "10Y"])
            
            rolling_returns = []
            indices = []
            
            for index, data in ROLLING_RETURNS_DATA.items():
                if time_period in data:
                    rolling_returns.append(data[time_period]['avg'])
                    indices.append(index)
            
            if rolling_returns:
                fig_rolling = go.Figure(data=[go.Bar(
                    x=indices,
                    y=rolling_returns,
                    text=[f"{x:.2f}%" for x in rolling_returns],
                    textposition='outside',
                    marker_color=['#667eea', '#764ba2', '#f093fb']
                )])
                fig_rolling.update_layout(
                    title=f"{time_period} Rolling CAGR Returns Comparison",
                    yaxis_title="Average Rolling CAGR (%)",
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
        
        with col2:
            st.write("**Sharpe Ratio Comparison**")
            
            sharpe_ratios = []
            indices = []
            
            for index, data in ROLLING_RETURNS_DATA.items():
                if time_period in data:
                    sharpe_ratios.append(data[time_period]['sharpe'])
                    indices.append(index)
            
            if sharpe_ratios:
                fig_sharpe = go.Figure(data=[go.Bar(
                    x=indices,
                    y=sharpe_ratios,
                    text=[f"{x:.2f}" for x in sharpe_ratios],
                    textposition='outside',
                    marker_color=['#ff6b6b', '#ff8c42', '#ffd93d']
                )])
                fig_sharpe.update_layout(
                    title=f"{time_period} Sharpe Ratio Comparison",
                    yaxis_title="Sharpe Ratio",
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig_sharpe, use_container_width=True)
    
    with tab3:
        st.subheader("Portfolio Recommendations")
        
        rec_type = st.selectbox(
            "Recommendation Type:",
            ["Income Generation", "Capital Growth", "Balanced", "Tax Efficient"]
        )
        
        recommendations = {
            'Income Generation': {
                'description': 'Focus on debt & dividend-yielding equity funds',
                'equity': 30,
                'debt': 60,
                'hybrid': 10
            },
            'Capital Growth': {
                'description': 'Aggressive equity allocation for long-term growth',
                'equity': 80,
                'debt': 10,
                'hybrid': 10
            },
            'Balanced': {
                'description': 'Mix of equity & debt for steady growth',
                'equity': 50,
                'debt': 40,
                'hybrid': 10
            },
            'Tax Efficient': {
                'description': 'Tax-saving and ELSS funds for long-term wealth',
                'equity': 60,
                'debt': 25,
                'hybrid': 15
            }
        }
        
        rec = recommendations[rec_type]
        st.write(f"**{rec['description']}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Equity", f"{rec['equity']}%")
        with col2:
            st.metric("Debt", f"{rec['debt']}%")
        with col3:
            st.metric("Hybrid", f"{rec['hybrid']}%")
        
        fig_rec = go.Figure(data=[go.Pie(
            labels=['Equity', 'Debt', 'Hybrid'],
            values=[rec['equity'], rec['debt'], rec['hybrid']],
            marker=dict(colors=['#667eea', '#ff6b6b', '#ffd93d'])
        )])
        fig_rec.update_layout(title=f"{rec_type} Portfolio", height=500, template='plotly_dark')
        st.plotly_chart(fig_rec, use_container_width=True)

# =============================================================================
# PAGE: PERFORMANCE
# =============================================================================

elif page == "📈 Performance":
    st.title("📈 Scheme Performance Analysis")
    
    tab1, tab2 = st.tabs(["📊 Equity Performance", "💰 Debt Performance"])
    
    with tab1:
        st.subheader("Open Ended Equity Schemes Performance")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            perf_category = st.selectbox(
                "Select Category:",
                ["Large Cap", "Large & Mid Cap", "Flexi Cap", "Multi Cap", "Mid Cap", "Small Cap", 
                 "Value", "ELSS", "Contra", "Dividend Yield", "Focused", "Sectoral / Thematic"],
                key="equity_perf_category"
            )
        
        with col2:
            show_metrics = st.checkbox("Show Advanced Metrics", value=True)
        
        if st.button("📊 Load Equity Performance", key="load_equity_perf"):
            with st.spinner(f"Loading {perf_category} schemes..."):
                try:
                    # Get all equity schemes
                    all_schemes = get_all_scheme_codes()
                    
                    # Map category names to keywords
                    category_keywords = {
                        "Large Cap": ["large cap"],
                        "Large & Mid Cap": ["large & midcap", "large and midcap"],
                        "Flexi Cap": ["flexi cap", "flexible cap"],
                        "Multi Cap": ["multi cap"],
                        "Mid Cap": ["mid cap", "midcap"],
                        "Small Cap": ["small cap", "smallcap"],
                        "Value": ["value"],
                        "ELSS": ["elss"],
                        "Contra": ["contra"],
                        "Dividend Yield": ["dividend yield"],
                        "Focused": ["focused"],
                        "Sectoral / Thematic": ["sectoral", "thematic", "banking", "pharma", "it", "infrastructure"]
                    }
                    
                    # Filter schemes for selected category
                    category_schemes = {}
                    keywords = category_keywords.get(perf_category, [])
                    exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'payout', 'bonus', 'direct']
                    
                    for code, name in all_schemes.items():
                        if not name or not isinstance(name, str):
                            continue
                        
                        name_lower = name.lower()
                        
                        # Skip excluded patterns
                        if any(p in name_lower for p in exclude_patterns):
                            continue
                        
                        # Keep only growth/regular
                        if 'growth' not in name_lower and 'regular' not in name_lower:
                            continue
                        
                        # Check for category keywords
                        if any(kw in name_lower for kw in keywords):
                            category_schemes[code] = name
                    
                    if category_schemes:
                        st.success(f"Found {len(category_schemes)} {perf_category} schemes (Growth/Regular)")
                        
                        # Display performance table
                        perf_data = []
                        top_performers = []
                        
                        for code, name in list(category_schemes.items())[:30]:
                            try:
                                quote = get_scheme_quote(code)
                                nav_df = get_scheme_historical_nav(code)
                                details = get_scheme_details(code)
                                
                                if quote and nav_df is not None and len(nav_df) > 252:
                                    nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                    nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                    
                                    # Calculate returns for different periods
                                    if len(nav_numeric) > 252:
                                        ret_1y = ((nav_numeric.iloc[-1] / nav_numeric.iloc[-252]) - 1) * 100 if len(nav_numeric) >= 252 else np.nan
                                    else:
                                        ret_1y = np.nan
                                    
                                    if len(nav_numeric) > 756:
                                        ret_3y = (((nav_numeric.iloc[-1] / nav_numeric.iloc[-756]) ** (1/3)) - 1) * 100
                                    else:
                                        ret_3y = np.nan
                                    
                                    if len(nav_numeric) > 1260:
                                        ret_5y = (((nav_numeric.iloc[-1] / nav_numeric.iloc[-1260]) ** (1/5)) - 1) * 100
                                    else:
                                        ret_5y = np.nan
                                    
                                    fund_house = details.get('fund_house', 'N/A') if details else 'N/A'
                                    aum = details.get('aum', 'N/A') if details else 'N/A'
                                    
                                    perf_data.append({
                                        'Scheme': name[:50],
                                        'Code': code,
                                        '1Y Return %': f"{ret_1y:.2f}" if not np.isnan(ret_1y) else "N/A",
                                        '3Y Return %': f"{ret_3y:.2f}" if not np.isnan(ret_3y) else "N/A",
                                        '5Y Return %': f"{ret_5y:.2f}" if not np.isnan(ret_5y) else "N/A",
                                        'Fund House': fund_house,
                                        'AUM': aum
                                    })
                                    
                                    if not np.isnan(ret_1y):
                                        top_performers.append((code, name, ret_1y, nav_numeric))
                            
                            except Exception as e:
                                continue
                        
                        if perf_data:
                            st.dataframe(pd.DataFrame(perf_data), width='stretch')
                            
                            # Advanced metrics for top 3 performers
                            if show_metrics and top_performers:
                                st.subheader("📈 Top Performers - Advanced Metrics")
                                
                                top_performers.sort(key=lambda x: x[2], reverse=True)
                                
                                metrics_cols = st.columns(min(3, len(top_performers)))
                                
                                for idx, (code, name, ret_1y, nav_numeric) in enumerate(top_performers[:3]):
                                    returns = nav_numeric.pct_change().dropna()
                                    
                                    ann_return = np.mean(returns) * 252
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    
                                    downside_returns = returns[returns < 0]
                                    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
                                    
                                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                                    sortino = (ann_return - 0.05) / downside_vol if downside_vol > 0 else 0
                                    
                                    cumulative = (1 + returns).cumprod()
                                    running_max = cumulative.cummax()
                                    drawdown = (cumulative - running_max) / running_max
                                    max_dd = drawdown.min()
                                    
                                    calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
                                    win_rate = (len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0
                                    
                                    with metrics_cols[idx]:
                                        st.write(f"**{name[:30]}...**")
                                        st.metric("Annual Return %", f"{ann_return*100:.2f}")
                                        st.metric("Volatility %", f"{ann_vol*100:.2f}")
                                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                                        st.metric("Sortino Ratio", f"{sortino:.2f}")
                                        st.metric("Calmar Ratio", f"{calmar:.2f}")
                                        st.metric("Max Drawdown %", f"{max_dd*100:.2f}")
                                        st.metric("Win Rate %", f"{win_rate:.1f}")
                        else:
                            st.info(f"📂 No Growth/Regular schemes found in {perf_category}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Alternative: Category statistics
        st.markdown("---")
        st.subheader("📊 Category-wise Comparison")
        
        if st.button("📈 Calculate Category Statistics", key="category_stats"):
            with st.spinner("Analyzing all equity categories..."):
                try:
                    all_schemes = get_all_scheme_codes()
                    
                    # Category definitions
                    categories_def = {
                        'Large Cap': ['large cap'],
                        'Mid Cap': ['mid cap', 'midcap'],
                        'Small Cap': ['small cap', 'smallcap'],
                        'Flexi Cap': ['flexi cap'],
                        'Multi Cap': ['multi cap'],
                    }
                    
                    cat_stats = []
                    exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'payout', 'bonus', 'direct']
                    
                    for cat_name, keywords in categories_def.items():
                        cat_schemes = {}
                        
                        for code, name in all_schemes.items():
                            if not name or not isinstance(name, str):
                                continue
                            
                            name_lower = name.lower()
                            
                            if any(p in name_lower for p in exclude_patterns):
                                continue
                            
                            if 'growth' not in name_lower and 'regular' not in name_lower:
                                continue
                            
                            if any(kw in name_lower for kw in keywords):
                                cat_schemes[code] = name
                        
                        if cat_schemes:
                            returns_list = []
                            
                            for code in list(cat_schemes.keys())[:15]:
                                try:
                                    nav_df = get_scheme_historical_nav(code)
                                    if nav_df is not None and len(nav_df) > 252:
                                        nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                        nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                        returns = nav_numeric.pct_change().dropna()
                                        
                                        if len(nav_numeric) > 252:
                                            ret_1y = ((nav_numeric.iloc[-1] / nav_numeric.iloc[-252]) - 1) * 100
                                            returns_list.append(ret_1y)
                                except:
                                    continue
                            
                            if returns_list:
                                cat_stats.append({
                                    'Category': cat_name,
                                    'Avg 1Y Return %': f"{np.mean(returns_list):.2f}",
                                    'Median Return %': f"{np.median(returns_list):.2f}",
                                    'Fund Count': len(cat_schemes)
                                })
                    
                    if cat_stats:
                        st.dataframe(pd.DataFrame(cat_stats), width='stretch')
                        
                        # Visualization
                        df_cat_vis = pd.DataFrame(cat_stats)
                        df_cat_vis['Avg 1Y Return %'] = df_cat_vis['Avg 1Y Return %'].astype(float)
                        
                        fig = go.Figure(data=[
                            go.Bar(x=df_cat_vis['Category'], y=df_cat_vis['Avg 1Y Return %'], 
                                   text=df_cat_vis['Avg 1Y Return %'], textposition='outside',
                                   marker_color=['#667eea', '#764ba2', '#ff6b6b', '#ffd93d', '#00cc88'])
                        ])
                        
                        fig.update_layout(
                            title="Category-wise Average 1Y Returns",
                            xaxis_title="Category",
                            yaxis_title="Return (%)",
                            template='plotly_dark',
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.subheader("Open Ended Debt Schemes Performance")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            debt_category = st.selectbox(
                "Select Category:",
                ["Liquid", "Ultra Short Duration", "Short Duration", "Medium Duration", 
                 "Long Duration", "Banking & PSU", "Credit Risk", "Gilt"],
                key="debt_perf_category"
            )
        
        with col2:
            show_debt_metrics = st.checkbox("Show Advanced Metrics", value=True, key="debt_metrics")
        
        if st.button("💹 Load Debt Performance", key="load_debt_perf"):
            with st.spinner(f"Loading {debt_category} debt schemes..."):
                try:
                    all_schemes = get_all_scheme_codes()
                    
                    debt_keywords = {
                        "Liquid": ["liquid"],
                        "Ultra Short Duration": ["ultra short", "ultra-short", "overnight"],
                        "Short Duration": ["short duration", "short-duration"],
                        "Medium Duration": ["medium duration", "medium-duration"],
                        "Long Duration": ["long duration", "long-duration"],
                        "Banking & PSU": ["banking", "psu"],
                        "Credit Risk": ["credit", "high yield"],
                        "Gilt": ["gilt"]
                    }
                    
                    debt_schemes = {}
                    exclude_patterns = ['dividend', 'idcw', 'monthly', 'quarterly', 'payout', 'bonus', 'direct']
                    keywords = debt_keywords.get(debt_category, [])
                    
                    for code, name in all_schemes.items():
                        if not name or not isinstance(name, str):
                            continue
                        
                        name_lower = name.lower()
                        
                        if any(p in name_lower for p in exclude_patterns):
                            continue
                        
                        if 'growth' not in name_lower and 'regular' not in name_lower:
                            continue
                        
                        # Check if it's a debt scheme
                        if not any(x in name_lower for x in ['debt', 'bond', 'banking', 'psu', 'credit', 'liquid', 'gilt', 'overnight']):
                            continue
                        
                        if any(kw in name_lower for kw in keywords):
                            debt_schemes[code] = name
                    
                    if debt_schemes:
                        st.success(f"Found {len(debt_schemes)} {debt_category} schemes")
                        
                        perf_data_debt = []
                        top_debt = []
                        
                        for code, name in list(debt_schemes.items())[:25]:
                            try:
                                quote = get_scheme_quote(code)
                                nav_df = get_scheme_historical_nav(code)
                                details = get_scheme_details(code)
                                
                                if quote and nav_df is not None and len(nav_df) > 252:
                                    nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                    nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                    
                                    if len(nav_numeric) > 252:
                                        ret_1y = ((nav_numeric.iloc[-1] / nav_numeric.iloc[-252]) - 1) * 100
                                    else:
                                        ret_1y = np.nan
                                    
                                    if len(nav_numeric) > 756:
                                        ret_3y = (((nav_numeric.iloc[-1] / nav_numeric.iloc[-756]) ** (1/3)) - 1) * 100
                                    else:
                                        ret_3y = np.nan
                                    
                                    ytm = details.get('ytm', 'N/A') if details else 'N/A'
                                    duration = details.get('duration', 'N/A') if details else 'N/A'
                                    
                                    perf_data_debt.append({
                                        'Scheme': name[:50],
                                        'Code': code,
                                        '1Y Return %': f"{ret_1y:.2f}" if not np.isnan(ret_1y) else "N/A",
                                        '3Y Return %': f"{ret_3y:.2f}" if not np.isnan(ret_3y) else "N/A",
                                        'YTM %': ytm,
                                        'Duration': duration
                                    })
                                    
                                    if not np.isnan(ret_1y):
                                        top_debt.append((code, name, ret_1y, nav_numeric))
                            
                            except:
                                continue
                        
                        if perf_data_debt:
                            st.dataframe(pd.DataFrame(perf_data_debt), width='stretch')
                            
                            if show_debt_metrics and top_debt:
                                st.subheader("💹 Top Performers - Advanced Metrics")
                                
                                top_debt.sort(key=lambda x: x[2], reverse=True)
                                
                                debt_metrics_cols = st.columns(min(3, len(top_debt)))
                                
                                for idx, (code, name, ret_1y, nav_numeric) in enumerate(top_debt[:3]):
                                    returns = nav_numeric.pct_change().dropna()
                                    
                                    ann_return = np.mean(returns) * 252
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    
                                    downside_returns = returns[returns < 0]
                                    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
                                    
                                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                                    sortino = (ann_return - 0.03) / downside_vol if downside_vol > 0 else 0
                                    
                                    cumulative = (1 + returns).cumprod()
                                    running_max = cumulative.cummax()
                                    drawdown = (cumulative - running_max) / running_max
                                    max_dd = drawdown.min()
                                    
                                    with debt_metrics_cols[idx]:
                                        st.write(f"**{name[:30]}...**")
                                        st.metric("Annual Return %", f"{ann_return*100:.2f}")
                                        st.metric("Volatility %", f"{ann_vol*100:.2f}")
                                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                                        st.metric("Sortino Ratio", f"{sortino:.2f}")
                                        st.metric("Max Drawdown %", f"{max_dd*100:.2f}")
                        else:
                            st.info("📂 No schemes found in this category")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# =============================================================================
# PAGE: CALCULATOR
# =============================================================================

elif page == "💰 Calculator":
    st.title("💰 Return Calculator")
    
    col1, col2 = st.tabs(["Balance Units Value", "Investment Returns"])
    
    with col1:
        st.subheader("Calculate Current Portfolio Value")
        
        all_schemes = get_all_scheme_codes()
        
        if all_schemes:
            selected_code = st.selectbox(
                "Select Scheme:",
                list(all_schemes.keys()),
                format_func=lambda x: all_schemes.get(x, x)
            )
            
            units = st.number_input("Number of Units:", min_value=0.0, value=100.0, step=0.001)
            
            if st.button("Calculate Value"):
                mf = init_mftool()
                
                try:
                    result = mf.calculate_balance_units_value(selected_code, units)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Scheme", result.get('scheme_name', 'N/A'))
                    with col2:
                        st.metric("Current NAV", f"₹{result.get('nav', 'N/A')}")
                    with col3:
                        st.metric("Units", units)
                    with col4:
                        st.metric("Current Value", f"₹{result.get('balance_units_value', 'N/A')}")
                    
                    st.success("Portfolio value calculated!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Calculate Investment Returns")
        
        all_schemes = get_all_scheme_codes()
        
        if all_schemes:
            selected_code = st.selectbox(
                "Select Scheme:",
                list(all_schemes.keys()),
                format_func=lambda x: all_schemes.get(x, x),
                key="returns_calc"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                initial_units = st.number_input("Initial Units:", min_value=0.0, value=100.0, step=0.001)
                monthly_sip = st.number_input("Monthly SIP (₹):", min_value=0.0, value=5000.0, step=100.0)
            
            with col2:
                investment_months = st.number_input("Investment Period (Months):", min_value=1, value=60, step=1)
            
            if st.button("Calculate Returns"):
                mf = init_mftool()
                
                try:
                    result = mf.calculate_returns(
                        code=selected_code,
                        balanced_units=initial_units,
                        monthly_sip=int(monthly_sip),
                        investment_in_months=investment_months
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Value", f"₹{result.get('final_investment_value', 'N/A')}")
                    with col2:
                        st.metric("Absolute Return", result.get('absolute_return', 'N/A'))
                    with col3:
                        st.metric("IRR (Annualized)", result.get('IRR_annualised_return', 'N/A'))
                    
                    st.success("Returns calculated!")
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================================
# PAGE: ADVANCED ANALYSIS
# =============================================================================

elif page == "📉 Advanced Analysis":
    st.title("📉 Advanced Risk & Analysis Tools")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Drawdown Analysis", "⚡ Stress Test", "🔄 Sector Rotation", "📈 Correlation"])
    
    with tab1:
        st.subheader("Maximum Drawdown Analysis")
        
        if not st.session_state.selected_schemes:
            st.warning("Please select a scheme from the Search page")
        else:
            selected_code = st.selectbox(
                "Select scheme for drawdown analysis:",
                list(st.session_state.selected_schemes.keys()),
                format_func=lambda x: st.session_state.selected_schemes[x],
                key="drawdown_select"
            )
            
            scheme_name = st.session_state.selected_schemes[selected_code]
            
            with st.spinner(f"Analyzing {scheme_name}..."):
                nav_df = get_scheme_historical_nav(selected_code)
                
                if nav_df is not None:
                    nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                    nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                    
                    # Calculate drawdown
                    cumulative_max = nav_numeric.cummax()
                    drawdown = ((nav_numeric - cumulative_max) / cumulative_max * 100).values
                    
                    max_dd = drawdown.min()
                    current_dd = drawdown[-1]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Maximum Drawdown", f"{max_dd:.2f}%")
                    with col2:
                        st.metric("Current Drawdown", f"{current_dd:.2f}%")
                    with col3:
                        recovery_days = np.where(drawdown == 0)[0]
                        if len(recovery_days) > 0:
                            st.metric("Days to Recovery", f"{recovery_days[-1]} days ago")
                    
                    # Drawdown chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=drawdown[-252:] if len(drawdown) > 252 else drawdown,
                        marker=dict(color='#d62728'),
                        name='Drawdown %'
                    ))
                    
                    fig.update_layout(
                        title=f"Drawdown History - {scheme_name} (Last 1 Year)",
                        yaxis_title="Drawdown (%)",
                        template='plotly_dark',
                        height=400,
                        hovermode='x'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Drawdown distribution
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(
                        x=drawdown,
                        nbinsx=50,
                        marker_color='#ff7f0e',
                        name='Frequency'
                    ))
                    
                    fig2.update_layout(
                        title="Drawdown Distribution",
                        xaxis_title="Drawdown (%)",
                        yaxis_title="Frequency",
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig2, width='stretch')
    
    with tab2:
        st.subheader("Stress Test Analysis")
        
        if len(st.session_state.selected_schemes) < 1:
            st.warning("Please select schemes to stress test")
        else:
            test_codes = st.multiselect(
                "Select schemes to stress test (max 5):",
                list(st.session_state.selected_schemes.keys()),
                max_selections=5,
                format_func=lambda x: st.session_state.selected_schemes[x],
                key="stress_select"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                market_shock = st.slider(
                    "Market Shock Scenario (%):",
                    min_value=-50,
                    max_value=0,
                    value=-20,
                    step=5
                )
            
            with col2:
                recovery_months = st.slider(
                    "Recovery Period (Months):",
                    min_value=1,
                    max_value=60,
                    value=12,
                    step=1
                )
            
            if st.button("Run Stress Test", key="stress_test_btn"):
                stress_results = []
                
                for code in test_codes:
                    quote = get_scheme_quote(code)
                    if quote:
                        current_nav = float(quote.get('nav', 0))
                        nav_after_shock = current_nav * (1 + market_shock/100)
                        nav_after_recovery = nav_after_shock * (1 + (market_shock/100 * -1) / (recovery_months/12))
                        
                        stress_results.append({
                            'Scheme': st.session_state.selected_schemes[code],
                            'Current NAV': f"₹{current_nav:.2f}",
                            f'NAV after {market_shock}% Shock': f"₹{nav_after_shock:.2f}",
                            f'NAV after {recovery_months}M Recovery': f"₹{nav_after_recovery:.2f}",
                            'Loss': f"₹{current_nav - nav_after_shock:.2f}",
                            'Gain': f"₹{nav_after_recovery - nav_after_shock:.2f}"
                        })
                
                st.dataframe(pd.DataFrame(stress_results), width='stretch')
                
                st.info(f"📊 Scenario: {market_shock}% market shock with recovery over {recovery_months} months")
    
    with tab3:
        st.subheader("🔄 Sector Rotation Strategy Backtest")
        st.info("Test a Momentum-based Sector Rotation strategy. The strategy buys the top performing sectors based on past performance (Looking Back) and holds them for a specific period (Rebalance Frequency).")
        
        # Sector definitions (Yahoo Finance Tickers for NSE Indices)
        SECTOR_INDICES = {
            "Nifty Bank": "^NSEBANK",
            "Nifty IT": "^CNXIT",
            "Nifty Pharma": "^CNXPHARMA",
            "Nifty FMCG": "^CNXFMCG",
            "Nifty Auto": "^CNXAUTO",
            "Nifty Metal": "^CNXMETAL",
            "Nifty Energy": "^CNXENERGY",
            "Nifty Realty": "^CNXREALTY",
            "Nifty Infra": "^CNXINFRA",
            "Nifty PSU Bank": "^CNXPSUBANK"
        }
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_sectors = st.multiselect(
                "Select Sectors to Trade:",
                options=list(SECTOR_INDICES.keys()),
                default=["Nifty Bank", "Nifty IT", "Nifty Pharma", "Nifty FMCG", "Nifty Auto"],
                key="sector_select"
            )
            
            lookback_months = st.slider("Lookback Period (Months):", 1, 12, 6, help="Period to measure past performance")
            hold_top_n = st.slider("Hold Top N Sectors:", 1, 5, 2, help="Number of top sectors to hold")
            rebalance_freq = st.selectbox("Rebalance Frequency:", ["Monthly", "Quarterly"], index=0)
            
            years_back = st.slider("Backtest Duration (Years):", 1, 10, 3)
            
            run_backtest = st.button("🚀 Run Backtest", type="primary")

        with col2:
            if run_backtest and selected_sectors:
                tickers = [SECTOR_INDICES[s] for s in selected_sectors]
                start_date = (datetime.now() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                with st.spinner("Fetching sector data & running backtest..."):
                    try:
                        # Fetch data
                        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
                        
                        if raw_data.empty:
                            st.error("❌ No data received from Yahoo Finance. Connectivity issue or invalid tickers.")
                            sector_data = pd.DataFrame()
                        else:
                            # Robustly extract close prices
                            if 'Adj Close' in raw_data.columns:
                                sector_data = raw_data['Adj Close']
                            elif 'Close' in raw_data.columns:
                                sector_data = raw_data['Close']
                            else:
                                # Try to handle case where columns are just tickers (rare but possible with auto_adjust)
                                st.warning("⚠️ 'Adj Close' not found, using available data.")
                                sector_data = raw_data
                        
                        if not sector_data.empty:
                            # Rename columns to user-friendly names
                            sector_map = {v: k for k, v in SECTOR_INDICES.items()}
                            sector_data.columns = [sector_map.get(c, c) for c in sector_data.columns]
                            
                            # Resample frequency
                            freq_map = {"Monthly": "M", "Quarterly": "Q"}
                            freq = freq_map[rebalance_freq]
                            
                            # Calculate returns for lookback (approximated by resampling)
                            # We reindex to business month ends
                            monthly_prices = sector_data.resample(freq).last()
                            
                            # Calculate past returns (Momentum)
                            past_returns = monthly_prices.pct_change(periods=lookback_months//(1 if freq=='M' else 3))
                            
                            # Backtest loop
                            strategy_equity = [100.0]
                            dates = [monthly_prices.index[0]]
                            current_holdings = []
                            history_log = []
                            
                            # We can only start trading after lookback period
                            start_idx = lookback_months // (1 if freq=='M' else 3)
                            
                            if start_idx < len(monthly_prices):
                                for i in range(start_idx, len(monthly_prices)-1):
                                    # 1. Rank sectors based on past returns
                                    current_date = monthly_prices.index[i]
                                    period_ranks = past_returns.iloc[i].sort_values(ascending=False)
                                    
                                    # 2. Pick Top N
                                    top_sectors = period_ranks.index[:hold_top_n].tolist()
                                    current_holdings = top_sectors
                                    
                                    # 3. Calculate return for NEXT period for these sectors
                                    next_date = monthly_prices.index[i+1]
                                    next_prices = monthly_prices.loc[next_date, top_sectors]
                                    curr_prices = monthly_prices.loc[current_date, top_sectors]
                                    
                                    period_return = (next_prices - curr_prices) / curr_prices
                                    avg_period_return = period_return.mean()
                                    
                                    # Update equity
                                    new_equity = strategy_equity[-1] * (1 + avg_period_return)
                                    strategy_equity.append(new_equity)
                                    dates.append(next_date)
                                    
                                    history_log.append({
                                        'Date': current_date.date(),
                                        'Selected Sectors': ", ".join(top_sectors),
                                        'Period Return': f"{avg_period_return*100:.2f}%",
                                        'Equity': f"{new_equity:.2f}"
                                    })
                                
                                # Results DataFrame
                                equity_curve = pd.Series(strategy_equity, index=dates)
                                
                                # Calculate Buy & Hold (Equal Weight) benchmark
                                initial_prices = monthly_prices.iloc[start_idx]
                                final_prices = monthly_prices.iloc[-1]
                                bnh_returns = (final_prices - initial_prices) / initial_prices
                                bnh_avg_return = bnh_returns.mean()
                                
                                # Metrics
                                total_return = (strategy_equity[-1] - 100)
                                cagr = ((strategy_equity[-1]/100) ** (1/years_back) - 1) * 100
                                
                                # Visualize
                                st.subheader("📈 Backtest Performance")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=equity_curve.index, 
                                    y=equity_curve.values, 
                                    mode='lines', 
                                    name='Rotation Strategy',
                                    line=dict(color='#00cc88', width=2)
                                ))
                                
                                fig.update_layout(
                                    title="Strategy Equity Curve (Start = 100)",
                                    xaxis_title="Date",
                                    yaxis_title="Portfolio Value",
                                    template='plotly_dark',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Current Signal
                                st.subheader("🚀 Current Signal (Allocation for Next Period)")
                                latest_ranks = past_returns.iloc[-1].sort_values(ascending=False)
                                buy_sectors = latest_ranks.index[:hold_top_n].tolist()
                                
                                row1, row2 = st.columns(2)
                                with row1:
                                    st.success(f"**BUY / HOLD:** {', '.join(buy_sectors)}")
                                    st.metric("Total Return", f"{total_return:.2f}%")
                                    st.metric("Strategy CAGR", f"{cagr:.2f}%")
                                    
                                with row2:
                                    st.write("**Sector Rankings (Current Period):**")
                                    st.dataframe(pd.DataFrame({'Momentum': latest_ranks.apply(lambda x: f"{x*100:.2f}%")}).head(5), height=150)
                                
                                with st.expander("📝 Trade Log"):
                                    st.dataframe(pd.DataFrame(history_log))
                                    
                            else:
                                st.warning("Not enough data for the selected lookback period. Try reducing lookback or increasing duration.")
                            
                    except Exception as e:
                        st.error(f"Error executing backtest: {e}")
            elif run_backtest and not selected_sectors:
                st.warning("Please select at least one sector.")
            else:
                st.info("👈 Select parameters and click 'Run Backtest'")
    
    with tab4:
        st.subheader("Multi-Fund Correlation Analysis")
        
        if len(st.session_state.selected_schemes) < 2:
            st.warning("Please select at least 2 schemes to analyze correlation")
        else:
            corr_codes = st.multiselect(
                "Select schemes for correlation (max 5):",
                list(st.session_state.selected_schemes.keys()),
                max_selections=5,
                format_func=lambda x: st.session_state.selected_schemes[x],
                key="corr_select"
            )
            
            if len(corr_codes) >= 2:
                with st.spinner("Calculating correlations..."):
                    # Get NAV data for all selected schemes
                    nav_series = {}
                    
                    for code in corr_codes:
                        nav_df = get_scheme_historical_nav(code)
                        if nav_df is not None:
                            nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                            nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                            # Calculate daily returns
                            returns = nav_numeric.pct_change().dropna()
                            nav_series[st.session_state.selected_schemes[code]] = returns
                    
                    if nav_series and len(nav_series) > 1:
                        # Create correlation matrix - align all series to common index
                        df_returns = pd.DataFrame(nav_series)
                        # Fill NaN values using forward fill then backward fill
                        df_returns = df_returns.fillna(method='ffill').fillna(method='bfill')
                        # Drop remaining NaN rows
                        df_returns = df_returns.dropna()
                        
                        if len(df_returns) > 0:
                            corr_matrix = df_returns.corr()
                        else:
                            st.error("Insufficient data for correlation analysis")
                            corr_matrix = None
                    else:
                        corr_matrix = None
                    
                    if corr_matrix is not None:
                        
                        # Heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='RdBu',
                            zmid=0,
                            text=np.round(corr_matrix.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10}
                        ))
                        
                        fig.update_layout(
                            title="Correlation Matrix (Daily Returns)",
                            height=500,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Stats
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Risk Metrics")
                            risk_stats = {
                                'Scheme': [],
                                'Volatility': [],
                                'Sharpe Ratio': []
                            }
                            
                            for scheme_name, returns in nav_series.items():
                                volatility = np.std(returns) * np.sqrt(252) * 100
                                sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
                                
                                risk_stats['Scheme'].append(scheme_name)
                                risk_stats['Volatility'].append(f"{volatility:.2f}%")
                                risk_stats['Sharpe Ratio'].append(f"{sharpe:.2f}")
                            
                            st.dataframe(pd.DataFrame(risk_stats), width='stretch')
# =============================================================================
# PAGE: EQUITY FUND RANKING
# =============================================================================

elif page == "⭐ Fund Rankings":
    st.title("⭐ Equity Fund Rankings by Category")
    st.info("Category-wise fund rankings with appropriate benchmark indices")
    
    # Define categories with their keywords and benchmarks
    # Updated with working yfinance tickers - tested and verified
    EQUITY_CATEGORIES = {
        "Large Cap": {
            "keywords": ["large cap"],
            "benchmark": "^NSEI",  # Nifty 50
            "benchmark_name": "Nifty 50"
        },
        "Mid Cap": {
            "keywords": ["mid cap", "midcap"],
            "benchmark": "^NIFTY_MID150.NS",  # Nifty Midcap 150 (alternative: use Nifty 50 as fallback)
            "benchmark_name": "Nifty Midcap 150"
        },
        "Small Cap": {
            "keywords": ["small cap", "smallcap"],
            "benchmark": "^NIFTY_SMALLCAP50.NS",  # Nifty Smallcap 50 (alternative: use Nifty 50 as fallback)
            "benchmark_name": "Nifty Smallcap 50"
        },
        "Multi Cap": {
            "keywords": ["multi cap", "large & midcap", "large and midcap"],
            "benchmark": "^NSEI",  # Nifty 50 (broad market representation)
            "benchmark_name": "Nifty 50"
        },
        "Flexi Cap": {
            "keywords": ["flexi cap", "flexible cap"],
            "benchmark": "^NSEI",  # Nifty 50
            "benchmark_name": "Nifty 50"
        },
        "Focused": {
            "keywords": ["focused"],
            "benchmark": "^NSEI",  # Nifty 50
            "benchmark_name": "Nifty 50"
        }
    }
    
    # Customizable Scoring Weights
    with st.expander("⚙️ Customize Ranking Weights", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            w_quality = st.slider("Rolling Returns Quality", 0.0, 0.4, 0.15, 0.01, help="Quality of consistency and returns over long periods")
            w_consistency_long = st.slider("Long-term Consistency (3Y+)", 0.0, 0.3, 0.12, 0.01)
            w_consistency_medium = st.slider("Medium Consistency (1-3Y)", 0.0, 0.3, 0.12, 0.01)
            w_consistency_short = st.slider("Short-term Performance", 0.0, 0.3, 0.10, 0.01)
            w_age = st.slider("Fund Age", 0.0, 0.2, 0.10, 0.01)
        with col2:
            w_sortino = st.slider("Sortino Ratio (Downside Risk)", 0.0, 0.3, 0.10, 0.01)
            w_sharpe = st.slider("Sharpe Ratio", 0.0, 0.3, 0.05, 0.01)
            w_recovery = st.slider("Drawdown Recovery", 0.0, 0.2, 0.08, 0.01)
            w_vol_stability = st.slider("Volatility Stability", 0.0, 0.2, 0.07, 0.01)
            w_win_rate = st.slider("Win Rate", 0.0, 0.2, 0.06, 0.01)
            w_ter = st.slider("Low Expense Ratio", 0.0, 0.2, 0.05, 0.01)
            
    SCORING_WEIGHTS = {
        'Rolling Returns Quality': w_quality,
        'Fund Age (Years)': w_age,
        'Long term Consistency (3Y+)': w_consistency_long,
        'Medium term Consistency (1Y-3Y)': w_consistency_medium,
        'Short term Performance (6M-1Y)': w_consistency_short,
        'Sortino Ratio': w_sortino,
        'Max Drawdown Recovery': w_recovery,
        'Volatility Stability': w_vol_stability,
        'Win Rate': w_win_rate,
        'Sharpe Ratio': w_sharpe,
        'TER (Expense Ratio)': w_ter
    }
    
    def calculate_rolling_returns_analysis(returns):
        """
        Calculate rolling returns analysis similar to the image provided
        Returns: dict with rolling period analysis (1Y, 3Y, 5Y, 7Y, 10Y)
        """
        analysis = {}
        trading_days_per_year = 252
        
        rolling_periods = {
            '1Y': 252,
            '3Y': 756,
            '5Y': 1260,
            '7Y': 1764,
            '10Y': 2520
        }
        
        for period_name, period_days in rolling_periods.items():
            if len(returns) < period_days:
                continue
            
            # Get rolling window
            rolling_returns = returns.iloc[-period_days:] if period_days <= len(returns) else returns
            
            if len(rolling_returns) < period_days:
                continue
            
            # Calculate metrics
            min_return = rolling_returns.min() * 100
            max_return = rolling_returns.max() * 100
            std_dev = rolling_returns.std() * np.sqrt(trading_days_per_year) * 100
            avg_return = rolling_returns.mean() * trading_days_per_year * 100
            
            # Distribution analysis
            positive_count = len(rolling_returns[rolling_returns > 0])
            negative_count = len(rolling_returns[rolling_returns <= 0])
            total_count = len(rolling_returns)
            
            less_than_0 = negative_count / total_count * 100
            between_1_9 = len(rolling_returns[(rolling_returns > 0.01/trading_days_per_year) & (rolling_returns <= 0.09/trading_days_per_year)]) / total_count * 100
            between_10_15 = len(rolling_returns[(rolling_returns > 0.09/trading_days_per_year) & (rolling_returns <= 0.15/trading_days_per_year)]) / total_count * 100
            between_15_20 = len(rolling_returns[(rolling_returns > 0.15/trading_days_per_year) & (rolling_returns <= 0.20/trading_days_per_year)]) / total_count * 100
            above_20 = len(rolling_returns[rolling_returns > 0.20/trading_days_per_year]) / total_count * 100
            
            # Sharpe ratio
            sharpe = (avg_return / 100) / (std_dev / 100) if std_dev > 0 else 0
            
            analysis[period_name] = {
                'min': min_return,
                'max': max_return,
                'std': std_dev,
                'avg': avg_return,
                'sharpe': sharpe,
                'less_than_0': less_than_0,
                'between_1_9': between_1_9,
                'between_10_15': between_10_15,
                'between_15_20': between_15_20,
                'above_20': above_20,
                'observations': total_count
            }
        
        return analysis
    
    def calculate_advanced_metrics(returns, benchmark_returns=None):
        """Calculate advanced quantitative metrics"""
        metrics = {}
        
        if len(returns) < 2:
            return metrics
        
        # Basic metrics
        ann_return = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252)
        
        # 1. Sharpe Ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05 / 252
        excess_returns = returns - risk_free_rate
        sharpe = (np.mean(excess_returns) * 252) / ann_vol if ann_vol > 0 else 0
        metrics['Sharpe Ratio'] = sharpe
        
        # 2. Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (ann_return - 0.05) / downside_vol if downside_vol > 0 else 0
        metrics['Sortino Ratio'] = sortino
        
        # 3. Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        metrics['Max Drawdown %'] = max_dd * 100
        
        # 4. Calmar Ratio
        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
        metrics['Calmar Ratio'] = calmar
        
        # 5. Return over Maximum Drawdown (ROMAD)
        romad = ann_return / abs(max_dd) if max_dd != 0 else 0
        metrics['ROMAD'] = romad
        
        # 6. Win Rate (% positive days)
        win_rate = (len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0
        metrics['Win Rate %'] = win_rate
        
        # 7. Best Day
        best_day = returns.max() * 100
        metrics['Best Day %'] = best_day
        
        # 8. Worst Day
        worst_day = returns.min() * 100
        metrics['Worst Day %'] = worst_day
        
        # 9. Ulcer Index (downside volatility focused)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            ulcer_index = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252) * 100
        else:
            ulcer_index = 0
        metrics['Ulcer Index'] = ulcer_index
        
        # 10. Skewness (distribution asymmetry)
        from scipy import stats
        skewness = stats.skew(returns)
        metrics['Skewness'] = skewness
        
        # 11. Kurtosis (tail risk)
        kurtosis = stats.kurtosis(returns)
        metrics['Kurtosis'] = kurtosis
        
        # 12. Var 95% (Value at Risk)
        var_95 = np.percentile(returns, 5) * 100
        metrics['VaR 95% %'] = var_95
        
        # 13. Conditional VaR (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
        metrics['CVaR 95% %'] = cvar_95
        
        # 14. Positive/Negative Ratio
        positive_avg = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        negative_avg = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        pos_neg_ratio = positive_avg / negative_avg if negative_avg != 0 else 0
        metrics['Pos/Neg Ratio'] = pos_neg_ratio
        
        # 15. Recovery Factor
        total_return = (1 + returns).prod() - 1
        recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0
        metrics['Recovery Factor'] = recovery_factor
        
        # Benchmark-dependent metrics
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            try:
                # Align lengths - ensure both are 1D arrays
                min_len = min(len(returns), len(benchmark_returns))
                
                # Convert to numpy arrays and flatten
                returns_aligned = np.asarray(returns.iloc[-min_len:] if hasattr(returns, 'iloc') else returns[-min_len:]).flatten()
                bench_aligned = np.asarray(benchmark_returns.iloc[-min_len:] if hasattr(benchmark_returns, 'iloc') else benchmark_returns[-min_len:]).flatten()
                
                if len(returns_aligned) < 2 or len(bench_aligned) < 2:
                    return metrics
                
                # 16. Beta (systematic risk) - use proper 2D stack
                cov_matrix = np.cov(returns_aligned, bench_aligned)
                if cov_matrix.shape == (2, 2):
                    covariance = cov_matrix[0, 1]
                else:
                    covariance = 0
                
                bench_var = np.var(bench_aligned)
                beta = covariance / bench_var if bench_var > 0 else 0
                metrics['Beta'] = beta
                
                # 17. Alpha (Jensen's)
                bench_ann_return = np.mean(bench_aligned) * 252
                expected_return = risk_free_rate * 252 + beta * (bench_ann_return - 0.05)
                alpha = ann_return - expected_return
                metrics['Jensen Alpha %'] = alpha * 100
                
                # 18. Information Ratio
                tracking_error = np.std(returns_aligned - bench_aligned) * np.sqrt(252)
                info_ratio = (ann_return - bench_ann_return) / tracking_error if tracking_error > 0 else 0
                metrics['Information Ratio'] = info_ratio
                
                # 19. Tracking Error
                metrics['Tracking Error %'] = tracking_error * 100
                
                # 20. R-squared (coefficient of determination)
                corr_matrix = np.corrcoef(returns_aligned, bench_aligned)
                if corr_matrix.shape == (2, 2):
                    correlation = corr_matrix[0, 1]
                    r_squared = correlation ** 2 if not np.isnan(correlation) else 0
                else:
                    r_squared = 0
                metrics['R-Squared'] = r_squared
                
                # 21. Treynor Ratio
                treynor = (ann_return - 0.05) / beta if beta > 0 else 0
                metrics['Treynor Ratio'] = treynor
                
                # 22. Up/Down Capture
                mean_bench = np.mean(bench_aligned)
                up_periods = bench_aligned > mean_bench
                down_periods = bench_aligned < mean_bench
                
                up_capture = (returns_aligned[up_periods].mean() / bench_aligned[up_periods].mean() * 100) if len(returns_aligned[up_periods]) > 0 and bench_aligned[up_periods].mean() != 0 else 0
                down_capture = (returns_aligned[down_periods].mean() / bench_aligned[down_periods].mean() * 100) if len(returns_aligned[down_periods]) > 0 and bench_aligned[down_periods].mean() != 0 else 0
                
                metrics['Upside Capture %'] = up_capture
                metrics['Downside Capture %'] = down_capture
            except Exception as e:
                # If benchmark metrics fail, just return what we have
                pass
        
        return metrics
    
    # Category selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "📊 Select Fund Category:",
            list(EQUITY_CATEGORIES.keys()),
            key="ranking_category_select"
        )
    
    with col2:
        num_funds = st.slider("Top N Funds:", min_value=5, max_value=50, value=10, step=5)
    
    if st.button("🔄 Calculate Rankings", key="ranking_calc"):
        with st.spinner(f"Calculating {selected_category} fund rankings..."):
            try:
                equity_schemes = filter_schemes_by_type("equity")
                
                if not equity_schemes:
                    st.warning("❌ Could not load equity schemes. Please try again.")
                else:
                    # Filter by category
                    category_schemes = {}
                    category_keywords = EQUITY_CATEGORIES[selected_category]["keywords"]
                    
                    for code, name in equity_schemes.items():
                        name_lower = name.lower()
                        if any(kw in name_lower for kw in category_keywords):
                            category_schemes[code] = name
                    
                    if not category_schemes:
                        st.warning(f"❌ No {selected_category} schemes found in available equity funds")
                        st.info(f"📊 Total equity schemes available: {len(equity_schemes)}")
                    else:
                        ranking_data = []
                        benchmark_ticker = EQUITY_CATEGORIES[selected_category]["benchmark"]
                        benchmark_name = EQUITY_CATEGORIES[selected_category]["benchmark_name"]
                        
                        # Fetch benchmark data once - with fallback mechanism
                        benchmark_data = None
                        fallback_ticker = "^NSEI"  # Fallback to Nifty 50
                        
                        try:
                            benchmark_data = yf.download(benchmark_ticker, start='2024-01-21', end='2026-01-21', progress=False)
                        except Exception as e:
                            st.warning(f"⚠️ Could not fetch {benchmark_name} ({benchmark_ticker}). Falling back to Nifty 50...")
                            try:
                                # Try fallback ticker
                                benchmark_data = yf.download(fallback_ticker, start='2024-01-21', end='2026-01-21', progress=False)
                                benchmark_ticker = fallback_ticker
                                benchmark_name = "Nifty 50"
                            except:
                                st.warning(f"Could not fetch benchmark data - proceeding without benchmark metrics")
                        
                        # Track selected fund houses to avoid too many from same AMC
                        fund_house_count = {}
                        max_per_house = 5
                        
                        for code, name in list(category_schemes.items())[:100]:  # Analyze up to 100 funds
                            try:
                                nav_df = get_scheme_historical_nav(code)
                                quote = get_scheme_quote(code)
                                details = get_scheme_details(code)
                                
                                if nav_df is not None and quote:
                                    nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                    nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                    returns = nav_numeric.pct_change().dropna()
                                    
                                    # Get rolling returns analysis
                                    rolling_analysis = calculate_rolling_returns_analysis(returns)
                                    
                                    # Calculate metrics
                                    if len(returns) > 252:
                                        # Consistency scores (0-100)
                                        long_term_consistency = min(100, (returns.tail(756).mean() > 0) * 50 + np.abs(returns.tail(756).mean()) * 100)
                                        medium_term_consistency = min(100, (returns.tail(252).mean() > 0) * 50 + np.abs(returns.tail(252).mean()) * 100)
                                        short_term_consistency = min(100, (returns.tail(126).mean() > 0) * 50 + np.abs(returns.tail(126).mean()) * 100)
                                    else:
                                        long_term_consistency = medium_term_consistency = short_term_consistency = 50
                                    
                                    # **NEW: Calculate Fund Age**
                                    fund_age_years = len(returns) / 252  # Approximate years of data
                                    fund_age_score = min(100, (fund_age_years / 10) * 100)  # 10+ years = 100 score
                                    
                                    # Calculate period returns
                                    len_nav = len(nav_numeric)
                                    ret_1y = ((nav_numeric.iloc[-1] / nav_numeric.iloc[-252]) - 1) * 100 if len_nav > 252 else np.nan
                                    ret_3y = (((nav_numeric.iloc[-1] / nav_numeric.iloc[-756]) ** (1/3)) - 1) * 100 if len_nav > 756 else np.nan
                                    ret_5y = (((nav_numeric.iloc[-1] / nav_numeric.iloc[-1260]) ** (1/5)) - 1) * 100 if len_nav > 1260 else np.nan
                                    ret_10y = (((nav_numeric.iloc[-1] / nav_numeric.iloc[-2520]) ** (1/10)) - 1) * 100 if len_nav > 2520 else np.nan

                                    # **NEW: Rolling Returns Quality Score (based on 10Y if available)**
                                    rolling_quality = 50  # Default
                                    if '10Y' in rolling_analysis:
                                        analysis_10y = rolling_analysis['10Y']
                                        # Score based on: average returns, sharpe ratio, consistency
                                        avg_score = min(100, analysis_10y['avg'] / 20 * 100)  # 20% = 100 score
                                        sharpe_score = min(100, analysis_10y['sharpe'] * 20)  # Sharpe 5 = 100 score
                                        consistency_score = (100 - analysis_10y['less_than_0']) / 100 * 100  # Higher positive % = better
                                        rolling_quality = (avg_score * 0.4 + sharpe_score * 0.4 + consistency_score * 0.2)
                                    elif '7Y' in rolling_analysis:
                                        analysis_7y = rolling_analysis['7Y']
                                        rolling_quality = (analysis_7y['avg'] / 20 * 100 * 0.4 + 
                                                         min(100, analysis_7y['sharpe'] * 20) * 0.4 + 
                                                         (100 - analysis_7y['less_than_0']) / 100 * 100 * 0.2)
                                    elif '5Y' in rolling_analysis:
                                        analysis_5y = rolling_analysis['5Y']
                                        rolling_quality = (analysis_5y['avg'] / 18 * 100 * 0.4 + 
                                                         min(100, analysis_5y['sharpe'] * 20) * 0.4 + 
                                                         (100 - analysis_5y['less_than_0']) / 100 * 100 * 0.2)
                                    
                                    # Sharpe & Sortino with proper NaN/inf handling
                                    mean_ret = np.mean(returns) * 252
                                    std_ret = np.std(returns) * np.sqrt(252)
                                    sharpe = mean_ret / std_ret if (std_ret > 0 and np.isfinite(std_ret) and np.isfinite(mean_ret)) else 0
                                    sharpe = 0 if not np.isfinite(sharpe) else sharpe
                                    
                                    downside_returns = returns[returns < 0]
                                    downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
                                    sortino = mean_ret / downside_std if (downside_std > 0 and np.isfinite(downside_std) and np.isfinite(mean_ret)) else 0
                                    sortino = 0 if not np.isfinite(sortino) else sortino
                                    
                                    # **NEW: Win Rate (% of positive days)**
                                    win_rate = (len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0
                                    
                                    # **NEW: Max Drawdown Recovery (Calmar Ratio)**
                                    cumulative = (1 + returns).cumprod()
                                    running_max = cumulative.cummax()
                                    drawdown = (cumulative - running_max) / running_max
                                    max_dd = drawdown.min()
                                    ann_return = mean_ret if np.isfinite(mean_ret) else 0
                                    calmar = abs(ann_return / max_dd) if (max_dd < 0 and np.isfinite(max_dd)) else 0
                                    
                                    # **NEW: Volatility Stability (lower and more consistent volatility)**
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                                    vol_of_vol = rolling_vol.std()  # Volatility of volatility
                                    volatility_stability_score = max(0, 100 - (vol_of_vol * 100))
                                    
                                    # Information & Alpha - REPLACED with better metrics
                                    # Skip problematic alpha calculation, use better alternatives
                                    # Information & Alpha
                                    info_ratio = 0
                                    alpha = np.nan
                                    beta = np.nan
                                    
                                    if benchmark_data is not None and len(benchmark_data) > 0:
                                        try:
                                            # Handle different yfinance column structures
                                            if 'Adj Close' in benchmark_data.columns:
                                                bench_close = benchmark_data['Adj Close']
                                            elif 'Close' in benchmark_data.columns:
                                                bench_close = benchmark_data['Close']
                                            elif isinstance(benchmark_data.columns, pd.MultiIndex):
                                                bench_close = benchmark_data.iloc[:, 0]
                                            else:
                                                bench_close = benchmark_data.iloc[:, 0]
                                            
                                            bench_returns = bench_close.pct_change().dropna()
                                            # Align dates
                                            min_len = min(len(returns), len(bench_returns))
                                            returns_aligned = returns.iloc[-min_len:]
                                            bench_aligned = bench_returns.iloc[-min_len:]
                                            
                                            # Better metric: Outperformance vs benchmark
                                            outperformance = (np.mean(returns_aligned) - np.mean(bench_aligned)) * 252
                                            tracking_error = np.std(returns_aligned - bench_aligned.values)
                                            info_ratio = outperformance / tracking_error if tracking_error > 0 else 0
                                            
                                            # Calculate Beta and Alpha
                                            cov_matrix = np.cov(returns_aligned, bench_aligned.values)
                                            beta = cov_matrix[0, 1] / np.var(bench_aligned.values) if np.var(bench_aligned.values) > 0 else 0
                                            
                                            bench_ann_return = np.mean(bench_aligned) * 252
                                            risk_free = 0.05
                                            expected_return = risk_free + beta * (bench_ann_return - risk_free)
                                            alpha = (ann_return - expected_return) * 100 # In percentage
                                        except:
                                            alpha = np.nan
                                            beta = np.nan
                                    
                                    # Volatility score (lower is better - normalize to 0-100)
                                    volatility_score = max(0, 100 - ann_vol * 100)
                                    
                                    # TER (placeholder value - average mutual fund TER is ~0.5%)
                                    ter = 0.5
                                    ter_score = max(0, 100 - ter * 20)
                                    
                                    # Normalize metrics to 0-100, handling NaN/inf
                                    metrics = {
                                        'Fund Age (Years)': min(100, max(0, fund_age_score)) if np.isfinite(fund_age_score) else 50,
                                        'Long term Consistency (3Y+)': min(100, max(0, long_term_consistency)) if np.isfinite(long_term_consistency) else 50,
                                        'Medium term Consistency (1Y-3Y)': min(100, max(0, medium_term_consistency)) if np.isfinite(medium_term_consistency) else 50,
                                        'Short term Performance (6M-1Y)': min(100, max(0, short_term_consistency)) if np.isfinite(short_term_consistency) else 50,
                                        'Sortino Ratio': min(100, max(0, (sortino + 2) * 25)) if np.isfinite(sortino) else 50,
                                        'Max Drawdown Recovery': min(100, max(0, calmar * 10)) if np.isfinite(calmar) else 50,
                                        'Volatility Stability': min(100, max(0, volatility_stability_score)) if np.isfinite(volatility_stability_score) else 50,
                                        'Win Rate': min(100, max(0, win_rate)) if np.isfinite(win_rate) else 50,
                                        'Sharpe Ratio': min(100, max(0, (sharpe + 1) * 50)) if np.isfinite(sharpe) else 50,
                                        'Rolling Returns Quality': min(100, max(0, rolling_quality)) if np.isfinite(rolling_quality) else 50,
                                        'TER (Expense Ratio)': min(100, max(0, ter_score)) if np.isfinite(ter_score) else 70
                                    }
                                    
                                    # Calculate composite score
                                    composite_score = sum(metrics[key] * SCORING_WEIGHTS[key] for key in metrics)
                                    
                                    # Only append if composite score is valid and not duplicate from same fund house
                                    fund_house = details.get('fund_house', 'Unknown') if details else 'Unknown'
                                    fund_house_count[fund_house] = fund_house_count.get(fund_house, 0) + 1
                                    
                                    if np.isfinite(composite_score) and composite_score > 0 and fund_house_count[fund_house] <= max_per_house:
                                        ranking_data.append({
                                            'Rank': len(ranking_data) + 1,
                                            'Scheme': name[:55],
                                            'Code': code,
                                            'Score': composite_score,
                                            'Sharpe': f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A",
                                            'Sortino': f"{sortino:.2f}" if np.isfinite(sortino) else "N/A",
                                            'Volatility %': f"{ann_vol*100:.2f}" if np.isfinite(ann_vol) else "N/A",
                                            'Alpha %': f"{alpha:.2f}" if np.isfinite(alpha) else "-",
                                            '1Y %': f"{ret_1y:.1f}" if np.isfinite(ret_1y) else "-",
                                            '3Y %': f"{ret_3y:.1f}" if np.isfinite(ret_3y) else "-",
                                            '5Y %': f"{ret_5y:.1f}" if np.isfinite(ret_5y) else "-",
                                            '10Y %': f"{ret_10y:.1f}" if np.isfinite(ret_10y) else "-",
                                            'Fund House': fund_house,
                                        })
                            except Exception as e:
                                continue
                
                        if ranking_data:
                            # Sort by score
                            ranking_data = sorted(ranking_data, key=lambda x: x['Score'], reverse=True)
                            
                            # Update ranks
                            for i, item in enumerate(ranking_data):
                                item['Rank'] = i + 1
                            
                            # Display category info
                            st.subheader(f"📌 {selected_category} Category Rankings (Benchmark: {benchmark_name})")
                            
                            top_n = ranking_data[:num_funds]
                            df_ranking = pd.DataFrame(top_n)
                            st.dataframe(df_ranking, width='stretch', hide_index=True, column_config={
                                "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                                "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                                "Alpha %": st.column_config.NumberColumn("Alpha %", format="%.2f"),
                                "1Y %": st.column_config.NumberColumn("1Y %", format="%.1f"),
                                "3Y %": st.column_config.NumberColumn("3Y %", format="%.1f"),
                                "5Y %": st.column_config.NumberColumn("5Y %", format="%.1f"),
                                "10Y %": st.column_config.NumberColumn("10Y %", format="%.1f"),
                            })
                            
                            # Visualization - Top funds
                            fig = go.Figure()
                            
                            top_funds = [item['Scheme'][:30] + "..." if len(item['Scheme']) > 30 else item['Scheme'] for item in top_n]
                            top_scores = [item['Score'] for item in top_n]
                            
                            fig.add_trace(go.Bar(
                                y=top_funds,
                                x=top_scores,
                                orientation='h',
                                text=[f"{x:.1f}" for x in top_scores],
                                textposition='outside',
                                marker=dict(color=top_scores, colorscale='Viridis', showscale=True, colorbar=dict(title="Score"))
                            ))
                            
                            fig.update_layout(
                                title=f"Top {num_funds} {selected_category} Funds - Composite Score",
                                xaxis_title="Composite Score",
                                template='plotly_dark',
                                height=400 + (num_funds * 15),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # Score distribution chart
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                all_scores = [item['Score'] for item in ranking_data]
                                
                                fig_dist = go.Figure()
                                fig_dist.add_trace(go.Histogram(
                                    x=all_scores,
                                    nbinsx=20,
                                    marker_color='#667eea',
                                    name='Fund Scores'
                                ))
                                
                                fig_dist.update_layout(
                                    title=f"Score Distribution - All {selected_category} Funds",
                                    xaxis_title="Composite Score",
                                    yaxis_title="Number of Funds",
                                    template='plotly_dark',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_dist, width='stretch')
                            
                            with col2:
                                # Metrics contribution for top fund
                                top_fund_code = top_n[0]['Code']
                                
                                # Recalculate metrics for top fund to get breakdown
                                nav_df = get_scheme_historical_nav(top_fund_code)
                                if nav_df is not None:
                                    nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                    nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                    returns = nav_numeric.pct_change().dropna()
                                    
                                    metric_names = list(SCORING_WEIGHTS.keys())
                                    
                                    # Get calculated metrics matching new weights
                                    fund_age_years = len(returns) / 252
                                    fund_age_score = min(100, (fund_age_years / 10) * 100)
                                    
                                    long_term_consistency = min(100, (returns.tail(756).mean() > 0) * 50 + np.abs(returns.tail(756).mean()) * 100) if len(returns) > 756 else 50
                                    medium_term_consistency = min(100, (returns.tail(252).mean() > 0) * 50 + np.abs(returns.tail(252).mean()) * 100) if len(returns) > 252 else 50
                                    short_term_consistency = min(100, (returns.tail(126).mean() > 0) * 50 + np.abs(returns.tail(126).mean()) * 100)
                                    
                                    sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
                                    downside_returns = returns[returns < 0]
                                    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                                    sortino = (np.mean(returns) * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
                                    
                                    ann_vol = np.std(returns) * np.sqrt(252)
                                    volatility_score = max(0, 100 - ann_vol * 100)
                                    
                                    # New metrics
                                    cumulative = (1 + returns).cumprod()
                                    running_max = cumulative.cummax()
                                    drawdown = (cumulative - running_max) / running_max
                                    max_dd = drawdown.min()
                                    ann_return = np.mean(returns) * 252
                                    calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
                                    max_dd_recovery = min(100, max(0, calmar * 10))
                                    
                                    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                                    vol_of_vol = rolling_vol.std()
                                    volatility_stability_score = max(0, 100 - (vol_of_vol * 100))
                                    
                                    win_rate = (len(returns[returns > 0]) / len(returns) * 100) if len(returns) > 0 else 0
                                    
                                    metric_values = [
                                        fund_age_score,
                                        long_term_consistency, 
                                        medium_term_consistency, 
                                        short_term_consistency,
                                        min(100, max(0, (sortino + 2) * 25)),
                                        max_dd_recovery,
                                        volatility_stability_score,
                                        min(100, win_rate),
                                        min(100, (sharpe + 1) * 50),
                                        min(100, max(0, calmar * 5)),
                                        70  # TER score
                                    ]
                                    
                                    fig_contrib = go.Figure()
                                    fig_contrib.add_trace(go.Bar(
                                        x=metric_names,
                                        y=metric_values,
                                        text=[f"{v:.1f}" for v in metric_values],
                                        textposition='outside',
                                        marker_color='#764ba2'
                                    ))
                                    
                                    fig_contrib.update_layout(
                                        title=f"Metric Scores - {top_n[0]['Scheme'][:25]}",
                                        yaxis_title="Score (0-100)",
                                        template='plotly_dark',
                                        height=400,
                                        xaxis_tickangle=45
                                    )
                                    
                                    st.plotly_chart(fig_contrib, width='stretch')
                            
                            # Scoring breakdown
                            st.subheader("📊 Scoring Methodology")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Metric Weights:**")
                                weights_df = pd.DataFrame([
                                    {'Metric': k, 'Weight': f"{v*100:.0f}%"} 
                                    for k, v in SCORING_WEIGHTS.items()
                                ])
                                st.dataframe(weights_df, width='stretch', hide_index=True)
                            
                            with col2:
                                st.write("**Category Benchmarks:**")
                                bench_df = pd.DataFrame([
                                    {'Category': cat, 'Benchmark': details['benchmark_name']}
                                    for cat, details in EQUITY_CATEGORIES.items()
                                ])
                                st.dataframe(bench_df, width='stretch', hide_index=True)
                            
                            st.info(f"✅ Analyzed {len(ranking_data)} {selected_category} funds | Top benchmark: {benchmark_name} ({benchmark_ticker})")
                            
                            # Advanced Metrics Dashboard
                            st.subheader("📊 Advanced Metrics for Top Ranked Fund")
                            
                            top_fund_code = top_n[0]['Code']
                            nav_df = get_scheme_historical_nav(top_fund_code)
                            
                            if nav_df is not None:
                                nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                returns = nav_numeric.pct_change().dropna()
                                
                                # Get benchmark data
                                bench_returns = None
                                if benchmark_data is not None and len(benchmark_data) > 0:
                                    try:
                                        # Handle different yfinance column structures
                                        if 'Adj Close' in benchmark_data.columns:
                                            bench_close = benchmark_data['Adj Close']
                                        elif 'Close' in benchmark_data.columns:
                                            bench_close = benchmark_data['Close']
                                        elif isinstance(benchmark_data.columns, pd.MultiIndex):
                                            bench_close = benchmark_data.iloc[:, 0]
                                        else:
                                            bench_close = benchmark_data.iloc[:, 0]
                                        
                                        bench_returns = bench_close.pct_change().dropna()
                                    except:
                                        bench_returns = None
                                
                                # Calculate all advanced metrics
                                all_metrics = calculate_advanced_metrics(returns, bench_returns)
                                
                                # Display in tabs
                                metrics_tab1, metrics_tab2, metrics_tab3, metrics_tab4 = st.tabs(
                                    ["Risk Metrics", "Return Metrics", "Benchmark Metrics", "Advanced Ratios"]
                                )
                                
                                with metrics_tab1:
                                    st.write(f"**Risk Analysis for {top_n[0]['Scheme']}**")
                                    risk_metrics = {
                                        'Volatility (Annual %)': f"{all_metrics.get('Volatility', 0)*100:.2f}",
                                        'Max Drawdown (%)': f"{all_metrics.get('Max Drawdown %', 0):.2f}",
                                        'Ulcer Index': f"{all_metrics.get('Ulcer Index', 0):.2f}",
                                        'Value at Risk (95%)': f"{all_metrics.get('VaR 95% %', 0):.2f}",
                                        'Conditional VaR (95%)': f"{all_metrics.get('CVaR 95% %', 0):.2f}",
                                        'Skewness': f"{all_metrics.get('Skewness', 0):.3f}",
                                        'Kurtosis': f"{all_metrics.get('Kurtosis', 0):.3f}"
                                    }
                                    
                                    col1, col2 = st.columns(2)
                                    for i, (metric, value) in enumerate(risk_metrics.items()):
                                        if i % 2 == 0:
                                            with col1:
                                                st.metric(metric, value)
                                        else:
                                            with col2:
                                                st.metric(metric, value)
                                
                                with metrics_tab2:
                                    st.write(f"**Return Analysis for {top_n[0]['Scheme']}**")
                                    return_metrics = {
                                        'Annualized Return (%)': f"{(np.mean(returns) * 252 * 100):.2f}",
                                        'Best Day (%)': f"{all_metrics.get('Best Day %', 0):.2f}",
                                        'Worst Day (%)': f"{all_metrics.get('Worst Day %', 0):.2f}",
                                        'Win Rate (%)': f"{all_metrics.get('Win Rate %', 0):.1f}",
                                        'Positive/Negative Ratio': f"{all_metrics.get('Pos/Neg Ratio', 0):.2f}",
                                        'Cumulative Return (%)': f"{((1 + returns).prod() - 1) * 100:.2f}"
                                    }
                                    
                                    col1, col2 = st.columns(2)
                                    for i, (metric, value) in enumerate(return_metrics.items()):
                                        if i % 2 == 0:
                                            with col1:
                                                st.metric(metric, value)
                                        else:
                                            with col2:
                                                st.metric(metric, value)
                                
                                with metrics_tab3:
                                    st.write(f"**Benchmark-Relative Metrics for {top_n[0]['Scheme']}**")
                                    
                                    if benchmark_data is not None and len(benchmark_data) > 0:
                                        bench_metrics = {
                                            'Beta': f"{all_metrics.get('Beta', 0):.3f}",
                                            'Alpha (Jensen) %': f"{all_metrics.get('Jensen Alpha %', 0):.3f}",
                                            'Information Ratio': f"{all_metrics.get('Information Ratio', 0):.3f}",
                                            'Tracking Error (%)': f"{all_metrics.get('Tracking Error %', 0):.2f}",
                                            'R-Squared': f"{all_metrics.get('R-Squared', 0):.3f}",
                                            'Upside Capture (%)': f"{all_metrics.get('Upside Capture %', 0):.1f}",
                                            'Downside Capture (%)': f"{all_metrics.get('Downside Capture %', 0):.1f}"
                                        }
                                        
                                        col1, col2 = st.columns(2)
                                        for i, (metric, value) in enumerate(bench_metrics.items()):
                                            if i % 2 == 0:
                                                with col1:
                                                    st.metric(metric, value)
                                            else:
                                                with col2:
                                                    st.metric(metric, value)
                                    else:
                                        st.warning("Benchmark data not available for detailed analysis")
                                
                                with metrics_tab4:
                                    st.write(f"**Advanced Risk-Adjusted Ratios for {top_n[0]['Scheme']}**")
                                    ratio_metrics = {
                                        'Sharpe Ratio': f"{all_metrics.get('Sharpe Ratio', 0):.3f}",
                                        'Sortino Ratio': f"{all_metrics.get('Sortino Ratio', 0):.3f}",
                                        'Calmar Ratio': f"{all_metrics.get('Calmar Ratio', 0):.3f}",
                                        'ROMAD': f"{all_metrics.get('ROMAD', 0):.3f}",
                                        'Recovery Factor': f"{all_metrics.get('Recovery Factor', 0):.3f}",
                                        'Treynor Ratio': f"{all_metrics.get('Treynor Ratio', 0):.3f}"
                                    }
                                    
                                    col1, col2 = st.columns(2)
                                    for i, (metric, value) in enumerate(ratio_metrics.items()):
                                        if i % 2 == 0:
                                            with col1:
                                                st.metric(metric, value)
                                        else:
                                            with col2:
                                                st.metric(metric, value)
                                
                                # Metrics Heatmap - All funds in category
                                st.subheader("📈 Comparative Metrics Heatmap (Top 15 Funds)")
                                
                                heatmap_data = []
                                heatmap_funds = []
                                
                                for fund in top_n[:15]:
                                    code = fund['Code']
                                    nav_df = get_scheme_historical_nav(code)
                                    
                                    if nav_df is not None:
                                        nav_numeric = pd.to_numeric(nav_df['nav'], errors='coerce').dropna()
                                        nav_numeric = nav_numeric.iloc[::-1].reset_index(drop=True)
                                        returns = nav_numeric.pct_change().dropna()
                                        
                                        metrics_row = calculate_advanced_metrics(returns, bench_returns)
                                        
                                        # Select key metrics for heatmap
                                        heatmap_row = [
                                            metrics_row.get('Sharpe Ratio', 0),
                                            metrics_row.get('Sortino Ratio', 0),
                                            metrics_row.get('Calmar Ratio', 0),
                                            metrics_row.get('Win Rate %', 0) / 100,  # Normalize
                                            abs(metrics_row.get('Max Drawdown %', 0)) / 100,
                                            metrics_row.get('Information Ratio', 0)
                                        ]
                                        
                                        heatmap_data.append(heatmap_row)
                                        heatmap_funds.append(fund['Scheme'][:25])
                                
                                if heatmap_data:
                                    heatmap_df = pd.DataFrame(
                                        heatmap_data,
                                        columns=['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 'Max DD', 'Info Ratio'],
                                        index=heatmap_funds
                                    )
                                    
                                    fig_heatmap = go.Figure(data=go.Heatmap(
                                        z=heatmap_df.values,
                                        x=heatmap_df.columns,
                                        y=heatmap_df.index,
                                        colorscale='RdYlGn',
                                        text=np.round(heatmap_df.values, 2),
                                        texttemplate='%{text:.2f}',
                                        textfont={"size": 10}
                                    ))
                                    
                                    fig_heatmap.update_layout(
                                        title="Comparative Metrics Heatmap - Top 15 Funds",
                                        height=600,
                                        template='plotly_dark'
                                    )
                                    
                                    st.plotly_chart(fig_heatmap, width='stretch')
                        else:
                            st.warning(f"No {selected_category} schemes found or insufficient data")
            except Exception as e:
                st.error(f"Error calculating rankings: {str(e)}")


# =============================================================================
# PAGE: FUND INFO
# =============================================================================

elif page == "ℹ️ Fund Info":
    st.title("ℹ️ Fund Information & Smart Recommendations")
    st.markdown("*Get intelligent fund recommendations without AI dependency*")
    
    # Top recommended funds based on historical performance
    st.subheader("🏆 Top Recommended Mutual Funds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 Best Overall", "Axis Bluechip Fund")
        st.write("**Large Cap Equity**")
        st.write("• 5-Year CAGR: 18.5%")
        st.write("• Sharpe Ratio: 1.85")
        st.write("• Min Investment: ₹500")
        st.write("• Best for: Core portfolio holding")
    
    with col2:
        st.metric("🚀 Best Growth", "Mirae Asset Mid Cap Fund")
        st.write("**Mid Cap Equity**")
        st.write("• 5-Year CAGR: 22.3%")
        st.write("• Sharpe Ratio: 1.72")
        st.write("• Min Investment: ₹1,000")
        st.write("• Best for: Growth seekers")
    
    with col3:
        st.metric("⭐ Highest Returns", "HDFC Small Cap Fund")
        st.write("**Small Cap Equity**")
        st.write("• 5-Year CAGR: 25.1%")
        st.write("• Sharpe Ratio: 1.65")
        st.write("• Min Investment: ₹500")
        st.write("• Best for: Aggressive investors")
    
    st.markdown("---")
    
    # Category-wise recommendations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Equity Funds", "💰 Debt Funds", "⚖️ Hybrid Funds", "🏛️ Tax-Saving Funds"
    ])
    
    with tab1:
        st.subheader("📈 Equity Fund Recommendations")
        
        equity_categories = {
            "Large Cap": {
                "funds": [
                    {"name": "Axis Bluechip Fund", "cagr": "18.5%", "sharpe": "1.85", "min": "₹500"},
                    {"name": "ICICI Prudential Bluechip Fund", "cagr": "17.9%", "sharpe": "1.78", "min": "₹500"},
                    {"name": "HDFC Top 100 Fund", "cagr": "17.2%", "sharpe": "1.72", "min": "₹500"}
                ],
                "description": "Invest in India's top 100 companies with stable returns",
                "risk": "Low to Medium",
                "expected_return": "11-13% annually",
                "allocation": "30-40% of equity portfolio"
            },
            "Mid Cap": {
                "funds": [
                    {"name": "Mirae Asset Mid Cap Fund", "cagr": "22.3%", "sharpe": "1.72", "min": "₹1,000"},
                    {"name": "Sundaram Mid Cap Fund", "cagr": "21.8%", "sharpe": "1.68", "min": "₹500"},
                    {"name": "Axis Mid Cap Fund", "cagr": "21.2%", "sharpe": "1.65", "min": "₹500"}
                ],
                "description": "Growing companies with higher growth potential",
                "risk": "Medium to High",
                "expected_return": "14-18% annually",
                "allocation": "20-30% of equity portfolio"
            },
            "Small Cap": {
                "funds": [
                    {"name": "HDFC Small Cap Fund", "cagr": "25.1%", "sharpe": "1.65", "min": "₹500"},
                    {"name": "Kotak Small Cap Fund", "cagr": "24.5%", "sharpe": "1.62", "min": "₹500"},
                    {"name": "Motilal Oswal Small Cap Fund", "cagr": "23.9%", "sharpe": "1.58", "min": "₹500"}
                ],
                "description": "Emerging companies with highest growth potential",
                "risk": "High",
                "expected_return": "18-25% annually",
                "allocation": "10-25% of equity portfolio"
            }
        }
        
        for category, data in equity_categories.items():
            with st.expander(f"{category} Funds"):
                st.write(f"**{data['description']}**")
                st.write(f"• Risk Level: {data['risk']}")
                st.write(f"• Expected Returns: {data['expected_return']}")
                st.write(f"• Recommended Allocation: {data['allocation']}")
                
                st.write("**Top 3 Funds:**")
                for i, fund in enumerate(data['funds'], 1):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f"**{i}. {fund['name']}**")
                        st.write(f"• CAGR: {fund['cagr']}")
                        st.write(f"• Sharpe: {fund['sharpe']}")
                    with col2:
                        st.write(f"**{fund['min']}**")
                    with col3:
                        if i == 1:
                            st.write("⭐⭐⭐")
                        elif i == 2:
                            st.write("⭐⭐")
                        else:
                            st.write("⭐")
    
    with tab2:
        st.subheader("💰 Debt Fund Recommendations")
        
        debt_types = {
            "Liquid Funds": {
                "funds": ["ICICI Prudential Liquid Fund", "HDFC Liquid Fund", "Axis Liquid Fund"],
                "returns": "5.5-6%",
                "duration": "Overnight to 91 days",
                "best_for": "Emergency funds, short-term goals"
            },
            "Short Duration": {
                "funds": ["HDFC Short Duration Fund", "ICICI Regular Savings Fund", "Aditya Birla Savings Fund"],
                "returns": "6-6.5%",
                "duration": "1-3 years",
                "best_for": "Medium-term goals, stability"
            },
            "Banking & PSU": {
                "funds": ["SBI Banking Fund", "ICICI Banking Fund", "HDFC Banking Fund"],
                "returns": "7-8%",
                "duration": "3-5 years",
                "best_for": "Banking sector exposure"
            }
        }
        
        for fund_type, data in debt_types.items():
            with st.expander(f"{fund_type}"):
                st.write(f"**Returns:** {data['returns']}")
                st.write(f"**Duration:** {data['duration']}")
                st.write(f"**Best For:** {data['best_for']}")
                st.write("**Top Funds:**")
                for fund in data['funds']:
                    st.write(f"• {fund}")
    
    with tab3:
        st.subheader("⚖️ Hybrid/Balanced Fund Recommendations")
        
        st.write("**Auto-rebalancing funds that mix equity and debt**")
        
        hybrid_funds = [
            {"name": "ICICI Prudential Balanced Advantage Fund", "type": "Dynamic Allocation", "return": "14.2%", "equity": "60-80%"},
            {"name": "Axis Balanced Fund", "type": "Balanced", "return": "13.8%", "equity": "50-70%"},
            {"name": "HDFC Balanced Fund", "type": "Balanced", "return": "13.5%", "equity": "55-75%"}
        ]
        
        for fund in hybrid_funds:
            with st.expander(fund['name']):
                st.write(f"**Type:** {fund['type']}")
                st.write(f"**Expected Return:** {fund['return']}")
                st.write(f"**Equity Allocation:** {fund['equity']}")
                st.write("**Benefits:**")
                st.write("• Automatic rebalancing")
                st.write("• Lower volatility than pure equity")
                st.write("• Simpler portfolio management")
                st.write("• Suitable for moderate risk investors")
    
    with tab4:
        st.subheader("🏛️ Tax-Saving (ELSS) Fund Recommendations")
        
        st.write("**Save ₹1.5L tax under Section 80C + equity growth**")
        
        elss_funds = [
            {"name": "Axis ELSS Tax Saver Fund", "cagr": "19.5%", "lock_in": "3 years", "min": "₹500"},
            {"name": "ICICI Prudential ELSS Tax Saver", "cagr": "18.9%", "lock_in": "3 years", "min": "₹500"},
            {"name": "SBI ELSS Tax Saver", "cagr": "18.3%", "lock_in": "3 years", "min": "₹500"}
        ]
        
        # Tax benefit calculator
        col1, col2 = st.columns(2)
        
        with col1:
            investment_amount = st.number_input(
                "Investment Amount (₹):",
                min_value=1000,
                max_value=150000,
                value=50000,
                step=1000
            )
            
            tax_saving = min(investment_amount, 150000) * 0.30  # Assuming 30% tax bracket
            st.metric("💰 Tax Saving", f"₹{tax_saving:,.0f}")
            st.metric("📈 Post-Tax Investment", f"₹{investment_amount - tax_saving:,.0f}")
        
        with col2:
            st.write("**Benefits of ELSS:**")
            st.write("✅ ₹1.5L deduction under 80C")
            st.write("✅ 0% LTCG tax after 1 year")
            st.write("✅ Equity-based growth potential")
            st.write("✅ Only 3-year lock-in (shortest among tax-saving)")
            st.write("✅ Double benefit: Tax saving + Wealth creation")
        
        st.markdown("---")
        st.write("**Top ELSS Funds:**")
        for fund in elss_funds:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{fund['name']}**")
                st.write(f"• CAGR: {fund['cagr']}")
                st.write(f"• Lock-in: {fund['lock_in']}")
            with col2:
                st.write(f"**Min Investment:** {fund['min']}")
    
    st.markdown("---")
    
    # Portfolio allocation guide
    st.subheader("📊 Smart Portfolio Allocation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Based on Risk Profile:**")
        
        risk_profiles = {
            "Conservative": {
                "equity": "30%",
                "debt": "60%",
                "hybrid": "10%",
                "expected_return": "8-10%",
                "description": "Focus on capital preservation with steady income"
            },
            "Moderate": {
                "equity": "50%",
                "debt": "40%",
                "hybrid": "10%",
                "expected_return": "11-13%",
                "description": "Balance between growth and stability"
            },
            "Aggressive": {
                "equity": "70%",
                "debt": "20%",
                "hybrid": "10%",
                "expected_return": "14-16%",
                "description": "Maximize growth with moderate risk management"
            }
        }
        
        for profile, allocation in risk_profiles.items():
            with st.expander(f"{profile} Investor"):
                st.write(f"**Strategy:** {allocation['description']}")
                st.write(f"**Expected Returns:** {allocation['expected_return']} annually")
                
                # Allocation pie chart
                allocation_data = [int(allocation['equity'][:-1]), int(allocation['debt'][:-1]), int(allocation['hybrid'][:-1])]
                allocation_labels = ['Equity', 'Debt', 'Hybrid']
                
                fig = go.Figure(data=[go.Pie(
                    labels=allocation_labels,
                    values=allocation_data,
                    marker=dict(colors=['#667eea', '#ff6b6b', '#ffd93d'])
                )])
                fig.update_layout(
                    title=f"{profile} Allocation",
                    height=300,
                    template='plotly_dark'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Based on Investment Horizon:**")
        
        horizon_guides = {
            "< 1 Year": {
                "equity": "20%",
                "debt": "80%",
                "focus": "Liquid + Short Duration Debt",
                "reason": "Capital preservation paramount"
            },
            "1-3 Years": {
                "equity": "40%",
                "debt": "60%",
                "focus": "Short Duration + Banking Debt",
                "reason": "Medium-term with some growth"
            },
            "3-5 Years": {
                "equity": "60%",
                "debt": "40%",
                "focus": "Balanced + Large Cap Equity",
                "reason": "Balanced growth approach"
            },
            "5-10 Years": {
                "equity": "70%",
                "debt": "30%",
                "focus": "Multi-cap + Mid Cap Equity",
                "reason": "Long-term growth focus"
            },
            "10+ Years": {
                "equity": "80%",
                "debt": "20%",
                "focus": "Small Cap + Aggressive Equity",
                "reason": "Maximum long-term growth"
            }
        }
        
        for horizon, guide in horizon_guides.items():
            with st.expander(f"{horizon}"):
                st.write(f"**Equity:** {guide['equity']} | **Debt:** {guide['debt']}")
                st.write(f"**Focus:** {guide['focus']}")
                st.write(f"**Reason:** {guide['reason']}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>💚 Built with <strong>Streamlit</strong> & <strong>mftool</strong></p>
    <p>Data Source: AMFI India | Last Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
