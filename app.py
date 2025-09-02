import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="IPL T20 Cricket Analysis Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Use native Streamlit components only (no HTML/CSS)

@st.cache_data
def load_data():
    """Load and cache all the data files"""
    try:
        matches_df = pd.read_csv('matches.csv')
        deliveries_df = pd.read_csv('deliveries.csv')
        points_table_df = pd.read_csv('points_table.csv')
        winners_df = pd.read_csv('IPL - Winners.csv')
        
        # Convert date columns
        matches_df['date'] = pd.to_datetime(matches_df['date'], format='%d-%m-%Y', errors='coerce')
        
        return matches_df, deliveries_df, points_table_df, winners_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def calculate_batting_stats(deliveries_df, matches_df):
    """Calculate comprehensive batting statistics"""
    # Merge deliveries with matches to get season info
    merged_df = deliveries_df.merge(matches_df[['id', 'season']], left_on='match_id', right_on='id', how='left')
    
    # Calculate batting stats
    batting_stats = deliveries_df.groupby('batter').agg({
        'batsman_runs': ['sum', 'count'],
        'match_id': 'nunique'
    }).round(2)
    
    batting_stats.columns = ['total_runs', 'balls_faced', 'matches_played']
    batting_stats = batting_stats.reset_index()
    
    # Calculate averages and strike rates
    batting_stats['batting_average'] = (batting_stats['total_runs'] / 
                                       batting_stats['matches_played']).round(2)
    batting_stats['strike_rate'] = np.where(
        batting_stats['balls_faced'] > 0,
        (batting_stats['total_runs'] * 100 / batting_stats['balls_faced']).round(2),
        0.0
    )
    
    # Calculate recent form (last 10 matches)
    recent_matches = matches_df.nlargest(10, 'date')['id'].tolist()
    recent_deliveries = deliveries_df[deliveries_df['match_id'].isin(recent_matches)]
    
    recent_stats = recent_deliveries.groupby('batter').agg({
        'batsman_runs': 'sum',
        'match_id': 'nunique'
    }).round(2)
    
    recent_stats.columns = ['recent_runs', 'recent_matches']
    recent_stats = recent_stats.reset_index()
    recent_stats['recent_average'] = np.where(
        recent_stats['recent_matches'] > 0,
        (recent_stats['recent_runs'] / recent_stats['recent_matches']).round(2),
        0.0
    )
    
    # Merge with main stats
    batting_stats = batting_stats.merge(recent_stats, on='batter', how='left')
    batting_stats = batting_stats.fillna(0)
    
    return batting_stats

def calculate_bowling_stats(deliveries_df, matches_df):
    """Calculate comprehensive bowling statistics"""
    # Merge deliveries with matches to get season info
    merged_df = deliveries_df.merge(matches_df[['id', 'season']], left_on='match_id', right_on='id', how='left')
    
    # Calculate bowling stats
    bowling_stats = deliveries_df.groupby('bowler').agg({
        'ball': 'count',
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'match_id': 'nunique'
    }).round(2)
    
    bowling_stats.columns = ['balls_bowled', 'runs_conceded', 'wickets_taken', 'matches_played']
    bowling_stats = bowling_stats.reset_index()
    
    # Calculate economy rate and bowling average
    bowling_stats['overs_bowled'] = (bowling_stats['balls_bowled'] / 6).round(2)
    bowling_stats['economy_rate'] = np.where(
        bowling_stats['balls_bowled'] > 0,
        (bowling_stats['runs_conceded'] * 6 / bowling_stats['balls_bowled']).round(2),
        0.0
    )
    bowling_stats['bowling_average'] = np.where(
        bowling_stats['wickets_taken'] > 0,
        (bowling_stats['runs_conceded'] / bowling_stats['wickets_taken']).round(2),
        0.0
    )
    
    # Calculate recent form (last 10 matches)
    recent_matches = matches_df.nlargest(10, 'date')['id'].tolist()
    recent_deliveries = deliveries_df[deliveries_df['match_id'].isin(recent_matches)]
    
    recent_bowling = recent_deliveries.groupby('bowler').agg({
        'ball': 'count',
        'total_runs': 'sum',
        'is_wicket': 'sum',
        'match_id': 'nunique'
    }).round(2)
    
    recent_bowling.columns = ['recent_balls', 'recent_runs', 'recent_wickets', 'recent_matches']
    recent_bowling = recent_bowling.reset_index()
    recent_bowling['recent_economy'] = np.where(
        recent_bowling['recent_balls'] > 0,
        (recent_bowling['recent_runs'] * 6 / recent_bowling['recent_balls']).round(2),
        0.0
    )
    
    # Merge with main stats
    bowling_stats = bowling_stats.merge(recent_bowling, on='bowler', how='left')
    bowling_stats = bowling_stats.fillna(0)
    
    return bowling_stats

def analyze_team_performance(matches_df, deliveries_df):
    """Analyze team performance and statistics"""
    # Team win-loss records
    team_stats = {}
    teams = list(set(matches_df['team1'].unique()) | set(matches_df['team2'].unique()))
    
    for team in teams:
        # Matches played
        matches_played = len(matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)])
        
        # Matches won
        matches_won = len(matches_df[matches_df['winner'] == team])
        
        # Win percentage
        win_percentage = (matches_won / matches_played * 100) if matches_played > 0 else 0
        
        # Recent form (last 10 matches)
        recent_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)].nlargest(10, 'date')
        recent_wins = len(recent_matches[recent_matches['winner'] == team])
        
        # Batting and bowling performance
        team_batting = deliveries_df[deliveries_df['batting_team'] == team]
        team_bowling = deliveries_df[deliveries_df['bowling_team'] == team]
        
        avg_score = team_batting.groupby('match_id')['total_runs'].sum().mean() if len(team_batting) > 0 else 0
        avg_conceded = team_bowling.groupby('match_id')['total_runs'].sum().mean() if len(team_bowling) > 0 else 0
        
        team_stats[team] = {
            'matches_played': matches_played,
            'matches_won': matches_won,
            'win_percentage': round(win_percentage, 2),
            'recent_wins': recent_wins,
            'recent_matches': len(recent_matches),
            'avg_score': round(avg_score, 2),
            'avg_conceded': round(avg_conceded, 2)
        }
    
    return pd.DataFrame.from_dict(team_stats, orient='index').reset_index().rename(columns={'index': 'team'})

def analyze_head_to_head(matches_df):
    """Analyze head-to-head statistics between teams"""
    h2h_stats = {}
    
    for _, match in matches_df.iterrows():
        team1, team2 = match['team1'], match['team2']
        winner = match['winner']
        
        # Create a sorted key for consistent team pairing
        team_pair = tuple(sorted([team1, team2]))
        
        if team_pair not in h2h_stats:
            h2h_stats[team_pair] = {'team1_wins': 0, 'team2_wins': 0, 'total_matches': 0}
        
        h2h_stats[team_pair]['total_matches'] += 1
        
        if winner == team1:
            h2h_stats[team_pair]['team1_wins'] += 1
        elif winner == team2:
            h2h_stats[team_pair]['team2_wins'] += 1
    
    return h2h_stats

def analyze_trends(deliveries_df, matches_df):
    """Analyze emerging trends and patterns"""
    # Merge data
    merged_df = deliveries_df.merge(matches_df[['id', 'season', 'date']], 
                                   left_on='match_id', right_on='id', how='left')
    
    # Powerplay analysis (overs 1-6)
    powerplay_data = merged_df[merged_df['over'] <= 6]
    powerplay_stats = powerplay_data.groupby('batter').agg({
        'batsman_runs': 'sum',
        'ball': 'count'
    }).round(2)
    powerplay_stats['powerplay_strike_rate'] = (powerplay_stats['batsman_runs'] * 100 / 
                                               powerplay_stats['ball']).round(2)
    
    # Death overs analysis (overs 16-20)
    death_overs_data = merged_df[merged_df['over'] >= 16]
    death_overs_stats = death_overs_data.groupby('batter').agg({
        'batsman_runs': 'sum',
        'ball': 'count'
    }).round(2)
    death_overs_stats['death_overs_strike_rate'] = (death_overs_stats['batsman_runs'] * 100 / 
                                                   death_overs_stats['ball']).round(2)
    
    # Bowling in death overs
    death_overs_bowling = death_overs_data.groupby('bowler').agg({
        'ball': 'count',
        'total_runs': 'sum',
        'is_wicket': 'sum'
    }).round(2)
    death_overs_bowling['death_overs_economy'] = (death_overs_bowling['total_runs'] * 6 / 
                                                 death_overs_bowling['ball']).round(2)
    
    return {
        'powerplay_stats': powerplay_stats.reset_index(),
        'death_overs_stats': death_overs_stats.reset_index(),
        'death_overs_bowling': death_overs_bowling.reset_index()
    }

def main():
    # Load data
    matches_df, deliveries_df, points_table_df, winners_df = load_data()
    
    if matches_df is None:
        st.error("Failed to load data. Please check if the CSV files are in the correct location.")
        return
    
    # Main header
    st.title("🏏 IPL T20 Cricket Analysis Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("📊 Analysis Sections")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["🏠 Overview", "👤 Player Performance", "🏆 Team Analysis", "📈 Trends & Insights", "🏅 Season Winners"]
    )
    
    # Overview Section
    if analysis_type == "🏠 Overview":
        st.header("IPL Overview & Key Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", len(matches_df))
        
        with col2:
            st.metric("Total Seasons", matches_df['season'].nunique())
        
        with col3:
            st.metric("Total Teams", len(set(matches_df['team1'].unique()) | set(matches_df['team2'].unique())))
        
        with col4:
            st.metric("Total Players", len(set(deliveries_df['batter'].unique()) | set(deliveries_df['bowler'].unique())))
        
        # Season-wise matches
        st.subheader("Matches Played by Season")
        season_matches = matches_df['season'].value_counts().sort_index()
        fig = px.bar(x=season_matches.index, y=season_matches.values, 
                    title="Number of Matches by Season",
                    labels={'x': 'Season', 'y': 'Number of Matches'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Venue analysis
        st.subheader("Top Venues")
        venue_matches = matches_df['venue'].value_counts().head(10)
        fig = px.bar(x=venue_matches.values, y=venue_matches.index, orientation='h',
                    title="Top 10 Venues by Number of Matches",
                    labels={'x': 'Number of Matches', 'y': 'Venue'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Player Performance Section
    elif analysis_type == "👤 Player Performance":
        st.header("Player Performance Analysis")
        
        # Calculate stats
        batting_stats = calculate_batting_stats(deliveries_df, matches_df)
        bowling_stats = calculate_bowling_stats(deliveries_df, matches_df)
        
        # Player type selection
        player_type = st.selectbox("Select Player Type", ["Batsmen", "Bowlers"])
        
        if player_type == "Batsmen":
            st.subheader("Top Batsmen Analysis")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_matches = st.slider("Minimum Matches", 1, 50, 5)
            with col2:
                sort_by = st.selectbox("Sort by", ["total_runs", "batting_average", "strike_rate", "recent_average"])
            
            # Filter and sort data
            filtered_batting = batting_stats[batting_stats['matches_played'] >= min_matches].sort_values(sort_by, ascending=False)
            
            # Display top batsmen
            st.subheader(f"Top 10 Batsmen by {sort_by.replace('_', ' ').title()}")
            display_cols = ['batter', 'total_runs', 'matches_played', 'batting_average', 'strike_rate', 'recent_average']
            st.dataframe(filtered_batting[display_cols].head(10), use_container_width=True)
            
            # Batting performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(filtered_batting.head(20), x='batting_average', y='strike_rate', 
                               hover_data=['batter'], title="Batting Average vs Strike Rate")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(filtered_batting.head(10), x='batter', y='total_runs',
                           title="Top 10 Run Scorers")
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Bowlers
            st.subheader("Top Bowlers Analysis")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_matches = st.slider("Minimum Matches", 1, 50, 5)
            with col2:
                sort_by = st.selectbox("Sort by", ["wickets_taken", "bowling_average", "economy_rate", "recent_wickets"])
            
            # Filter and sort data
            filtered_bowling = bowling_stats[bowling_stats['matches_played'] >= min_matches].sort_values(sort_by, ascending=False)
            
            # Display top bowlers
            st.subheader(f"Top 10 Bowlers by {sort_by.replace('_', ' ').title()}")
            display_cols = ['bowler', 'wickets_taken', 'matches_played', 'bowling_average', 'economy_rate', 'recent_wickets']
            st.dataframe(filtered_bowling[display_cols].head(10), use_container_width=True)
            
            # Bowling performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(filtered_bowling.head(20), x='bowling_average', y='economy_rate',
                               hover_data=['bowler'], title="Bowling Average vs Economy Rate")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(filtered_bowling.head(10), x='bowler', y='wickets_taken',
                           title="Top 10 Wicket Takers")
                st.plotly_chart(fig, use_container_width=True)
    
    # Team Analysis Section
    elif analysis_type == "🏆 Team Analysis":
        st.header("Team Performance Analysis")
        
        # Calculate team stats
        team_stats = analyze_team_performance(matches_df, deliveries_df)
        h2h_stats = analyze_head_to_head(matches_df)
        
        # Team selection
        selected_team = st.selectbox("Select Team for Detailed Analysis", team_stats['team'].unique())
        
        # Team overview
        team_data = team_stats[team_stats['team'] == selected_team].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Matches Played", team_data['matches_played'])
        with col2:
            st.metric("Matches Won", team_data['matches_won'])
        with col3:
            st.metric("Win Percentage", f"{team_data['win_percentage']}%")
        with col4:
            st.metric("Recent Wins (Last 10)", f"{team_data['recent_wins']}/{team_data['recent_matches']}")
        
        # Team performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Win percentage comparison
            fig = px.bar(team_stats.sort_values('win_percentage', ascending=False), 
                        x='team', y='win_percentage',
                        title="Team Win Percentages")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average scores
            fig = px.bar(team_stats.sort_values('avg_score', ascending=False),
                        x='team', y='avg_score',
                        title="Average Team Scores")
            st.plotly_chart(fig, use_container_width=True)
        
        # Head-to-head analysis
        st.subheader("Head-to-Head Analysis")
        
        # Find head-to-head data for selected team
        team_h2h = {}
        for pair, stats in h2h_stats.items():
            if selected_team in pair:
                opponent = pair[1] if pair[0] == selected_team else pair[0]
                team_wins = stats['team1_wins'] if pair[0] == selected_team else stats['team2_wins']
                opponent_wins = stats['team2_wins'] if pair[0] == selected_team else stats['team1_wins']
                team_h2h[opponent] = {'wins': team_wins, 'losses': opponent_wins, 'total': stats['total_matches']}
        
        if team_h2h:
            h2h_df = pd.DataFrame.from_dict(team_h2h, orient='index').reset_index()
            h2h_df.columns = ['Opponent', 'Wins', 'Losses', 'Total Matches']
            h2h_df['Win Rate'] = (h2h_df['Wins'] / h2h_df['Total Matches'] * 100).round(2)
            
            st.dataframe(h2h_df, use_container_width=True)
            
            # H2H visualization
            fig = px.bar(h2h_df, x='Opponent', y=['Wins', 'Losses'],
                        title=f"Head-to-Head Record: {selected_team}",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    # Trends & Insights Section
    elif analysis_type == "📈 Trends & Insights":
        st.header("Emerging Trends & Insights")
        
        # Calculate trends
        trends_data = analyze_trends(deliveries_df, matches_df)
        
        # Powerplay specialists
        st.subheader("Powerplay Specialists (Overs 1-6)")
        powerplay_specialists = trends_data['powerplay_stats'].sort_values('powerplay_strike_rate', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(powerplay_specialists[['batter', 'batsman_runs', 'ball', 'powerplay_strike_rate']], 
                        use_container_width=True)
        
        with col2:
            fig = px.bar(powerplay_specialists.head(10), x='batter', y='powerplay_strike_rate',
                        title="Top Powerplay Strike Rates")
            st.plotly_chart(fig, use_container_width=True)
        
        # Death overs specialists
        st.subheader("Death Overs Specialists (Overs 16-20)")
        death_overs_specialists = trends_data['death_overs_stats'].sort_values('death_overs_strike_rate', ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(death_overs_specialists[['batter', 'batsman_runs', 'ball', 'death_overs_strike_rate']], 
                        use_container_width=True)
        
        with col2:
            fig = px.bar(death_overs_specialists.head(10), x='batter', y='death_overs_strike_rate',
                        title="Top Death Overs Strike Rates")
            st.plotly_chart(fig, use_container_width=True)
        
        # Death overs bowling
        st.subheader("Death Overs Bowling Analysis")
        death_bowling = trends_data['death_overs_bowling'].sort_values('death_overs_economy').head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(death_bowling[['bowler', 'ball', 'total_runs', 'is_wicket', 'death_overs_economy']], 
                        use_container_width=True)
        
        with col2:
            fig = px.bar(death_bowling.head(10), x='bowler', y='death_overs_economy',
                        title="Best Death Overs Economy Rates")
            st.plotly_chart(fig, use_container_width=True)
        
        # Over-by-over analysis
        st.subheader("Over-by-Over Run Rate Analysis")
        over_analysis = deliveries_df.groupby('over').agg({
            'total_runs': 'mean',
            'ball': 'count'
        }).reset_index()
        over_analysis['run_rate'] = (over_analysis['total_runs'] * 6 / over_analysis['ball']).round(2)
        
        fig = px.line(over_analysis, x='over', y='run_rate',
                     title="Average Run Rate by Over",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Season Winners Section
    elif analysis_type == "🏅 Season Winners":
        st.header("IPL Season Winners & Records")
        
        # Winners overview
        st.subheader("IPL Champions by Year")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(winners_df, x='Year', y='Orange cap runs',
                        color='Orange Cap', title="Orange Cap Winners & Runs")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(winners_df, x='Year', y='Purple cap wickets',
                        color='Purple Cap', title="Purple Cap Winners & Wickets")
            st.plotly_chart(fig, use_container_width=True)
        
        # Team success analysis
        st.subheader("Most Successful Teams")
        team_wins = winners_df['Winning team'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=team_wins.values, names=team_wins.index,
                        title="IPL Titles Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=team_wins.index, y=team_wins.values,
                        title="Number of IPL Titles by Team")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed winners table
        st.subheader("Complete IPL Winners List")
        display_cols = ['Year', 'Winning team', 'runners up', 'Orange Cap', 'Orange cap runs', 
                       'Purple Cap', 'Purple cap wickets', 'Final Venue']
        st.dataframe(winners_df[display_cols], use_container_width=True)
        
        # Records analysis
        st.subheader("Record Breaking Performances")
        
        col1, col2 = st.columns(2)
        with col1:
            max_runs = winners_df.loc[winners_df['Orange cap runs'].idxmax()]
            st.metric("Highest Orange Cap Runs", f"{max_runs['Orange cap runs']} by {max_runs['Orange Cap']} ({max_runs['Year']})")
        
        with col2:
            max_wickets = winners_df.loc[winners_df['Purple cap wickets'].idxmax()]
            st.metric("Highest Purple Cap Wickets", f"{max_wickets['Purple cap wickets']} by {max_wickets['Purple Cap']} ({max_wickets['Year']})")

if __name__ == "__main__":
    main()
