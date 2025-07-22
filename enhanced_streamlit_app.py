"""
Enhanced Streamlit app with all advanced prediction features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from master_predictor import MasterPredictor
from adaptive_seed_manager import AdaptiveSeedManager
from enhanced_draw_fetcher import EnhancedDrawFetcher
from enhanced_scorer import EnhancedScorer
from simulator import simulate_generation
from config import *

st.set_page_config(
    page_title="Enhanced Ozlotter Evolution Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Enhanced Ozlotter Evolution Engine — Advanced AI Prediction System")

@st.cache_resource
def initialize_components():
    return {
        'master_predictor': MasterPredictor(),
        'seed_manager': AdaptiveSeedManager(),
        'draw_fetcher': EnhancedDrawFetcher(),
        'scorer': EnhancedScorer()
    }

components = initialize_components()

st.sidebar.header("🎯 Prediction Configuration")

st.sidebar.subheader("📊 Data Management")
if st.sidebar.button("🔄 Fetch Latest Draws"):
    with st.spinner("Fetching latest lottery draws..."):
        draws_df = components['draw_fetcher'].fetch_draws_with_retry(limit=300)
        if not draws_df.empty:
            st.sidebar.success(f"✅ Fetched {len(draws_df)} draws")
        else:
            st.sidebar.error("❌ Failed to fetch draws")
else:
    draws_df = components['draw_fetcher'].load_local_draws()

if draws_df.empty:
    st.error("❌ No historical draws found. Please fetch draws first.")
    st.stop()

data_info = components['draw_fetcher'].get_data_freshness()
st.sidebar.info(f"📈 Data: {data_info['record_count']} draws\n🕒 Last updated: {data_info['last_modified']}")

seeds, seed_performance = components['seed_manager'].load_seeds_with_performance()
st.sidebar.markdown(f"🌱 **Elite Seeds Loaded:** {len(seeds)}")

st.sidebar.subheader("🧠 AI Prediction Methods")
enable_neural = st.sidebar.checkbox("🤖 Neural Networks (LSTM)", value=True)
enable_patterns = st.sidebar.checkbox("📊 Pattern Analysis", value=True)
enable_chaos = st.sidebar.checkbox("🌀 Chaos Theory", value=True)
enable_psychology = st.sidebar.checkbox("🧠 Market Psychology", value=True)

st.sidebar.subheader("⚙️ Generation Settings")
n_predictions = st.sidebar.slider("🎯 Number of Predictions", 50, 500, 100, step=25)
ensemble_intensity = st.sidebar.slider("🔥 Ensemble Intensity", 0.1, 1.0, 0.7, step=0.1)

with st.sidebar.expander("🔧 Advanced Options"):
    optimize_weights = st.checkbox("⚖️ Auto-optimize Weights", value=False)
    anti_popular_intensity = st.slider("💰 Anti-Popular Strategy", 0.0, 1.0, 0.3, step=0.1)
    show_detailed_analysis = st.checkbox("📈 Show Detailed Analysis", value=True)

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 Launch Enhanced Prediction Engine", type="primary", use_container_width=True):
        with st.spinner("🧠 AI is analyzing patterns and generating predictions..."):
            
            if optimize_weights:
                st.info("⚖️ Optimizing prediction weights...")
                components['master_predictor'].optimize_ensemble_weights(draws_df)
            
            results = components['master_predictor'].generate_ensemble_predictions(
                draws_df=draws_df,
                seed_sets=seeds,
                n_predictions=n_predictions,
                enable_neural=enable_neural,
                enable_chaos=enable_chaos,
                enable_patterns=enable_patterns,
                enable_psychology=enable_psychology
            )
            
            predictions = results['predictions']
            scored_df = results['scored_dataframe']
            method_contributions = results['method_contributions']
            ensemble_info = results['ensemble_info']
            
            st.success(f"✅ Generated {len(predictions)} enhanced predictions using {ensemble_info['total_methods_used']} AI methods!")
            
            st.subheader("🔬 AI Method Contributions")
            contrib_data = pd.DataFrame(list(method_contributions.items()), columns=['Method', 'Predictions'])
            contrib_data = contrib_data[contrib_data['Predictions'] > 0]
            
            fig_contrib = px.pie(contrib_data, values='Predictions', names='Method', 
                               title="Prediction Methods Used")
            st.plotly_chart(fig_contrib, use_container_width=True)
            
            st.subheader("🏆 Top Enhanced Predictions")
            top_predictions = scored_df.head(20)
            
            display_cols = ["ID", "Prediction", "EnhancedScore", "PayoutMultiplier", 
                          "Entropy", "FreqScore", "TemporalScore", "ChaosScore"]
            available_cols = [col for col in display_cols if col in top_predictions.columns]
            
            st.dataframe(
                top_predictions[available_cols].round(4),
                use_container_width=True,
                height=400
            )
            
            st.subheader("🎲 Historical Performance Simulation")
            sim_df = simulate_generation(top_predictions["Prediction"].head(10).tolist(), draws_df)
            st.dataframe(sim_df, use_container_width=True)
            
            if show_detailed_analysis:
                st.subheader("📊 Advanced Analytics")
                
                insights = components['master_predictor'].get_prediction_insights(predictions, draws_df)
                
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistics", "🔍 Patterns", "🧠 Psychology", "🌀 Chaos"])
                
                with tab1:
                    stats = insights['statistical_analysis']
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Total Predictions", stats['total_predictions'])
                        st.metric("Unique Numbers", stats['unique_numbers_used'])
                    
                    with col_b:
                        st.metric("Average Sum", f"{stats['average_sum']:.1f}")
                        st.metric("Sum Std Dev", f"{stats['sum_std']:.1f}")
                    
                    with col_c:
                        st.metric("Even/Odd Ratio", f"{stats['even_odd_ratio']:.2f}")
                        st.metric("Diversity Score", f"{ensemble_info['diversity_score']:.3f}")
                    
                    freq_data = pd.DataFrame(list(stats['most_frequent_numbers'].items()), 
                                           columns=['Number', 'Frequency'])
                    fig_freq = px.bar(freq_data, x='Number', y='Frequency', 
                                    title="Most Frequently Predicted Numbers")
                    st.plotly_chart(fig_freq, use_container_width=True)
                
                with tab2:
                    if 'diversity_metrics' in insights:
                        div_metrics = insights['diversity_metrics']
                        
                        dist_data = {
                            'Range': ['Low (1-15)', 'Mid (16-31)', 'High (32-47)'],
                            'Percentage': [
                                div_metrics['number_distribution']['low_range_percentage'],
                                div_metrics['number_distribution']['mid_range_percentage'],
                                div_metrics['number_distribution']['high_range_percentage']
                            ]
                        }
                        
                        fig_dist = px.bar(pd.DataFrame(dist_data), x='Range', y='Percentage',
                                        title="Number Range Distribution")
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        if 'gap_analysis' in div_metrics and 'status' not in div_metrics['gap_analysis']:
                            gap_info = div_metrics['gap_analysis']
                            col_gap1, col_gap2 = st.columns(2)
                            
                            with col_gap1:
                                st.metric("Average Gap", f"{gap_info['average_gap']:.1f}")
                                st.metric("Gap Consistency", f"{gap_info['gap_consistency']:.3f}")
                            
                            with col_gap2:
                                st.metric("Ideal Gap", f"{gap_info['ideal_gap']:.1f}")
                                st.metric("Gap Std Dev", f"{gap_info['gap_std']:.1f}")
                
                with tab3:
                    if 'psychology_analysis' in insights:
                        psych = insights['psychology_analysis']
                        
                        col_p1, col_p2 = st.columns(2)
                        
                        with col_p1:
                            st.metric("Average Popularity", f"{psych['average_popularity']:.3f}")
                            st.metric("Avg Payout Multiplier", f"{ensemble_info['average_payout_multiplier']:.2f}x")
                        
                        with col_p2:
                            pop_dist = psych['popularity_distribution']
                            st.metric("Low Popularity", pop_dist['low'])
                            st.metric("High Popularity", pop_dist['high'])
                        
                        pop_data = pd.DataFrame([
                            {'Category': 'Low Popularity', 'Count': pop_dist['low']},
                            {'Category': 'Medium Popularity', 'Count': pop_dist['medium']},
                            {'Category': 'High Popularity', 'Count': pop_dist['high']}
                        ])
                        
                        fig_pop = px.pie(pop_data, values='Count', names='Category',
                                       title="Prediction Popularity Distribution")
                        st.plotly_chart(fig_pop, use_container_width=True)
                
                with tab4:
                    if 'chaos_metrics' in insights and insights['chaos_metrics']:
                        chaos_data = []
                        for num, metrics in list(insights['chaos_metrics'].items())[:10]:
                            chaos_data.append({
                                'Number': num,
                                'Chaos Score': metrics['chaos_score'],
                                'Lyapunov': metrics['lyapunov_exponent'],
                                'Fractal Dim': metrics['fractal_dimension']
                            })
                        
                        if chaos_data:
                            chaos_df = pd.DataFrame(chaos_data)
                            
                            fig_chaos = px.scatter(chaos_df, x='Lyapunov', y='Fractal Dim', 
                                                 size='Chaos Score', hover_data=['Number'],
                                                 title="Chaos Theory Analysis")
                            st.plotly_chart(fig_chaos, use_container_width=True)
                            
                            st.dataframe(chaos_df, use_container_width=True)
                    else:
                        st.info("Chaos analysis data not available")

with col2:
    st.subheader("🎯 Quick Actions")
    
    if st.button("📊 Analyze Current Data", use_container_width=True):
        with st.spinner("Analyzing data patterns..."):
            insights = components['master_predictor'].get_prediction_insights([], draws_df)
            
            st.metric("Total Historical Draws", len(draws_df))
            
            if 'statistical_analysis' in insights:
                stats = insights['statistical_analysis']
                st.metric("Unique Numbers in History", stats.get('unique_numbers_used', 'N/A'))
    
    if st.button("🌱 Optimize Seeds", use_container_width=True):
        with st.spinner("Optimizing seed performance..."):
            performance_insights = components['seed_manager'].get_performance_insights()
            
            if 'total_seeds_tracked' in performance_insights:
                st.metric("Seeds Tracked", performance_insights['total_seeds_tracked'])
                st.metric("Best Performance", f"{performance_insights.get('best_performance', 0):.3f}")
            else:
                st.info("No seed performance data available yet")
    
    if st.button("⚖️ Optimize Weights", use_container_width=True):
        with st.spinner("Optimizing prediction weights..."):
            optimized = components['master_predictor'].optimize_ensemble_weights(draws_df)
            
            st.success("✅ Weights optimized!")
            
            weight_data = pd.DataFrame(list(optimized.items()), columns=['Method', 'Weight'])
            fig_weights = px.bar(weight_data, x='Method', y='Weight', 
                               title="Optimized Method Weights")
            st.plotly_chart(fig_weights, use_container_width=True)

st.markdown("---")
st.markdown("""

**🧠 AI Methods:**
- **Neural Networks (LSTM)**: Deep learning sequence prediction
- **Pattern Analysis**: Temporal, positional, and correlation analysis  
- **Chaos Theory**: Lyapunov exponents and fractal analysis
- **Market Psychology**: Anti-popular number strategies
- **Cross-lottery Intelligence**: Global pattern mining
- **Dynamic Evolution**: Adaptive genetic algorithms

**📊 Advanced Analytics:**
- Real-time weight optimization
- Ensemble prediction scoring
- Performance tracking and insights
- Statistical validation framework

**💡 Key Benefits:**
- Maximizes prediction diversity
- Optimizes expected payout multipliers
- Adapts to changing lottery patterns
- Combines multiple AI approaches
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 System Status**")
st.sidebar.info(f"🧠 Neural Engine: {'✅ Ready' if enable_neural else '⏸️ Disabled'}")
st.sidebar.info(f"📊 Pattern Analysis: {'✅ Active' if enable_patterns else '⏸️ Disabled'}")
st.sidebar.info(f"🌀 Chaos Theory: {'✅ Active' if enable_chaos else '⏸️ Disabled'}")
st.sidebar.info(f"🧠 Psychology: {'✅ Active' if enable_psychology else '⏸️ Disabled'}")
