# utils/visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

class DisasterVisualizer:
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_data(self, tweet_path, youtube_comment_path):
        """Load data from both sources"""
        self.tweets_df = pd.read_csv(tweet_path) if tweet_path else pd.DataFrame()
        self.comments_df = pd.read_csv(youtube_comment_path) if youtube_comment_path else pd.DataFrame()
        
        # Try to find date columns with different possible names
        self._process_date_columns()
    
    def _process_date_columns(self):
        """Handle different possible date column names"""
        date_columns = ['date', 'created_at', 'published_at', 'timestamp', 'time']
        
        for df_name in ['tweets_df', 'comments_df']:
            df = getattr(self, df_name)
            if not df.empty:
                # Find which date column exists
                found_date_col = None
                for col in date_columns:
                    if col in df.columns:
                        found_date_col = col
                        break
                
                if found_date_col:
                    # Convert to datetime and extract date
                    df['date'] = pd.to_datetime(df[found_date_col]).dt.date
                    setattr(self, df_name, df)
    
    def create_trend_dashboard(self, output_path='trends_dashboard.png'):
        """Create comprehensive trend dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Disaster Monitoring Trends Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Daily Trend (if date data available)
        self.plot_daily_trends(axes[0, 0])
        
        # Plot 2: Source Distribution
        self.plot_source_distribution(axes[0, 1])
        
        # Plot 3: Disaster Type Distribution
        self.plot_disaster_types(axes[1, 0])
        
        # Plot 4: Regional Distribution
        self.plot_regional_distribution(axes[1, 1])  # Correct
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved to {output_path}")
        plt.show()
    
    def plot_daily_trends(self, ax):
        """Plot daily disaster mentions"""
        all_data = []
        
        # Check if we have date data for tweets
        if not self.tweets_df.empty and 'date' in self.tweets_df.columns:
            tweets_daily = self.tweets_df.groupby('date').size().reset_index(name='count')
            tweets_daily['source'] = 'Twitter'
            all_data.append(tweets_daily)
        
        # Check if we have date data for comments
        if not self.comments_df.empty and 'date' in self.comments_df.columns:
            comments_daily = self.comments_df.groupby('date').size().reset_index(name='count')
            comments_daily['source'] = 'YouTube'
            all_data.append(comments_daily)
        
        if all_data:
            daily_df = pd.concat(all_data)
            
            for source in daily_df['source'].unique():
                source_data = daily_df[daily_df['source'] == source]
                ax.plot(source_data['date'], source_data['count'], 
                       marker='o', linewidth=2, markersize=4, label=source)
            
            ax.set_title('Daily Disaster Mentions Trend', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Mentions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No date data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Daily Trends (No Date Data)', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    def plot_source_distribution(self, ax):
        """Plot data source distribution"""
        sources = []
        counts = []
        
        if not self.tweets_df.empty:
            sources.append('Twitter')
            counts.append(len(self.tweets_df))
        
        if not self.comments_df.empty:
            sources.append('YouTube Comments')
            counts.append(len(self.comments_df))
        
        if sources:
            ax.pie(counts, labels=sources, autopct='%1.1f%%', startangle=90)
            ax.set_title('Data Source Distribution', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Source Distribution (No Data)', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    def plot_disaster_types(self, ax):
        """Plot disaster type distribution"""
        disaster_types = {
            'flood': 0, 'cyclone': 0, 'tsunami': 0, 
            'storm': 0, 'earthquake': 0, 'landslide': 0
        }
        
        # Check both datasets
        for df in [self.tweets_df, self.comments_df]:
            if not df.empty and 'text' in df.columns:
                for text in df['text'].dropna():
                    text_lower = str(text).lower()
                    for disaster in disaster_types:
                        if disaster in text_lower:
                            disaster_types[disaster] += 1
        
        # Plot
        disasters = list(disaster_types.keys())
        counts = list(disaster_types.values())
        
        if sum(counts) > 0:
            ax.bar(disasters, counts, color=sns.color_palette("husl", len(disasters)))
            ax.set_title('Disaster Type Distribution', fontweight='bold')
            ax.set_xlabel('Disaster Type')
            ax.set_ylabel('Count')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'No disaster data found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Disaster Types (No Data)', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    def plot_regional_distribution(self, ax):  # FIXED: This method is NOW INSIDE the class
        """Plot regional distribution with enhanced location detection"""
        # Expanded Indian coastal locations list
        indian_locations = {
            # States and regions
            'maharastra': 'Maharashtra', 'maharashtra': 'Maharashtra',
            'tamil nadu': 'Tamil Nadu', 'tamilnadu': 'Tamil Nadu',
            'kerala': 'Kerala', 'kerela': 'Kerala',
            'karnataka': 'Karnataka', 
            'andhra': 'Andhra Pradesh', 'andhra pradesh': 'Andhra Pradesh',
            'odisha': 'Odisha', 'orissa': 'Odisha',
            'west bengal': 'West Bengal', 'westbengal': 'West Bengal',
            'gujarat': 'Gujarat', 'gujrat': 'Gujarat',
            'goa': 'Goa',
            'pondicherry': 'Puducherry', 'puducherry': 'Puducherry',
            
            # Major coastal cities
            'mumbai': 'Maharashtra', 'bombay': 'Maharashtra',
            'chennai': 'Tamil Nadu', 'madras': 'Tamil Nadu',
            'kolkata': 'West Bengal', 'calcutta': 'West Bengal',
            'kochi': 'Kerala', 'cochin': 'Kerala',
            'visakhapatnam': 'Andhra Pradesh', 'vizag': 'Andhra Pradesh',
            'mangalore': 'Karnataka', 
            'surat': 'Gujarat', 
            'puri': 'Odisha', 'bhubaneswar': 'Odisha',
            'varanasi': 'Uttar Pradesh', 
            'trivandrum': 'Kerala', 'thiruvananthapuram': 'Kerala',
            
            # Coastal areas and landmarks
            'konkan': 'Maharashtra', 
            'malabar': 'Kerala',
            'coromandel': 'Tamil Nadu',
            'sunderban': 'West Bengal', 'sundarbans': 'West Bengal',
            'andaman': 'Andaman Islands', 
            'nicobar': 'Nicobar Islands',
            'lakshadweep': 'Lakshadweep',
            'gateway of india': 'Maharashtra',
            'marine drive': 'Maharashtra',
            'juhu beach': 'Maharashtra',
            'marina beach': 'Tamil Nadu',
            'kovalam beach': 'Kerala',
            'goa beach': 'Goa',
            
            # River deltas and coastal regions
            'ganga': 'Uttar Pradesh', 'ganges': 'Uttar Pradesh',
            'godavari': 'Andhra Pradesh', 
            'krishna': 'Andhra Pradesh',
            'kaveri': 'Tamil Nadu', 'cauvery': 'Tamil Nadu',
            'narmada': 'Gujarat',
            
            # Common misspellings and abbreviations
            'mum': 'Maharashtra', 'bom': 'Maharashtra',
            'chn': 'Tamil Nadu', 'mad': 'Tamil Nadu',
            'kol': 'West Bengal', 'cal': 'West Bengal',
            'hyd': 'Telangana', 
            'blr': 'Karnataka', 'bangalore': 'Karnataka'
        }
        
        location_counts = {}
        total_texts = 0
        found_locations = 0
        
        # Check both datasets
        for df in [self.tweets_df, self.comments_df]:
            if not df.empty and 'text' in df.columns:
                for text in df['text'].dropna():
                    total_texts += 1
                    text_lower = str(text).lower()
                    
                    # Check for each location
                    for location, state in indian_locations.items():
                        if location in text_lower:
                            location_counts[state] = location_counts.get(state, 0) + 1
                            found_locations += 1
                            break  # Count only one location per text
        
        print(f"\n📍 Regional Analysis: Found {found_locations} location mentions in {total_texts} texts")
        
        if location_counts:
            # Convert to lists and sort
            states = list(location_counts.keys())
            counts = list(location_counts.values())
            
            # Sort by count (descending)
            sorted_indices = np.argsort(counts)[::-1]
            states = [states[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(states))
            bars = ax.barh(y_pos, counts, color=sns.color_palette("coolwarm", len(states)))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(states)
            ax.set_xlabel('Number of Mentions')
            ax.set_title('Regional Distribution', fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                       f'{count}', ha='left', va='center')
            
            # Print debug info
            print("   Locations found:")
            for state, count in zip(states, counts):
                print(f"   - {state}: {count} mentions")
                
        else:
            ax.text(0.5, 0.5, 'No regional data found\nTexts may not contain location names', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Regional Distribution', fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Debug info
            print("   ❌ No locations detected.")
            print("   ℹ️  Sample texts:")
            if not self.tweets_df.empty:
                sample_text = self.tweets_df['text'].iloc[0] if len(self.tweets_df) > 0 else "No text available"
                print(f"   Sample tweet: {sample_text[:100]}...")
            if not self.comments_df.empty:
                sample_text = self.comments_df['text'].iloc[0] if len(self.comments_df) > 0 else "No text available"
                print(f"   Sample comment: {sample_text[:100]}...")