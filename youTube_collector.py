import os
import pandas as pd
import googleapiclient.discovery
from dotenv import load_dotenv

load_dotenv()

class YouTubeDataCollector:
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key not found. Please check your .env file")
        
        self.youtube = googleapiclient.discovery.build(
            'youtube', 
            'v3', 
            developerKey=self.api_key,
            static_discovery=False
        )
    
    def search_videos(self, query, max_results=5):
        """Search for videos with error handling"""
        try:
            request = self.youtube.search().list(
                q=query,
                part='snippet',
                maxResults=max_results,
                type='video',
                order='date',
                regionCode='IN'  # Focus on India region
            )
            
            response = request.execute()
            videos = []
            
            for item in response.get('items', []):
                videos.append({
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'published_at': item['snippet']['publishedAt'],
                    'channel_title': item['snippet']['channelTitle'],
                    'search_query': query  # Track which query found this
                })
            
            return videos
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}...")
            return []

    def get_video_comments(self, video_id, max_comments=20):
        """Get comments from a specific video"""
        try:
            comments_data = []
            
            # Get comments from the video
            comments_request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_comments,
                textFormat='plainText',
                order='relevance'  # Get most relevant comments first
            )
            
            comments_response = comments_request.execute()
            
            for item in comments_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comments_data.append({
                    'video_id': video_id,
                    'comment_id': item['id'],
                    'author': comment['authorDisplayName'],
                    'text': comment['textDisplay'],
                    'published_at': comment['publishedAt'],
                    'like_count': comment.get('likeCount', 0),
                    'reply_count': item['snippet'].get('totalReplyCount', 0)
                })
            
            return comments_data
            
        except Exception as e:
            print(f"   ❌ Error getting comments: {str(e)[:100]}...")
            return []

def is_coastal_hazard(text):
    """Check if text contains coastal hazard keywords"""
    if not text or not isinstance(text, str):
        return False
        
    text_lower = text.lower()
    hazard_keywords = [
        'flood', 'cyclone', 'tsunami', 'storm', 'surge', 'wave',
        'coastal', 'beach', 'tide', 'erosion', 'disaster', 'emergency',
        'rain', 'water', 'warning', 'alert', 'high water', 'rough sea',
        'heavy rain', 'landslide', 'evacuation', 'rescue', 'damage',
        'destruction', 'warning', 'alert', 'emergency'
    ]
    return any(keyword in text_lower for keyword in hazard_keywords)

def is_indian_coastal(text):
    """Check if text mentions Indian coastal regions"""
    if not text or not isinstance(text, str):
        return False
        
    text_lower = text.lower()
    
    # Comprehensive Indian coastal indicators
    indian_indicators = [
        # Country and general terms
        'india', 'indian', 'bharat', 'hindustan',
        
        # West Coast States & Cities
        'mumbai', 'maharashtra', 'goa', 'mangalore', 'karnataka', 
        'kochi', 'kerala', 'trivandrum', 'thiruvananthapuram', 'kozhikode',
        'alappuzha', 'kollam', 'kannur', 'malabar',
        
        # East Coast States & Cities  
        'chennai', 'tamil nadu', 'madras', 'pondicherry', 'puducherry',
        'visakhapatnam', 'vizag', 'andhra pradesh', 'kakinada', 'nellore',
        'puri', 'odisha', 'orissa', 'bhubaneswar', 'paradip', 'ganjam',
        'kolkata', 'west bengal', 'digha', 'sagar island', 'haldia',
        
        # Union Territories and Islands
        'andaman', 'nicobar', 'port blair', 'havelock', 'car nicobar',
        'lakshadweep', 'kavaratti', 'minicoy', 'agatti',
        'daman', 'diu', 'daman and diu',
        
        # Coastal Regions
        'konkan', 'malabar coast', 'coromandel', 'sunderbans', 'sundarbans',
        'bay of bengal', 'arabian sea', 'indian ocean'
    ]
    
    return any(indicator in text_lower for indicator in indian_indicators)

def main():
    print("Starting YouTube data collection for Indian coastal disasters...")
    
    # Initialize collector
    try:
        collector = YouTubeDataCollector()
        print("✅ YouTube API client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # FOCUSED SEARCH QUERIES FOR INDIAN COASTAL REGIONS
    search_queries = [
        # State-wise disaster searches
        "flood mumbai maharashtra",
        "cyclone chennai tamil nadu", 
        "tsunami andaman nicobar",
        "storm surge kerala coast",
        "coastal erosion odisha",
        "high waves goa beach",
        "heavy rain kolkata west bengal",
        "cyclone warning andhra pradesh",
        
        # Specific disaster types
        "india flood emergency",
        "indian cyclone alert",
        "storm surge warning india",
        "coastal disaster india",
        
        # Regional specific
        "mumbai rain flood",
        "chennai cyclone damage",
        "kerala tsunami alert",
        "odisha storm surge",
        "andaman earthquake tsunami"
    ]
    
    all_videos = []
    all_comments = []
    
    print(f"🔍 Searching across {len(search_queries)} Indian coastal queries...")
    
    for i, query in enumerate(search_queries):
        print(f"\n📊 Query {i+1}/{len(search_queries)}: '{query}'")
        
        # Search for videos
        videos = collector.search_videos(query, max_results=3)
        
        if videos:
            print(f"   ✅ Found {len(videos)} videos")
            all_videos.extend(videos)
            
            # Get comments for each video
            for video in videos:
                print(f"   💬 Getting comments for: {video['title'][:30]}...")
                comments = collector.get_video_comments(video['video_id'], max_comments=15)
                
                if comments:
                    print(f"   📝 Found {len(comments)} comments")
                    
                    # Add video info to each comment and filter for relevant ones
                    for comment in comments:
                        comment['video_title'] = video['title']
                        comment['video_channel'] = video['channel_title']
                        comment['is_coastal_hazard'] = is_coastal_hazard(comment['text'])
                        comment['is_indian_coastal'] = is_indian_coastal(comment['text'])
                    
                    all_comments.extend(comments)
                
                # Be gentle with API limits
                import time
                time.sleep(1)  # 1 second delay between comment requests
            
        else:
            print("   ❌ No videos found for this query")
        
        # Delay between search queries to avoid rate limits
        if i < len(search_queries) - 1:
            import time
            time.sleep(2)  # 2 second delay between searches
    
    # Save results
    if all_videos:
        videos_df = pd.DataFrame(all_videos)
        videos_df.to_csv('indian_coastal_videos.csv', index=False, encoding='utf-8')
        print(f"\n💾 Saved {len(all_videos)} videos to indian_coastal_videos.csv")
    
    if all_comments:
        comments_df = pd.DataFrame(all_comments)
        comments_df.to_csv('indian_coastal_comments.csv', index=False, encoding='utf-8')
        
        # Filter and analyze relevant comments
        hazard_comments = comments_df[comments_df['is_coastal_hazard']]
        indian_comments = comments_df[comments_df['is_indian_coastal']]
        relevant_comments = comments_df[(comments_df['is_coastal_hazard']) & (comments_df['is_indian_coastal'])]
        
        print(f"💾 Saved {len(comments_df)} comments to indian_coastal_comments.csv")
        print(f"📊 Analysis:")
        print(f"   🌊 Hazard-related comments: {len(hazard_comments)}")
        print(f"   🇮🇳 India-related comments: {len(indian_comments)}")
        print(f"   🎯 Relevant comments (both): {len(relevant_comments)}")
        
        # Show sample of relevant comments
        if not relevant_comments.empty:
            print("\n--- Sample Relevant Comments ---")
            for i, (_, comment) in enumerate(relevant_comments.head(3).iterrows()):
                print(f"\n{i+1}. [{comment['published_at'][:10]}] {comment['author']}:")
                print(f"   {comment['text'][:100]}...")
                print(f"   📹 From: {comment['video_title'][:30]}...")
    
    else:
        print("\n❌ No data collected. Possible issues:")
        print("   - API key not valid")
        print("   - No search results for these queries")
        print("   - API quota exceeded")

if __name__ == "__main__":
    main()