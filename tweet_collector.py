import tweepy
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()


client = tweepy.Client(bearer_token=os.getenv('TWITTER_BEARER_TOKEN'))


SEARCH_QUERIES = [
    # Broad disaster types covering all India
    "(flood OR cyclone OR tsunami OR storm surge) India -is:retweet",
    
    # State-wise coverage - West Coast
    "(flood OR cyclone) (maharashtra OR mumbai OR goa OR karnataka OR mangalore OR kerala OR kochi) -is:retweet",
    
    # State-wise coverage - East Coast  
    "(flood OR cyclone) (tamil nadu OR chennai OR pondicherry OR andhra pradesh OR visakhapatnam OR vizag OR odisha OR puri OR west bengal OR kolkata) -is:retweet",
    
    # Union Territories and Islands
    "(flood OR cyclone) (andaman OR nicobar OR lakshadweep OR diu OR daman) -is:retweet",
    
    # Specific coastal phenomena
    "(high waves OR coastal erosion OR beach erosion OR tidal waves) (india OR coastal) -is:retweet",
    
    # Major port cities
    "(flood OR storm surge) (mumbai OR chennai OR kochi OR visakhapatnam OR kolkata OR mangalore) -is:retweet",
    
    # Regional specific terms
    "(rain flood OR heavy rain OR monsoon flood) (kerala OR mumbai OR chennai OR goa) -is:retweet",
    
    # Disaster alerts and warnings
    "(warning OR alert OR emergency) (cyclone OR tsunami OR flood) india -is:retweet"
]

def is_coastal_hazard(text):
    """Check if text contains coastal hazard keywords"""
    if not text:
        return False
        
    text_lower = text.lower()
    hazard_keywords = ['wave', 'surge', 'flood', 'cyclone', 'tsunami', 'storm']
    return any(keyword in text_lower for keyword in hazard_keywords)

def is_indian_location(user_location):
    """Check if user location indicates India"""
    if not user_location:
        return False
        
    location_lower = user_location.lower()
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
    
    # Coastal Regions and Geographic Features
    'konkan', 'malabar coast', 'coromandel', 'sunderbans', 'sundarbans',
    'bay of bengal', 'arabian sea', 'indian ocean',
    
    # Major Port Cities
    'mormugao', 'mumbai port', 'chennai port', 'kochi port', 'vizag port',
    'kandla', 'mangalore port', 'tuticorin', 'ennore', 'paradip port',
    
    # Coastal Districts and Towns
    'ratnagiri', 'sindhudurg', 'udupi', 'karwar', 'kumta', 'honnavar',
    'bhatkal', 'murdeshwar', 'kundapura', 'kasaragod', 'kannur', 'thalassery',
    'vadakara', 'koyilandy', 'ernakulam', 'thrissur', 'kottayam', 'pathanamthitta',
    'kanyakumari', 'nagapattinam', 'cuddalore', 'puducherry', 'karaikal',
    'machilipatnam', 'kakinada', 'rajahmundry', 'gopalpur', 'chandipur',
    'digha', 'mandarmani', 'talsari', 'junput',
    
    # River Deltas and Coastal Plains
    'konkan coast', 'kanara coast', 'malabar', 'coromandel coast',
    'kutch', 'kutch district', 'kathiawar', 'saurashtra'
]
    return any(indicator in location_lower for indicator in indian_indicators)

def main():
    all_tweets_data = []
    
    print("Starting data collection for Indian coastal hazards...")
    
    for i, query in enumerate(SEARCH_QUERIES):
        print(f"\nSearch {i+1}/{len(SEARCH_QUERIES)}: {query}")
        
        try:
            tweets = client.search_recent_tweets(
                query=query,
                max_results=5,
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                user_fields=['location'],
                expansions='author_id'
            )
            
            if tweets and tweets.data:
                print(f"Found {len(tweets.data)} tweets")
                
                # Process tweets
                users = {u.id: u for u in tweets.includes['users']} if tweets.includes and 'users' in tweets.includes else {}
                
                for tweet in tweets.data:
                    user = users.get(tweet.author_id)
                    user_location = user.location if user else ""
                    
                    all_tweets_data.append({
                        "tweet_id": tweet.id,
                        "created_at": tweet.created_at,
                        "text": tweet.text,
                        "user_location": user_location,
                        "like_count": tweet.public_metrics['like_count'],
                        "is_coastal_hazard": is_coastal_hazard(tweet.text),
                        "is_indian_location": is_indian_location(user_location)
                    })
                    
        except tweepy.TooManyRequests:
            print("Rate limit exceeded. Stopping collection.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save results
    if all_tweets_data:
        df = pd.DataFrame(all_tweets_data)
        df.to_csv("coastal_tweets_data.csv", index=False, encoding='utf-8')
        
        coastal_count = sum(1 for t in all_tweets_data if t['is_coastal_hazard'])
        indian_count = sum(1 for t in all_tweets_data if t['is_indian_location'])
        
        print(f"\nCollection complete! Saved {len(all_tweets_data)} tweets")
        print(f"Coastal hazard tweets: {coastal_count}")
        print(f"Indian location tweets: {indian_count}")
        
    else:
        print("No tweets collected.")

if __name__ == "__main__":
    main()