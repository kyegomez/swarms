import uuid
import os
from swarms import Agent
from swarm_models import OpenAIChat
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Define custom system prompts for each social media platform
TWITTER_AGENT_SYS_PROMPT = """
You are a Twitter marketing expert specializing in real estate. Your task is to create engaging, concise tweets to promote properties, analyze trends to maximize engagement, and use appropriate hashtags and timing to reach potential buyers.
"""

INSTAGRAM_AGENT_SYS_PROMPT = """
You are an Instagram marketing expert focusing on real estate. Your task is to create visually appealing posts with engaging captions and hashtags to showcase properties, targeting specific demographics interested in real estate.
"""

FACEBOOK_AGENT_SYS_PROMPT = """
You are a Facebook marketing expert for real estate. Your task is to craft posts optimized for engagement and reach on Facebook, including using images, links, and targeted messaging to attract potential property buyers.
"""

LINKEDIN_AGENT_SYS_PROMPT = """
You are a LinkedIn marketing expert for the real estate industry. Your task is to create professional and informative posts, highlighting property features, market trends, and investment opportunities, tailored to professionals and investors.
"""

EMAIL_AGENT_SYS_PROMPT = """
You are an Email marketing expert specializing in real estate. Your task is to write compelling email campaigns to promote properties, focusing on personalization, subject lines, and effective call-to-action strategies to drive conversions.
"""

# Example usage:
api_key = os.getenv("OPENAI_API_KEY")

# Model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize your agents for different social media platforms
agents = [
    Agent(
        agent_name="Twitter-RealEstate-Agent",
        system_prompt=TWITTER_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="twitter_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Instagram-RealEstate-Agent",
        system_prompt=INSTAGRAM_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="instagram_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Facebook-RealEstate-Agent",
        system_prompt=FACEBOOK_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="facebook_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="LinkedIn-RealEstate-Agent",
        system_prompt=LINKEDIN_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="linkedin_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
    Agent(
        agent_name="Email-RealEstate-Agent",
        system_prompt=EMAIL_AGENT_SYS_PROMPT,
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        saved_state_path="email_realestate_agent.json",
        user_name="realestate_swarms",
        retry_attempts=1,
    ),
]

# Create a Swarm with the list of agents
swarm = SpreadSheetSwarm(
    name="Real-Estate-Marketing-Swarm",
    description="A swarm that processes real estate marketing tasks using multiple agents on different threads.",
    agents=agents,
    autosave_on=True,
    save_file_path=f"{uuid.uuid4().hex}_real_estate_swarm.csv",
    run_all_agents=False,
    max_loops=2,
    workspace_dir="real-estate-swarm",
)


# Run the swarm
swarm.run(
    task="""
    Create posts to promote luxury properties in North Texas, highlighting their features, location, and investment potential. Include relevant hashtags, images, and engaging captions.

    
    Property:
    $10,399,000
    1609 Meandering Way Dr, Roanoke, TX 76262
    Link to the property: https://www.zillow.com/homedetails/1609-Meandering-Way-Dr-Roanoke-TX-76262/308879785_zpid/
    
    What's special
    Unveiling a new custom estate in the prestigious gated Quail Hollow Estates! This impeccable residence, set on a sprawling acre surrounded by majestic trees, features a gourmet kitchen equipped with top-tier Subzero and Wolf appliances. European soft-close cabinets and drawers, paired with a double Cambria Quartzite island, perfect for family gatherings. The first-floor game room&media room add extra layers of entertainment. Step into the outdoor sanctuary, where a sparkling pool and spa, and sunken fire pit, beckon leisure. The lavish master suite features stunning marble accents, custom his&her closets, and a secure storm shelter.Throughout the home,indulge in the visual charm of designer lighting and wallpaper, elevating every space. The property is complete with a 6-car garage and a sports court, catering to the preferences of basketball or pickleball enthusiasts. This residence seamlessly combines luxury&recreational amenities, making it a must-see for the discerning buyer.
    
    Facts & features
    Interior
    Bedrooms & bathrooms
    Bedrooms: 6
    Bathrooms: 8
    Full bathrooms: 7
    1/2 bathrooms: 1
    Primary bedroom
    Bedroom
    Features: Built-in Features, En Suite Bathroom, Walk-In Closet(s)
    Cooling
    Central Air, Ceiling Fan(s), Electric
    Appliances
    Included: Built-In Gas Range, Built-In Refrigerator, Double Oven, Dishwasher, Gas Cooktop, Disposal, Ice Maker, Microwave, Range, Refrigerator, Some Commercial Grade, Vented Exhaust Fan, Warming Drawer, Wine Cooler
    Features
    Wet Bar, Built-in Features, Dry Bar, Decorative/Designer Lighting Fixtures, Eat-in Kitchen, Elevator, High Speed Internet, Kitchen Island, Pantry, Smart Home, Cable TV, Walk-In Closet(s), Wired for Sound
    Flooring: Hardwood
    Has basement: No
    Number of fireplaces: 3
    Fireplace features: Living Room, Primary Bedroom
    Interior area
    Total interior livable area: 10,466 sqft
    Total spaces: 12
    Parking features: Additional Parking
    Attached garage spaces: 6
    Carport spaces: 6
    Features
    Levels: Two
    Stories: 2
    Patio & porch: Covered
    Exterior features: Built-in Barbecue, Barbecue, Gas Grill, Lighting, Outdoor Grill, Outdoor Living Area, Private Yard, Sport Court, Fire Pit
    Pool features: Heated, In Ground, Pool, Pool/Spa Combo
    Fencing: Wrought Iron
    Lot
    Size: 1.05 Acres
    Details
    Additional structures: Outdoor Kitchen
    Parcel number: 42232692
    Special conditions: Standard
    Construction
    Type & style
    Home type: SingleFamily
    Architectural style: Contemporary/Modern,Detached
    Property subtype: Single Family Residence
    """
)

print(swarm.export_to_json())
