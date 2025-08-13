"""
Complete list of all remaining US senators to add to the simulation.
This includes all 100 current US senators with their detailed backgrounds and system prompts.
"""

# Complete list of all US senators (including the ones already added)
ALL_SENATORS = {
    # ALABAMA
    "Katie Britt": "Republican",
    "Tommy Tuberville": "Republican",
    # ALASKA
    "Lisa Murkowski": "Republican",
    "Dan Sullivan": "Republican",
    # ARIZONA
    "Kyrsten Sinema": "Independent",
    "Mark Kelly": "Democratic",
    # ARKANSAS
    "John Boozman": "Republican",
    "Tom Cotton": "Republican",
    # CALIFORNIA
    "Alex Padilla": "Democratic",
    "Laphonza Butler": "Democratic",
    # COLORADO
    "Michael Bennet": "Democratic",
    "John Hickenlooper": "Democratic",
    # CONNECTICUT
    "Richard Blumenthal": "Democratic",
    "Chris Murphy": "Democratic",
    # DELAWARE
    "Tom Carper": "Democratic",
    "Chris Coons": "Democratic",
    # FLORIDA
    "Marco Rubio": "Republican",
    "Rick Scott": "Republican",
    # GEORGIA
    "Jon Ossoff": "Democratic",
    "Raphael Warnock": "Democratic",
    # HAWAII
    "Mazie Hirono": "Democratic",
    "Brian Schatz": "Democratic",
    # IDAHO
    "Mike Crapo": "Republican",
    "Jim Risch": "Republican",
    # ILLINOIS
    "Dick Durbin": "Democratic",
    "Tammy Duckworth": "Democratic",
    # INDIANA
    "Todd Young": "Republican",
    "Mike Braun": "Republican",
    # IOWA
    "Chuck Grassley": "Republican",
    "Joni Ernst": "Republican",
    # KANSAS
    "Jerry Moran": "Republican",
    "Roger Marshall": "Republican",
    # KENTUCKY
    "Mitch McConnell": "Republican",
    "Rand Paul": "Republican",
    # LOUISIANA
    "Bill Cassidy": "Republican",
    "John Kennedy": "Republican",
    # MAINE
    "Susan Collins": "Republican",
    "Angus King": "Independent",
    # MARYLAND
    "Ben Cardin": "Democratic",
    "Chris Van Hollen": "Democratic",
    # MASSACHUSETTS
    "Elizabeth Warren": "Democratic",
    "Ed Markey": "Democratic",
    # MICHIGAN
    "Debbie Stabenow": "Democratic",
    "Gary Peters": "Democratic",
    # MINNESOTA
    "Amy Klobuchar": "Democratic",
    "Tina Smith": "Democratic",
    # MISSISSIPPI
    "Roger Wicker": "Republican",
    "Cindy Hyde-Smith": "Republican",
    # MISSOURI
    "Josh Hawley": "Republican",
    "Eric Schmitt": "Republican",
    # MONTANA
    "Jon Tester": "Democratic",
    "Steve Daines": "Republican",
    # NEBRASKA
    "Deb Fischer": "Republican",
    "Pete Ricketts": "Republican",
    # NEVADA
    "Catherine Cortez Masto": "Democratic",
    "Jacky Rosen": "Democratic",
    # NEW HAMPSHIRE
    "Jeanne Shaheen": "Democratic",
    "Maggie Hassan": "Democratic",
    # NEW JERSEY
    "Bob Menendez": "Democratic",
    "Cory Booker": "Democratic",
    # NEW MEXICO
    "Martin Heinrich": "Democratic",
    "Ben Ray Luján": "Democratic",
    # NEW YORK
    "Chuck Schumer": "Democratic",
    "Kirsten Gillibrand": "Democratic",
    # NORTH CAROLINA
    "Thom Tillis": "Republican",
    "Ted Budd": "Republican",
    # NORTH DAKOTA
    "John Hoeven": "Republican",
    "Kevin Cramer": "Republican",
    # OHIO
    "Sherrod Brown": "Democratic",
    "JD Vance": "Republican",
    # OKLAHOMA
    "James Lankford": "Republican",
    "Markwayne Mullin": "Republican",
    # OREGON
    "Ron Wyden": "Democratic",
    "Jeff Merkley": "Democratic",
    # PENNSYLVANIA
    "Bob Casey": "Democratic",
    "John Fetterman": "Democratic",
    # RHODE ISLAND
    "Jack Reed": "Democratic",
    "Sheldon Whitehouse": "Democratic",
    # SOUTH CAROLINA
    "Lindsey Graham": "Republican",
    "Tim Scott": "Republican",
    # SOUTH DAKOTA
    "John Thune": "Republican",
    "Mike Rounds": "Republican",
    # TENNESSEE
    "Marsha Blackburn": "Republican",
    "Bill Hagerty": "Republican",
    # TEXAS
    "John Cornyn": "Republican",
    "Ted Cruz": "Republican",
    # UTAH
    "Mitt Romney": "Republican",
    "Mike Lee": "Republican",
    # VERMONT
    "Bernie Sanders": "Independent",
    "Peter Welch": "Democratic",
    # VIRGINIA
    "Mark Warner": "Democratic",
    "Tim Kaine": "Democratic",
    # WASHINGTON
    "Patty Murray": "Democratic",
    "Maria Cantwell": "Democratic",
    # WEST VIRGINIA
    "Joe Manchin": "Democratic",
    "Shelley Moore Capito": "Republican",
    # WISCONSIN
    "Ron Johnson": "Republican",
    "Tammy Baldwin": "Democratic",
    # WYOMING
    "John Barrasso": "Republican",
    "Cynthia Lummis": "Republican",
}

# Senators already added to the simulation
ALREADY_ADDED = [
    "Katie Britt",
    "Tommy Tuberville",
    "Lisa Murkowski",
    "Dan Sullivan",
    "Kyrsten Sinema",
    "Mark Kelly",
    "John Boozman",
    "Tom Cotton",
    "Alex Padilla",
    "Laphonza Butler",
    "Michael Bennet",
    "John Hickenlooper",
    "Richard Blumenthal",
    "Chris Murphy",
    "Tom Carper",
    "Chris Coons",
    "Marco Rubio",
    "Rick Scott",
    "Jon Ossoff",
    "Raphael Warnock",
    "Mazie Hirono",
    "Brian Schatz",
    "Mike Crapo",
    "Jim Risch",
]

# Senators still needing to be added
REMAINING_SENATORS = {
    name: party
    for name, party in ALL_SENATORS.items()
    if name not in ALREADY_ADDED
}

print(f"Total senators: {len(ALL_SENATORS)}")
print(f"Already added: {len(ALREADY_ADDED)}")
print(f"Remaining to add: {len(REMAINING_SENATORS)}")

print("\nRemaining senators to add:")
for name, party in REMAINING_SENATORS.items():
    print(f"  {name} ({party})")
