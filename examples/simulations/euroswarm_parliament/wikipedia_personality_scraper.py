#!/usr/bin/env python3
"""
Wikipedia Personality Scraper for EuroSwarm Parliament MEPs

This module scrapes Wikipedia data for each MEP to create realistic, personality-driven
AI agents based on their real backgrounds, political history, and personal beliefs.
"""

import json
import os
import time
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from loguru import logger
import xml.etree.ElementTree as ET


@dataclass
class MEPPersonalityProfile:
    """
    Comprehensive personality profile for an MEP based on Wikipedia data.
    
    Attributes:
        full_name: Full name of the MEP
        mep_id: Unique MEP identifier
        wikipedia_url: URL of the MEP's Wikipedia page
        summary: Brief summary of the MEP's background
        early_life: Early life and education information
        political_career: Political career and positions held
        political_views: Key political views and positions
        policy_focus: Areas of policy expertise and focus
        achievements: Notable achievements and accomplishments
        controversies: Any controversies or notable incidents
        personal_life: Personal background and family information
        education: Educational background
        professional_background: Professional experience before politics
        party_affiliations: Political party history
        committee_experience: Parliamentary committee experience
        voting_record: Notable voting patterns or positions
        public_statements: Key public statements or quotes
        interests: Personal and professional interests
        languages: Languages spoken
        awards: Awards and recognitions
        publications: Publications or written works
        social_media: Social media presence
        last_updated: When the profile was last updated
    """
    
    full_name: str
    mep_id: str
    wikipedia_url: Optional[str] = None
    summary: str = ""
    early_life: str = ""
    political_career: str = ""
    political_views: str = ""
    policy_focus: str = ""
    achievements: str = ""
    controversies: str = ""
    personal_life: str = ""
    education: str = ""
    professional_background: str = ""
    party_affiliations: str = ""
    committee_experience: str = ""
    voting_record: str = ""
    public_statements: str = ""
    interests: str = ""
    languages: str = ""
    awards: str = ""
    publications: str = ""
    social_media: str = ""
    last_updated: str = ""


class WikipediaPersonalityScraper:
    """
    Scraper for gathering Wikipedia personality data for MEPs.
    """
    
    def __init__(self, output_dir: str = "mep_personalities", verbose: bool = True):
        """
        Initialize the Wikipedia personality scraper.
        
        Args:
            output_dir: Directory to store personality profiles
            verbose: Enable verbose logging
        """
        self.output_dir = output_dir
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EuroSwarm Parliament Personality Scraper/1.0 (https://github.com/swarms-democracy)'
        })
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            logger.info(f"Wikipedia Personality Scraper initialized. Output directory: {output_dir}")

    def extract_mep_data_from_xml(self, xml_file: str = "EU.xml") -> List[Dict[str, str]]:
        """
        Extract MEP data from EU.xml file.
        
        Args:
            xml_file: Path to EU.xml file
            
        Returns:
            List of MEP data dictionaries
        """
        meps = []
        
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use regex to extract MEP data
            mep_pattern = r'<mep>\s*<fullName>(.*?)</fullName>\s*<country>(.*?)</country>\s*<politicalGroup>(.*?)</politicalGroup>\s*<id>(.*?)</id>\s*<nationalPoliticalGroup>(.*?)</nationalPoliticalGroup>\s*</mep>'
            mep_matches = re.findall(mep_pattern, content, re.DOTALL)
            
            for full_name, country, political_group, mep_id, national_party in mep_matches:
                meps.append({
                    'full_name': full_name.strip(),
                    'country': country.strip(),
                    'political_group': political_group.strip(),
                    'mep_id': mep_id.strip(),
                    'national_party': national_party.strip()
                })
            
            if self.verbose:
                logger.info(f"Extracted {len(meps)} MEPs from {xml_file}")
                
        except Exception as e:
            logger.error(f"Error extracting MEP data from {xml_file}: {e}")
            
        return meps

    def search_wikipedia_page(self, mep_name: str, country: str) -> Optional[str]:
        """
        Search for a Wikipedia page for an MEP.
        
        Args:
            mep_name: Full name of the MEP
            country: Country of the MEP
            
        Returns:
            Wikipedia page title if found, None otherwise
        """
        try:
            # Search for the MEP on Wikipedia
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'"{mep_name}" {country}',
                'srlimit': 5,
                'srnamespace': 0
            }
            
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            
            if search_results:
                # Return the first result
                return search_results[0]['title']
            
            # Try alternative search without quotes
            search_params['srsearch'] = f'{mep_name} {country}'
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            
            if search_results:
                return search_results[0]['title']
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error searching Wikipedia for {mep_name}: {e}")
        
        return None

    def get_wikipedia_content(self, page_title: str) -> Optional[Dict[str, Any]]:
        """
        Get Wikipedia content for a specific page.
        
        Args:
            page_title: Wikipedia page title
            
        Returns:
            Dictionary containing page content and metadata
        """
        try:
            # Get page content
            content_url = "https://en.wikipedia.org/w/api.php"
            content_params = {
                'action': 'query',
                'format': 'json',
                'titles': page_title,
                'prop': 'extracts|info|categories',
                'exintro': True,
                'explaintext': True,
                'inprop': 'url',
                'cllimit': 50
            }
            
            response = self.session.get(content_url, params=content_params)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            if pages:
                page_id = list(pages.keys())[0]
                page_data = pages[page_id]
                
                return {
                    'title': page_data.get('title', ''),
                    'extract': page_data.get('extract', ''),
                    'url': page_data.get('fullurl', ''),
                    'categories': [cat['title'] for cat in page_data.get('categories', [])],
                    'pageid': page_data.get('pageid', ''),
                    'length': page_data.get('length', 0)
                }
                
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error getting Wikipedia content for {page_title}: {e}")
        
        return None

    def parse_wikipedia_content(self, content: str, mep_name: str) -> Dict[str, str]:
        """
        Parse Wikipedia content to extract structured personality information.
        
        Args:
            content: Raw Wikipedia content
            mep_name: Name of the MEP
            
        Returns:
            Dictionary of parsed personality information
        """
        personality_data = {
            'summary': '',
            'early_life': '',
            'political_career': '',
            'political_views': '',
            'policy_focus': '',
            'achievements': '',
            'controversies': '',
            'personal_life': '',
            'education': '',
            'professional_background': '',
            'party_affiliations': '',
            'committee_experience': '',
            'voting_record': '',
            'public_statements': '',
            'interests': '',
            'languages': '',
            'awards': '',
            'publications': '',
            'social_media': ''
        }
        
        # Extract summary (first paragraph)
        paragraphs = content.split('\n\n')
        if paragraphs:
            personality_data['summary'] = paragraphs[0][:1000]  # Limit summary length
        
        # Look for specific sections
        content_lower = content.lower()
        
        # Early life and education
        early_life_patterns = [
            r'early life[^.]*\.',
            r'born[^.]*\.',
            r'childhood[^.]*\.',
            r'grew up[^.]*\.',
            r'education[^.]*\.'
        ]
        
        for pattern in early_life_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                personality_data['early_life'] = ' '.join(matches[:3])  # Take first 3 matches
                break
        
        # Political career
        political_patterns = [
            r'political career[^.]*\.',
            r'elected[^.]*\.',
            r'parliament[^.]*\.',
            r'minister[^.]*\.',
            r'party[^.]*\.'
        ]
        
        for pattern in political_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                personality_data['political_career'] = ' '.join(matches[:5])  # Take first 5 matches
                break
        
        # Political views
        views_patterns = [
            r'political views[^.]*\.',
            r'positions[^.]*\.',
            r'advocates[^.]*\.',
            r'supports[^.]*\.',
            r'opposes[^.]*\.'
        ]
        
        for pattern in views_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                personality_data['political_views'] = ' '.join(matches[:3])
                break
        
        # Policy focus
        policy_patterns = [
            r'policy[^.]*\.',
            r'focus[^.]*\.',
            r'issues[^.]*\.',
            r'legislation[^.]*\.'
        ]
        
        for pattern in policy_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                personality_data['policy_focus'] = ' '.join(matches[:3])
                break
        
        # Achievements
        achievement_patterns = [
            r'achievements[^.]*\.',
            r'accomplishments[^.]*\.',
            r'success[^.]*\.',
            r'won[^.]*\.',
            r'received[^.]*\.'
        ]
        
        for pattern in achievement_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                personality_data['achievements'] = ' '.join(matches[:3])
                break
        
        return personality_data

    def create_personality_profile(self, mep_data: Dict[str, str]) -> MEPPersonalityProfile:
        """
        Create a personality profile for an MEP.
        
        Args:
            mep_data: MEP data from XML file
            
        Returns:
            MEPPersonalityProfile object
        """
        mep_name = mep_data['full_name']
        country = mep_data['country']
        
        # Search for Wikipedia page
        page_title = self.search_wikipedia_page(mep_name, country)
        
        if page_title:
            # Get Wikipedia content
            wiki_content = self.get_wikipedia_content(page_title)
            
            if wiki_content:
                # Parse content
                personality_data = self.parse_wikipedia_content(wiki_content['extract'], mep_name)
                
                # Create profile
                profile = MEPPersonalityProfile(
                    full_name=mep_name,
                    mep_id=mep_data['mep_id'],
                    wikipedia_url=wiki_content['url'],
                    summary=personality_data['summary'],
                    early_life=personality_data['early_life'],
                    political_career=personality_data['political_career'],
                    political_views=personality_data['political_views'],
                    policy_focus=personality_data['policy_focus'],
                    achievements=personality_data['achievements'],
                    controversies=personality_data['controversies'],
                    personal_life=personality_data['personal_life'],
                    education=personality_data['education'],
                    professional_background=personality_data['professional_background'],
                    party_affiliations=personality_data['party_affiliations'],
                    committee_experience=personality_data['committee_experience'],
                    voting_record=personality_data['voting_record'],
                    public_statements=personality_data['public_statements'],
                    interests=personality_data['interests'],
                    languages=personality_data['languages'],
                    awards=personality_data['awards'],
                    publications=personality_data['publications'],
                    social_media=personality_data['social_media'],
                    last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                if self.verbose:
                    logger.info(f"Created personality profile for {mep_name} from Wikipedia")
                
                return profile
        
        # Create minimal profile if no Wikipedia data found
        profile = MEPPersonalityProfile(
            full_name=mep_name,
            mep_id=mep_data['mep_id'],
            summary=f"{mep_name} is a Member of the European Parliament representing {country}.",
            political_career=f"Currently serving as MEP for {country}.",
            political_views=f"Member of {mep_data['political_group']} and {mep_data['national_party']}.",
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        if self.verbose:
            logger.warning(f"No Wikipedia data found for {mep_name}, created minimal profile")
        
        return profile

    def save_personality_profile(self, profile: MEPPersonalityProfile) -> str:
        """
        Save personality profile to JSON file.
        
        Args:
            profile: MEPPersonalityProfile object
            
        Returns:
            Path to saved file
        """
        # Create safe filename
        safe_name = re.sub(r'[^\w\s-]', '', profile.full_name).strip()
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        filename = f"{safe_name}_{profile.mep_id}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to dictionary and save
        profile_dict = asdict(profile)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            logger.info(f"Saved personality profile: {filepath}")
        
        return filepath

    def scrape_all_mep_personalities(self, xml_file: str = "EU.xml", delay: float = 1.0) -> Dict[str, str]:
        """
        Scrape personality data for all MEPs.
        
        Args:
            xml_file: Path to EU.xml file
            delay: Delay between requests to be respectful to Wikipedia
            
        Returns:
            Dictionary mapping MEP names to their personality profile file paths
        """
        meps = self.extract_mep_data_from_xml(xml_file)
        profile_files = {}
        
        if self.verbose:
            logger.info(f"Starting personality scraping for {len(meps)} MEPs")
        
        for i, mep_data in enumerate(meps, 1):
            mep_name = mep_data['full_name']
            
            if self.verbose:
                logger.info(f"Processing {i}/{len(meps)}: {mep_name}")
            
            try:
                # Create personality profile
                profile = self.create_personality_profile(mep_data)
                
                # Save profile
                filepath = self.save_personality_profile(profile)
                profile_files[mep_name] = filepath
                
                # Respectful delay
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error processing {mep_name}: {e}")
                continue
        
        if self.verbose:
            logger.info(f"Completed personality scraping. {len(profile_files)} profiles created.")
        
        return profile_files

    def load_personality_profile(self, filepath: str) -> MEPPersonalityProfile:
        """
        Load personality profile from JSON file.
        
        Args:
            filepath: Path to personality profile JSON file
            
        Returns:
            MEPPersonalityProfile object
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return MEPPersonalityProfile(**data)

    def get_personality_summary(self, profile: MEPPersonalityProfile) -> str:
        """
        Generate a personality summary for use in AI agent system prompts.
        
        Args:
            profile: MEPPersonalityProfile object
            
        Returns:
            Formatted personality summary
        """
        summary_parts = []
        
        if profile.summary:
            summary_parts.append(f"Background: {profile.summary}")
        
        if profile.political_career:
            summary_parts.append(f"Political Career: {profile.political_career}")
        
        if profile.political_views:
            summary_parts.append(f"Political Views: {profile.political_views}")
        
        if profile.policy_focus:
            summary_parts.append(f"Policy Focus: {profile.policy_focus}")
        
        if profile.achievements:
            summary_parts.append(f"Notable Achievements: {profile.achievements}")
        
        if profile.education:
            summary_parts.append(f"Education: {profile.education}")
        
        if profile.professional_background:
            summary_parts.append(f"Professional Background: {profile.professional_background}")
        
        return "\n".join(summary_parts)


def main():
    """Main function to run the Wikipedia personality scraper."""
    
    print("🏛️  WIKIPEDIA PERSONALITY SCRAPER FOR EUROSWARM PARLIAMENT")
    print("=" * 70)
    
    # Initialize scraper
    scraper = WikipediaPersonalityScraper(output_dir="mep_personalities", verbose=True)
    
    # Scrape all MEP personalities
    profile_files = scraper.scrape_all_mep_personalities(delay=1.0)
    
    print(f"\n✅ Scraping completed!")
    print(f"📁 Profiles saved to: {scraper.output_dir}")
    print(f"📊 Total profiles created: {len(profile_files)}")
    
    # Show sample profile
    if profile_files:
        sample_name = list(profile_files.keys())[0]
        sample_file = profile_files[sample_name]
        sample_profile = scraper.load_personality_profile(sample_file)
        
        print(f"\n📋 Sample Profile: {sample_name}")
        print("-" * 50)
        print(scraper.get_personality_summary(sample_profile))


if __name__ == "__main__":
    main() 