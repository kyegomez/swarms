"""
Professional Persona Generator

A system that generates detailed professional persona prompts for each profession
in a CSV dataset using AI agents. Creates comprehensive personas suitable for
use as AI agent prompts.

FIXED ISSUES:
- Changed agent output_type from "str-all-except-first" to "str" to prevent context accumulation
- Modified both sequential and concurrent processing to create fresh agent instances per profession
- Added clear_progress() method to restart processing from scratch
- Disabled streaming and memory retention to ensure clean, independent generations

Author: Swarms Team
"""

import uuid
import pandas as pd
import csv
import json
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple
import time
import os
from pathlib import Path

from swarms import Agent
from loguru import logger


class ProfessionPersonaGenerator:
    """
    A comprehensive system for generating detailed professional persona prompts.

    This class processes profession data from CSV files and uses an AI agent to
    generate detailed, world-class professional personas that can be used as
    prompts for AI agents.

    Attributes:
        input_file (Path): Path to the input CSV file containing profession data
        output_file (Path): Path where the output CSV will be saved
        json_progress_file (Path): Path where the JSON progress file is saved
        agent (Optional[Agent]): The AI agent used for generating personas
        processed_count (int): Number of professions processed so far
        current_data (List[Dict]): Current progress data for autosaving
        progress_lock (threading.Lock): Thread lock for safe progress updates
        max_workers (int): Maximum number of concurrent workers

    Example:
        >>> generator = ProfessionPersonaGenerator("data.csv", "personas.csv")
        >>> generator.process_all_professions(concurrent=True)
    """

    def __init__(
        self,
        input_file: str = "data.csv",
        output_file: str = "profession_personas.csv",
    ) -> None:
        """
        Initialize the Professional Persona Generator.

        Args:
            input_file: Path to the input CSV file containing profession data
            output_file: Path where the generated personas CSV will be saved

        Raises:
            FileNotFoundError: If the input file doesn't exist
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.json_progress_file = Path(output_file).with_suffix(
            ".progress.json"
        )
        self.agent: Optional[Agent] = None
        self.processed_count: int = 0
        self.current_data: List[Dict[str, str]] = []
        self.progress_lock = (
            threading.Lock()
        )  # Thread safety for progress updates

        # Calculate optimal worker count (90% of CPU cores, minimum 1, maximum 8)
        cpu_count = os.cpu_count() or 1
        self.max_workers = max(1, min(8, int(cpu_count * 0.9)))

        # Configure logging
        self._setup_logging()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Validate input file exists
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            raise FileNotFoundError(
                f"Input file not found: {self.input_file}"
            )

        # Load existing progress if available
        self._load_existing_progress()

        logger.info("Initialized ProfessionPersonaGenerator")
        logger.info(f"Input: {self.input_file}")
        logger.info(f"Output: {self.output_file}")
        logger.info(f"JSON Progress: {self.json_progress_file}")
        logger.info(
            f"Max Workers: {self.max_workers} (90% of {cpu_count} CPU cores)"
        )
        if self.processed_count > 0:
            logger.info(
                f"Resuming from {self.processed_count} previously processed professions"
            )

    def _setup_logging(self) -> None:
        return logger

    def _create_persona_agent(self) -> Agent:
        """
        Create and configure the AI agent for generating professional personas.

        Returns:
            Agent: Configured agent specialized in generating professional personas

        Raises:
            Exception: If agent creation fails
        """
        logger.info("Creating persona generator agent...")

        system_prompt = """
        You are an expert professional persona generator with deep expertise in:
        - Career development and professional psychology
        - Industry-specific knowledge across all sectors
        - Professional skill development and competencies
        - Workplace dynamics and best practices
        - Leadership and expertise development
        - Professional communication and mindset
        
        Your task is to create comprehensive persona prompts for professionals that include:
        
        1. **UNIQUE PROFESSIONAL NAME**: Create a realistic, memorable name that fits the profession
        
        2. **EXPERIENCE HISTORY**: Design a compelling 15-20 year career trajectory with:
           - Educational background (specific degrees, certifications, training)
           - Career progression with specific roles and companies
           - Key achievements and milestones
           - Notable projects or accomplishments
           - Professional development activities
        
        3. **CORE INSTRUCTIONS**: Define the professional's:
           - Primary responsibilities and duties
           - Key performance indicators and success metrics
           - Professional standards and ethics
           - Stakeholder relationships and communication protocols
           - Decision-making frameworks
        
        4. **COMMON WORKFLOWS**: Outline typical:
           - Daily/weekly/monthly routines and processes
           - Project management approaches
           - Problem-solving methodologies
           - Collaboration and team interaction patterns
           - Tools, software, and systems used
        
        5. **MENTAL MODELS**: Describe the cognitive frameworks for:
           - Strategic thinking patterns
           - Risk assessment and management
           - Innovation and continuous improvement
           - Professional judgment and expertise application
           - Industry-specific analytical approaches
           - Best practice implementation
        
        6. **WORLD-CLASS EXCELLENCE**: Define what makes them the best:
           - Unique expertise and specializations
           - Industry recognition and thought leadership
           - Innovative approaches and methodologies
           - Mentorship and knowledge sharing
           - Continuous learning and adaptation
        
        Create a comprehensive, realistic persona that could serve as a detailed prompt for an AI agent 
        to embody this professional role at the highest level of expertise and performance.
        
        Format your response as a complete, ready-to-use agent prompt that starts with:
        "You are [Name], a world-class [profession]..."
        
        Make it detailed, specific, and actionable while maintaining professional authenticity."""

        try:
            agent = Agent(
                agent_name="Professional-Persona-Generator",
                agent_description="Expert agent for creating detailed professional persona prompts",
                system_prompt=system_prompt,
                max_loops=1,
                model_name="gpt-4.1",
                dynamic_temperature_enabled=True,
                output_type="final",  # Changed from "str-all-except-first" to prevent context accumulation
                streaming_on=False,  # Disabled streaming for cleaner output
                saved_state_path=None,  # Ensure no state persistence
                long_term_memory=None,  # Ensure no memory retention
            )

            logger.success(
                "Persona generator agent created successfully"
            )
            return agent

        except Exception as e:
            logger.error(f"Failed to create persona agent: {str(e)}")
            raise

    def _generate_persona_prompt(
        self, profession_title: str, profession_description: str
    ) -> str:
        """
        Generate a detailed persona prompt for a specific profession.

        Args:
            profession_title: The title/name of the profession
            profession_description: Detailed description of the profession

        Returns:
            str: Generated persona prompt

        Raises:
            Exception: If persona generation fails
        """
        if not self.agent:
            logger.error("Agent not initialized")
            raise RuntimeError(
                "Agent not initialized. Call _create_persona_agent() first."
            )

        prompt = f"""
        Create a comprehensive professional persona prompt for the following profession:
        
        **Profession Title**: {profession_title}
        **Profession Description**: {profession_description}
        
        Generate a complete persona that includes:
        1. A unique professional name
        2. Detailed experience history (15-20 years)
        3. Core instructions and responsibilities
        4. Common workflows and processes
        5. Mental models for world-class thinking
        6. Excellence characteristics that make them the best in the world
        
        Make this persona realistic, detailed, and suitable for use as an AI agent prompt.
        The persona should embody the highest level of expertise and professionalism in this field.
        """

        try:
            logger.debug(
                f"Generating persona for: {profession_title}"
            )
            start_time = time.time()

            response = self.agent.run(prompt)

            end_time = time.time()
            generation_time = end_time - start_time

            logger.debug(
                f"Generated persona in {generation_time:.2f}s, length: {len(response)} chars"
            )
            return response

        except Exception as e:
            error_msg = f"Error generating persona for {profession_title}: {str(e)}"
            logger.error(error_msg)
            return f"Error generating persona: {str(e)}"

    def _load_profession_data(self) -> pd.DataFrame:
        """
        Load profession data from the input CSV file.

        Returns:
            pd.DataFrame: DataFrame containing profession data

        Raises:
            Exception: If CSV loading fails
        """
        try:
            df = pd.read_csv(self.input_file)
            logger.info(
                f"Loaded {len(df)} professions from {self.input_file}"
            )
            return df

        except Exception as e:
            logger.error(f"Error reading {self.input_file}: {str(e)}")
            raise

    def _save_results(self, data: List[Dict[str, str]]) -> None:
        """
        Save the generated personas to a CSV file.

        Args:
            data: List of dictionaries containing profession titles and personas

        Raises:
            Exception: If saving fails
        """
        try:
            df = pd.DataFrame(data)

            # Save with proper CSV formatting for long text fields
            df.to_csv(
                self.output_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )

            file_size_mb = (
                os.path.getsize(self.output_file) / 1024 / 1024
            )

            logger.success(f"Results saved to {self.output_file}")
            logger.info(f"Total professions processed: {len(data)}")
            logger.info(f"File size: {file_size_mb:.2f} MB")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def _save_progress(self, data: List[Dict[str, str]]) -> None:
        """
        Save progress to avoid losing work during long processing runs.

        Args:
            data: Current progress data to save
        """
        try:
            progress_file = self.output_file.with_suffix(
                ".progress.csv"
            )
            df = pd.DataFrame(data)
            df.to_csv(
                progress_file,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )
            logger.debug(f"Progress saved to {progress_file}")

        except Exception as e:
            logger.warning(f"Failed to save progress: {str(e)}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown on keyboard interrupt."""

        def signal_handler(signum, frame):
            logger.warning(
                "🚨 Keyboard interrupt received! Saving progress..."
            )
            self._save_progress_json(
                self.current_data, force_save=True
            )
            logger.success("✅ Progress saved. Exiting gracefully.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.debug(
            "Signal handlers configured for graceful shutdown"
        )

    def _load_existing_progress(self) -> None:
        """Load existing progress from JSON file if it exists."""
        if self.json_progress_file.exists():
            try:
                with open(
                    self.json_progress_file, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    self.current_data = data.get("professions", [])
                    self.processed_count = len(self.current_data)

                logger.info(
                    f"📂 Loaded existing progress: {self.processed_count} professions"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to load existing progress: {str(e)}"
                )
                self.current_data = []
                self.processed_count = 0

    def _save_progress_json(
        self, data: List[Dict[str, str]], force_save: bool = False
    ) -> None:
        """
        Save progress to JSON file with metadata (thread-safe).

        Args:
            data: Current progress data to save
            force_save: Force save even if it's not a checkpoint interval
        """
        with self.progress_lock:  # Ensure thread safety
            try:
                progress_data = {
                    "metadata": {
                        "total_processed": len(data),
                        "last_updated": time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "input_file": str(self.input_file),
                        "output_file": str(self.output_file),
                        "processing_status": "in_progress",
                        "max_workers": self.max_workers,
                    },
                    "professions": data,
                }

                # Create backup of existing file before overwriting
                if self.json_progress_file.exists():
                    backup_file = self.json_progress_file.with_suffix(
                        ".backup.json"
                    )
                    try:
                        with open(
                            self.json_progress_file,
                            "r",
                            encoding="utf-8",
                        ) as src:
                            with open(
                                backup_file, "w", encoding="utf-8"
                            ) as dst:
                                dst.write(src.read())
                    except Exception:
                        pass  # Backup failed, but continue with main save

                # Save current progress
                with open(
                    self.json_progress_file, "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        progress_data, f, indent=2, ensure_ascii=False
                    )

                file_size_mb = (
                    os.path.getsize(self.json_progress_file)
                    / 1024
                    / 1024
                )

                if force_save:
                    logger.success(
                        f"🚨 Emergency progress saved to {self.json_progress_file} ({file_size_mb:.2f} MB)"
                    )
                else:
                    logger.debug(
                        f"💾 Progress saved to {self.json_progress_file} ({file_size_mb:.2f} MB)"
                    )

            except Exception as e:
                logger.error(
                    f"❌ Critical error saving progress: {str(e)}"
                )
                # Try to save to a fallback location
                try:
                    fallback_file = Path(
                        f"emergency_backup_{int(time.time())}.json"
                    )
                    with open(
                        fallback_file, "w", encoding="utf-8"
                    ) as f:
                        json.dump(
                            {"professions": data},
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.warning(
                        f"📁 Emergency backup saved to {fallback_file}"
                    )
                except Exception as fallback_error:
                    logger.critical(
                        f"💥 Failed to save emergency backup: {str(fallback_error)}"
                    )

    def _process_single_profession(
        self, profession_data: Tuple[int, str, str]
    ) -> Dict[str, str]:
        """
        Process a single profession and generate its persona (thread-safe).

        Args:
            profession_data: Tuple of (index, profession_title, profession_description)

        Returns:
            Dict containing profession name and persona prompt

        Raises:
            Exception: If persona generation fails
        """
        index, profession_title, profession_description = (
            profession_data
        )

        try:
            logger.debug(f"🔄 Worker processing: {profession_title}")

            # Create a separate agent instance for this thread to avoid conflicts
            thread_agent = self._create_persona_agent()

            # Generate persona prompt
            persona_prompt = self._generate_persona_prompt_with_agent(
                thread_agent, profession_title, profession_description
            )

            result = {
                "profession_name": profession_title,
                "persona_prompt": persona_prompt,
            }

            logger.debug(f"✅ Completed: {profession_title}")
            return result

        except Exception as e:
            error_msg = (
                f"❌ Error processing '{profession_title}': {str(e)}"
            )
            logger.error(error_msg)

            # Return error entry
            return {
                "profession_name": profession_title,
                "persona_prompt": f"ERROR: Failed to generate persona - {str(e)}",
            }

    def _generate_persona_prompt_with_agent(
        self,
        agent: Agent,
        profession_title: str,
        profession_description: str,
    ) -> str:
        """
        Generate a detailed persona prompt using a specific agent instance.

        Args:
            agent: Agent instance to use for generation
            profession_title: The title/name of the profession
            profession_description: Detailed description of the profession

        Returns:
            str: Generated persona prompt
        """
        prompt = f"""
        Create a comprehensive professional persona prompt for the following profession:
        
        **Profession Title**: {profession_title}
        **Profession Description**: {profession_description}
        
        Generate a complete persona that includes:
        1. A unique professional name
        2. Detailed experience history (15-20 years)
        3. Core instructions and responsibilities
        4. Common workflows and processes
        5. Mental models for world-class thinking
        6. Excellence characteristics that make them the best in the world
        
        Make this persona realistic, detailed, and suitable for use as an AI agent prompt.
        The persona should embody the highest level of expertise and professionalism in this field.
        """

        try:
            start_time = time.time()
            response = agent.run(prompt)
            end_time = time.time()
            generation_time = end_time - start_time

            logger.debug(
                f"Generated persona in {generation_time:.2f}s, length: {len(response)} chars"
            )
            return response

        except Exception as e:
            error_msg = f"Error generating persona for {profession_title}: {str(e)}"
            logger.error(error_msg)
            return f"Error generating persona: {str(e)}"

    def _update_progress_safely(
        self, new_entry: Dict[str, str]
    ) -> None:
        """
        Thread-safe method to update progress data.

        Args:
            new_entry: New profession entry to add
        """
        with self.progress_lock:
            self.current_data.append(new_entry)
            # Save progress after every update for maximum safety
            self._save_progress_json(self.current_data)

    def _mark_processing_complete(self) -> None:
        """Mark the processing as complete in the JSON progress file."""
        try:
            if self.json_progress_file.exists():
                with open(
                    self.json_progress_file, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)

                data["metadata"]["processing_status"] = "completed"
                data["metadata"]["completion_time"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                with open(
                    self.json_progress_file, "w", encoding="utf-8"
                ) as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                logger.success(
                    "✅ Processing marked as complete in progress file"
                )

        except Exception as e:
            logger.warning(
                f"Failed to mark processing complete: {str(e)}"
            )

    def process_professions(
        self,
        start_row: Optional[int] = None,
        end_row: Optional[int] = None,
        max_rows: Optional[int] = None,
        concurrent: bool = False,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Process a range of professions and generate persona prompts.

        Args:
            start_row: Starting row index (0-based), None for beginning
            end_row: Ending row index (exclusive), None for end
            max_rows: Maximum number of rows to process, None for no limit
            concurrent: Whether to process professions concurrently
            max_workers: Maximum number of concurrent workers, None for auto-calculated

        Returns:
            List[Dict[str, str]]: Generated profession personas

        Raises:
            Exception: If processing fails
        """
        # Override max_workers if specified
        if max_workers is not None:
            self.max_workers = max(
                1, min(max_workers, 16)
            )  # Cap at 16 for API safety
            logger.info(
                f"Using custom max_workers: {self.max_workers}"
            )

        mode_str = "concurrent" if concurrent else "sequential"
        worker_info = (
            f" (max workers: {self.max_workers})"
            if concurrent
            else ""
        )
        logger.info(
            f"🚀 Starting Professional Persona Generation - {mode_str} mode{worker_info}"
        )

        try:
            # Load data (agent will be created per profession to avoid context retention)
            df = self._load_profession_data()

            # Apply max_rows limit first if specified
            if max_rows is not None:
                original_length = len(df)
                df = df.head(max_rows)
                logger.info(
                    f"Limited to first {max_rows} rows (from {original_length} total)"
                )

            # Determine processing range
            if start_row is not None and end_row is not None:
                df = df.iloc[start_row:end_row]
                logger.info(
                    f"Processing rows {start_row} to {end_row} ({len(df)} professions)"
                )
            else:
                logger.info(f"Processing {len(df)} professions")

            # Skip already processed professions if resuming
            professions_to_process = []
            processed_titles = set(
                item["profession_name"] for item in self.current_data
            )

            for index, row in df.iterrows():
                profession_title = row["O*NET-SOC 2019 Title"]
                profession_description = row[
                    "O*NET-SOC 2019 Description"
                ]
                if profession_title not in processed_titles:
                    professions_to_process.append(
                        (
                            index,
                            profession_title,
                            profession_description,
                        )
                    )

            if len(professions_to_process) < len(df):
                skipped_count = len(df) - len(professions_to_process)
                logger.info(
                    f"📋 Skipping {skipped_count} already processed professions"
                )
                logger.info(
                    f"📋 Processing {len(professions_to_process)} remaining professions"
                )

            if not professions_to_process:
                logger.success(
                    "✅ All professions already processed!"
                )
                return self.current_data

            # Process professions based on mode
            if concurrent:
                self._process_concurrent(professions_to_process)
            else:
                self._process_sequential(professions_to_process)

            # Update final count
            self.processed_count = len(self.current_data)

            logger.success(
                f"✅ Completed processing {len(professions_to_process)} new professions"
            )
            logger.success(
                f"✅ Total professions in dataset: {len(self.current_data)}"
            )

            return self.current_data

        except Exception as e:
            logger.error(
                f"❌ Critical error in process_professions: {str(e)}"
            )
            # Emergency save
            self._save_progress_json(
                self.current_data, force_save=True
            )
            raise

    def _process_sequential(
        self, professions_to_process: List[Tuple[int, str, str]]
    ) -> None:
        """Process professions sequentially (original behavior)."""
        logger.info("🔄 Processing sequentially...")

        for prof_index, (
            index,
            profession_title,
            profession_description,
        ) in enumerate(professions_to_process):
            current_progress = len(self.current_data) + prof_index + 1
            total_to_process = len(professions_to_process)

            logger.info(
                f"📋 Processing {current_progress}/{len(professions_to_process) + len(self.current_data)}: {profession_title}"
            )

            try:
                # Create a fresh agent for this profession to avoid context retention
                fresh_agent = self._create_persona_agent()

                # Generate persona prompt with the fresh agent
                persona_prompt = (
                    self._generate_persona_prompt_with_agent(
                        fresh_agent,
                        profession_title,
                        profession_description,
                    )
                )

                # Add to current data
                new_entry = {
                    "profession_name": profession_title,
                    "persona_prompt": persona_prompt,
                }
                self.current_data.append(new_entry)

                # Save progress after every single profession (critical for safety)
                self._save_progress_json(self.current_data)

                # Additional checkpoint every 5 professions
                if (prof_index + 1) % 5 == 0:
                    logger.info(
                        f"💾 Checkpoint: Successfully processed {prof_index + 1}/{total_to_process} professions"
                    )

                # Brief pause to avoid overwhelming the API
                time.sleep(1)

            except Exception as e:
                error_msg = f"❌ Error processing '{profession_title}': {str(e)}"
                logger.error(error_msg)

                # Save progress even after errors
                self._save_progress_json(
                    self.current_data, force_save=True
                )

                # Add error entry to maintain progress tracking
                error_entry = {
                    "profession_name": profession_title,
                    "persona_prompt": f"ERROR: Failed to generate persona - {str(e)}",
                }
                self.current_data.append(error_entry)

                # Continue processing other professions
                logger.info("🔄 Continuing with next profession...")
                continue

    def _process_concurrent(
        self, professions_to_process: List[Tuple[int, str, str]]
    ) -> None:
        """Process professions concurrently using ThreadPoolExecutor."""
        logger.info(
            f"⚡ Processing concurrently with {self.max_workers} workers..."
        )

        completed_count = 0
        total_count = len(professions_to_process)

        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Submit all tasks
            future_to_profession = {
                executor.submit(
                    self._process_single_profession, prof_data
                ): prof_data
                for prof_data in professions_to_process
            }

            # Process completed tasks
            for future in as_completed(future_to_profession):
                prof_data = future_to_profession[future]
                _, profession_title, _ = prof_data

                try:
                    result = future.result()

                    # Thread-safe progress update
                    self._update_progress_safely(result)

                    completed_count += 1
                    logger.info(
                        f"✅ Completed {completed_count}/{total_count}: {profession_title}"
                    )

                    # Checkpoint every 10 completions
                    if completed_count % 10 == 0:
                        logger.info(
                            f"💾 Checkpoint: {completed_count}/{total_count} professions completed"
                        )

                    # Brief pause between API calls to avoid rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(
                        f"❌ Future failed for {profession_title}: {str(e)}"
                    )

                    # Add error entry
                    error_entry = {
                        "profession_name": profession_title,
                        "persona_prompt": f"ERROR: Concurrent processing failed - {str(e)}",
                    }
                    self._update_progress_safely(error_entry)
                    completed_count += 1

        logger.success(
            f"⚡ Concurrent processing completed: {completed_count}/{total_count} professions"
        )

    def process_all_professions(
        self,
        max_rows: Optional[int] = None,
        concurrent: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Process all professions in the input CSV and save results.

        Args:
            max_rows: Maximum number of rows to process, None for no limit
            concurrent: Whether to process professions concurrently
            max_workers: Maximum number of concurrent workers, None for auto-calculated

        This is the main method to run the complete persona generation process.
        """
        mode_str = "concurrent" if concurrent else "sequential"
        logger.info(
            f"🎯 Starting complete profession processing - {mode_str} mode"
        )

        try:
            # Process professions (with optional limit and concurrency)
            results = self.process_professions(
                max_rows=max_rows,
                concurrent=concurrent,
                max_workers=max_workers,
            )

            # Save final results to CSV
            self._save_results(results)

            # Mark processing as complete
            self._mark_processing_complete()

            logger.success(
                f"🎉 Successfully generated {len(results)} profession personas!"
            )

        except KeyboardInterrupt:
            logger.warning("🚨 Processing interrupted by user")
            self._save_progress_json(
                self.current_data, force_save=True
            )
            logger.success("✅ Progress saved before exit")
            raise

        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            self._save_progress_json(
                self.current_data, force_save=True
            )
            logger.success("✅ Progress saved despite error")
            raise

    def resume_from_json(self) -> None:
        """Resume processing from the JSON progress file."""
        if not self.json_progress_file.exists():
            logger.warning("No progress file found to resume from")
            return

        try:
            with open(
                self.json_progress_file, "r", encoding="utf-8"
            ) as f:
                data = json.load(f)

            if data["metadata"]["processing_status"] == "completed":
                logger.info(
                    "✅ Processing already completed according to progress file"
                )
                return

            logger.info(
                f"📂 Resuming from {len(data['professions'])} previously processed professions"
            )
            self.current_data = data["professions"]
            self.processed_count = len(self.current_data)

        except Exception as e:
            logger.error(f"Failed to resume from JSON: {str(e)}")
            raise

    def get_progress_summary(self) -> Dict:
        """Get a summary of current progress."""
        if self.json_progress_file.exists():
            try:
                with open(
                    self.json_progress_file, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                return data["metadata"]
            except Exception:
                pass

        return {
            "total_processed": len(self.current_data),
            "processing_status": "not_started",
            "last_updated": "never",
        }

    def clear_progress(self) -> None:
        """Clear all progress data and start fresh."""
        self.current_data = []
        self.processed_count = 0

        # Remove progress files
        if self.json_progress_file.exists():
            self.json_progress_file.unlink()
            logger.info(
                f"🗑️  Removed progress file: {self.json_progress_file}"
            )

        backup_file = self.json_progress_file.with_suffix(
            ".backup.json"
        )
        if backup_file.exists():
            backup_file.unlink()
            logger.info(f"🗑️  Removed backup file: {backup_file}")

        progress_csv = self.output_file.with_suffix(".progress.csv")
        if progress_csv.exists():
            progress_csv.unlink()
            logger.info(f"🗑️  Removed progress CSV: {progress_csv}")

        logger.success("✅ Progress cleared. Ready to start fresh!")

    def process_sample(
        self, sample_size: int = 5, concurrent: bool = False
    ) -> None:
        """
        Process a small sample of professions for testing.

        Args:
            sample_size: Number of professions to process
            concurrent: Whether to process concurrently
        """
        mode_str = "concurrent" if concurrent else "sequential"
        logger.info(
            f"🧪 Processing sample of {sample_size} professions for testing - {mode_str} mode"
        )

        try:
            results = self.process_professions(
                start_row=0,
                end_row=sample_size,
                concurrent=concurrent,
            )

            # Save with sample suffix
            sample_output = self.output_file.with_suffix(
                ".sample.csv"
            )
            sample_data = results

            df = pd.DataFrame(sample_data)
            df.to_csv(
                sample_output,
                index=False,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )

            logger.success(
                f"🧪 Sample processing complete! Results saved to {sample_output}"
            )

        except Exception as e:
            logger.error(f"❌ Sample processing failed: {str(e)}")
            raise

    def process_limited(
        self,
        limit: int = 20,
        concurrent: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Process a limited number of professions and save results.

        Args:
            limit: Maximum number of professions to process (default: 20)
            concurrent: Whether to process concurrently
            max_workers: Maximum number of concurrent workers, None for auto-calculated
        """
        mode_str = "concurrent" if concurrent else "sequential"
        logger.info(
            f"🎯 Processing limited set of {limit} professions - {mode_str} mode"
        )

        try:
            # Process limited professions
            results = self.process_professions(
                max_rows=limit,
                concurrent=concurrent,
                max_workers=max_workers,
            )

            # Save final results
            self._save_results(results)

            logger.success(
                f"🎉 Successfully generated {len(results)} profession personas!"
            )

        except Exception as e:
            logger.error(f"❌ Limited processing failed: {str(e)}")
            raise

    def preview_result(
        self, output_file: Optional[str] = None, index: int = 0
    ) -> None:
        """
        Preview a generated persona from the results file.

        Args:
            output_file: Path to results file, uses default if None
            index: Index of the profession to preview
        """
        file_path = (
            Path(output_file) if output_file else self.output_file
        )

        try:
            df = pd.read_csv(file_path)

            if len(df) > index:
                sample = df.iloc[index]

                logger.info("🔍 PERSONA PREVIEW:")
                logger.info(
                    f"📋 Profession: {sample['profession_name']}"
                )
                logger.info("📝 Persona Prompt (first 500 chars):")
                logger.info(f"{sample['persona_prompt'][:500]}...")

            else:
                logger.warning(f"No data at index {index}")

        except Exception as e:
            logger.error(f"Error previewing result: {str(e)}")


def main() -> None:
    """Main function to run the persona generation system."""

    # Configuration
    INPUT_FILE = "data.csv"
    OUTPUT_FILE = (
        f"profession_personas_new_10_{str(uuid.uuid4())}.csv"
    )

    # Initialize generator
    generator = ProfessionPersonaGenerator(INPUT_FILE, OUTPUT_FILE)

    # Display system information
    logger.info("🖥️  System Configuration:")
    logger.info(f"   CPU Cores: {os.cpu_count()}")
    logger.info(
        f"   Max Workers: {generator.max_workers} (90% of CPU cores)"
    )
    logger.info("   Processing Options: sequential | concurrent")

    # ===== PROCESSING OPTIONS =====

    # Option 1: Process a small sample for testing (5 professions)
    # generator.process_sample(5, concurrent=False)  # Sequential
    # generator.process_sample(5, concurrent=True)   # Concurrent

    # Option 2: Process a limited number using max_rows parameter
    # Sequential processing (original behavior)
    # generator.process_all_professions(max_rows=10, concurrent=False)

    # Concurrent processing (faster, uses multiple threads)
    # generator.process_all_professions(max_rows=10, concurrent=True)

    # Concurrent with custom worker count
    # generator.process_all_professions(max_rows=10, concurrent=True, max_workers=4)

    # Option 3: Process a limited number using the convenience method
    # generator.process_limited(20, concurrent=False)  # Sequential
    # generator.process_limited(20, concurrent=True)   # Concurrent
    # generator.process_limited(20, concurrent=True, max_workers=6)  # Custom workers

    # Option 4: Process all professions
    # ⚠️  WARNING: This will process the entire dataset!

    # Sequential processing (safe, slower)
    # generator.process_all_professions(concurrent=False)

    # Concurrent processing (faster, but more resource intensive)
    # generator.process_all_professions(concurrent=True)

    # ===== CURRENT EXECUTION =====
    # Clear any existing progress to start fresh with fixed generation
    logger.info("🧹 Clearing previous progress to start fresh...")
    generator.clear_progress()

    # For demonstration, running a small sequential batch with fixed agent configuration
    logger.info(
        "🚀 Running demonstration with fixed agent configuration..."
    )
    generator.process_all_professions(max_rows=20, concurrent=False)

    # Preview a sample result
    generator.preview_result()

    # Show final progress summary
    summary = generator.get_progress_summary()
    logger.info("📊 Final Summary:")
    logger.info(
        f"   Total Processed: {summary['total_processed']} professions"
    )
    logger.info(f"   Status: {summary['processing_status']}")
    logger.info(f"   Last Updated: {summary['last_updated']}")


if __name__ == "__main__":
    main()
