#!/usr/bin/env python3
"""
Automated changelog generation script.

This script generates changelog entries from git commits and pull requests,
following the Keep a Changelog format.
"""

import argparse
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChangelogEntry:
    """Represents a single changelog entry."""
    type: str
    description: str
    pr_number: Optional[int] = None
    commit_hash: str = ""
    breaking: bool = False


class ChangelogGenerator:
    """Generates changelog from git history."""
    
    # Commit type mapping to changelog sections
    TYPE_MAPPING = {
        "feat": "Added",
        "fix": "Fixed", 
        "docs": "Changed",
        "style": "Changed",
        "refactor": "Changed",
        "perf": "Changed",
        "test": "Changed",
        "chore": "Changed",
        "ci": "Changed",
        "build": "Changed",
        "revert": "Fixed",
        "security": "Security",
        "deps": "Changed",
        "breaking": "Breaking Changes"
    }
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        
    def get_git_commits(self, since_tag: Optional[str] = None) -> List[Dict]:
        """Get git commits since a specific tag or all commits."""
        cmd = ["git", "log", "--oneline", "--no-merges"]
        
        if since_tag:
            cmd.append(f"{since_tag}..HEAD")
        else:
            # Get commits from last 30 days if no tag specified
            cmd.extend(["--since", "30 days ago"])
            
        try:
            result = subprocess.run(
                cmd, 
                cwd=self.repo_path, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        commits.append({
                            'hash': parts[0],
                            'message': parts[1]
                        })
            
            return commits
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting git commits: {e}")
            return []
    
    def parse_commit_message(self, message: str) -> Optional[ChangelogEntry]:
        """Parse a commit message into a changelog entry."""
        # Conventional commit pattern: type(scope): description
        pattern = r'^(\w+)(?:\(([^)]+)\))?\s*:\s*(.+)$'
        match = re.match(pattern, message)
        
        if not match:
            # If not conventional commit, try to guess type from keywords
            return self._guess_change_type(message)
            
        commit_type = match.group(1).lower()
        scope = match.group(2)
        description = match.group(3)
        
        # Check for breaking changes
        breaking = "BREAKING CHANGE" in message or "!" in match.group(1)
        
        # Extract PR number if present
        pr_match = re.search(r'#(\d+)', description)
        pr_number = int(pr_match.group(1)) if pr_match else None
        
        # Map commit type to changelog section
        changelog_type = self.TYPE_MAPPING.get(commit_type, "Changed")
        
        if breaking:
            changelog_type = "Breaking Changes"
            
        # Format description
        if scope:
            description = f"**{scope}**: {description}"
            
        return ChangelogEntry(
            type=changelog_type,
            description=description,
            pr_number=pr_number,
            breaking=breaking
        )
    
    def _guess_change_type(self, message: str) -> Optional[ChangelogEntry]:
        """Guess change type from commit message keywords."""
        message_lower = message.lower()
        
        # Security keywords
        security_keywords = ["security", "vulnerability", "cve", "exploit", "patch security"]
        if any(keyword in message_lower for keyword in security_keywords):
            return ChangelogEntry(type="Security", description=message)
            
        # Fix keywords
        fix_keywords = ["fix", "bug", "error", "issue", "patch", "resolve"]
        if any(keyword in message_lower for keyword in fix_keywords):
            return ChangelogEntry(type="Fixed", description=message)
            
        # Feature keywords
        feature_keywords = ["add", "new", "feature", "implement", "introduce"]
        if any(keyword in message_lower for keyword in feature_keywords):
            return ChangelogEntry(type="Added", description=message)
            
        # Default to Changed
        return ChangelogEntry(type="Changed", description=message)
    
    def group_entries(self, entries: List[ChangelogEntry]) -> Dict[str, List[ChangelogEntry]]:
        """Group changelog entries by type."""
        grouped = {}
        
        for entry in entries:
            if entry.type not in grouped:
                grouped[entry.type] = []
            grouped[entry.type].append(entry)
            
        return grouped
    
    def format_changelog(self, grouped_entries: Dict[str, List[ChangelogEntry]], 
                        version: str, date: str = None) -> str:
        """Format changelog entries into markdown."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        changelog = f"## [{version}] - {date}\n\n"
        
        # Define section order
        section_order = [
            "Breaking Changes",
            "Security", 
            "Added",
            "Changed",
            "Fixed",
            "Deprecated",
            "Removed"
        ]
        
        for section in section_order:
            if section in grouped_entries:
                changelog += f"### {section}\n\n"
                
                for entry in grouped_entries[section]:
                    line = f"- {entry.description}"
                    if entry.pr_number:
                        line += f" ([#{entry.pr_number}])"
                    changelog += line + "\n"
                    
                changelog += "\n"
        
        return changelog
    
    def get_latest_tag(self) -> Optional[str]:
        """Get the latest git tag."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def get_next_version(self, current_version: Optional[str] = None, 
                        bump_type: str = "patch") -> str:
        """Calculate next version number."""
        if not current_version:
            current_version = self.get_latest_tag() or "0.0.0"
            
        # Remove 'v' prefix if present
        version = current_version.lstrip('v')
        
        try:
            major, minor, patch = map(int, version.split('.'))
        except ValueError:
            # If version doesn't follow semver, default to 0.1.0
            major, minor, patch = 0, 1, 0
            
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
            
        return f"{major}.{minor}.{patch}"
    
    def update_changelog_file(self, new_content: str, 
                             changelog_path: str = "CHANGELOG.md"):
        """Update the changelog file with new content."""
        changelog_file = self.repo_path / changelog_path
        
        if changelog_file.exists():
            with open(changelog_file, 'r') as f:
                existing_content = f.read()
                
            # Insert new content after the header
            lines = existing_content.split('\n')
            header_end = 0
            
            for i, line in enumerate(lines):
                if line.startswith('## '):
                    header_end = i
                    break
            else:
                # No existing releases, add after title
                for i, line in enumerate(lines):
                    if line.startswith('# '):
                        header_end = i + 1
                        break
                        
            # Insert new content
            new_lines = lines[:header_end] + [''] + new_content.split('\n') + lines[header_end:]
            content = '\n'.join(new_lines)
        else:
            # Create new changelog file
            header = """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
            content = header + new_content
            
        with open(changelog_file, 'w') as f:
            f.write(content)
    
    def generate(self, version: str = None, since_tag: str = None, 
                bump_type: str = "patch", output_file: str = None) -> str:
        """Generate changelog for the specified version."""
        # Get commits
        commits = self.get_git_commits(since_tag)
        
        if not commits:
            print("No commits found for changelog generation")
            return ""
            
        # Parse commits into changelog entries
        entries = []
        for commit in commits:
            entry = self.parse_commit_message(commit['message'])
            if entry:
                entry.commit_hash = commit['hash']
                entries.append(entry)
        
        if not entries:
            print("No valid changelog entries found")
            return ""
            
        # Determine version
        if not version:
            version = self.get_next_version(bump_type=bump_type)
            
        # Group and format entries
        grouped = self.group_entries(entries)
        changelog_content = self.format_changelog(grouped, version)
        
        # Update changelog file
        if output_file:
            self.update_changelog_file(changelog_content, output_file)
        else:
            self.update_changelog_file(changelog_content)
            
        return changelog_content


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate changelog from git commits")
    parser.add_argument("--version", help="Version number for the release")
    parser.add_argument("--since-tag", help="Generate changelog since this tag")
    parser.add_argument("--bump-type", choices=["major", "minor", "patch"], 
                       default="patch", help="Version bump type")
    parser.add_argument("--output", help="Output file path", default="CHANGELOG.md")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print changelog without updating file")
    
    args = parser.parse_args()
    
    generator = ChangelogGenerator()
    
    if args.dry_run:
        # Just print the changelog
        commits = generator.get_git_commits(args.since_tag)
        entries = []
        for commit in commits:
            entry = generator.parse_commit_message(commit['message'])
            if entry:
                entries.append(entry)
                
        if entries:
            version = args.version or generator.get_next_version(bump_type=args.bump_type)
            grouped = generator.group_entries(entries)
            changelog = generator.format_changelog(grouped, version)
            print(changelog)
        else:
            print("No changelog entries found")
    else:
        changelog = generator.generate(
            version=args.version,
            since_tag=args.since_tag,
            bump_type=args.bump_type,
            output_file=args.output
        )
        
        if changelog:
            print(f"Changelog updated in {args.output}")
            print("\nGenerated content:")
            print(changelog)
        else:
            print("No changelog generated")


if __name__ == "__main__":
    main()